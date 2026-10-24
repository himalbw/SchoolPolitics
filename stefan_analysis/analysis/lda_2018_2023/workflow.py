"""Full-corpus, streaming DistrictView cleaning and LDA comparison.

The notebook calls these functions. No raw file is modified. Outputs are cached
and checkpoints are written per completed training shard.
"""
from pathlib import Path
import os,re,json,gzip,tarfile,hashlib,time,bisect,collections,heapq
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS,CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib,sklearn

ROOT=Path(__file__).resolve().parent
RAW=Path(os.environ.get('STAT236_RAW_DIR', Path(__file__).resolve().parents[2] / 'data' / 'raw')).expanduser().resolve()
SEED=236
TOPICS=[10,20,30]
CHUNK=1000
MIN_WORDS=200
SHARDS=32
BATCH=1024
VOCAB_MAX=12000
# Fixed symmetric priors across K: changing K does not change each alpha/eta.
ALPHA=.1
ETA=.01
STOP=sorted((set(ENGLISH_STOP_WORDS)-{'no','not','nor','never'})|set('um uh erm hmm yeah okay ok ah oh thank thanks ll ve don doesn didn isn couldn shouldn wouldn haven hasn wasn weren ain'.split()))
TS=re.compile(r'\[(\d+):(\d{2}):(\d{2})(?:\.\d+)?\]')
TAG=re.compile(r'\[(?!\d+:\d{2}:\d{2})[^\]\n]{0,100}\]')
WORD=re.compile(r"[a-z]+(?:'[a-z]+)?")
CONTRACTIONS={"can't":"cannot","won't":"will not","shan't":"shall not","ain't":"is not","it's":"it is","that's":"that is","there's":"there is","here's":"here is","what's":"what is","he's":"he is","she's":"she is","let's":"let us","i'm":"i am","gonna":"going to","wanna":"want to"}
EXPAND=re.compile(r'\b(?:'+ '|'.join(re.escape(x) for x in CONTRACTIONS)+r")\b|n't\b|'re\b|'ve\b|'ll\b|'d\b")
SUFFIX={"n't":' not',"'re":' are',"'ve":' have',"'ll":' will',"'d":' would'}
schema=pa.schema([('id',pa.string()),('leaid',pa.string()),('year',pa.int16()),('chunk',pa.int32()),('start_seconds',pa.float64()),('end_seconds',pa.float64()),('words',pa.int32()),('split',pa.string()),('text',pa.string())])

def save_json(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2,default=str))

def clean_text(raw):
    text=TAG.sub(' ',raw.lower().replace('’',"'"))
    return EXPAND.sub(lambda m:CONTRACTIONS.get(m[0],SUFFIX.get(m[0])),text)

def prepare():
    """Read every dated 2018–2023 archive entry; write full cleaned chunks."""
    if (ROOT/'preparation.json').exists():return json.loads((ROOT/'preparation.json').read_text())
    (ROOT/'chunks').mkdir(exist_ok=True)
    for filename in ['meta_districtview.parquet'] + [f'transcripts_districtview_part{i}.tar' for i in (1,2,3)]:
        if not (RAW / filename).is_file():
            raise FileNotFoundError(f'Missing {RAW / filename}. See stefan_analysis/README.md for data setup.')
    d=pd.read_parquet(RAW/'meta_districtview.parquet')
    d=d.loc[d.meeting_date.dt.year.between(2018,2023)].copy()
    d['year']=d.meeting_date.dt.year
    meta=d.set_index('id')[['leaid','year','title','meeting_date']].to_dict('index')
    writers={};buffers=collections.defaultdict(list);audit=[];seen={};found=set();counts=collections.Counter();started=time.time()
    def flush(shard):
        if not buffers[shard]:return
        if shard not in writers:writers[shard]=pq.ParquetWriter(ROOT/'chunks'/f'shard_{shard:02}.parquet',schema,compression='zstd')
        writers[shard].write_table(pa.Table.from_pylist(buffers[shard],schema=schema));buffers[shard].clear()
    for archive in sorted(RAW.glob('*.tar')):
        with tarfile.open(archive,mode='r|') as tar:
            for m in tar:
                if not m.isfile():continue
                vid=Path(m.name).name.removesuffix('.txt.gz').partition('__')[2]
                if vid not in meta:continue
                found.add(vid);record=dict(id=vid,archive=archive.name,member=m.name,**meta[vid]);record['chunks']=0
                try:raw=gzip.decompress(tar.extractfile(m).read()).decode('utf-8')
                except Exception as e:record.update(status='decode_error',error=str(e));audit.append(record);continue
                if raw.strip()=='<No transcript available>':record['status']='placeholder';audit.append(record);continue
                text=clean_text(raw);matches=list(WORD.finditer(text));tokens=[x.group() for x in matches]
                record['words']=len(tokens)
                if raw.isascii():record['ascii_letter_fraction']=1.
                else:
                    ascii_letters=len(re.findall('[A-Za-z]',raw));other=sum(c.isalpha() for c in re.findall(r'[^\x00-\x7F]',raw))
                    record['ascii_letter_fraction']=ascii_letters/max(1,ascii_letters+other)
                # Flag-only: no automatic exclusion of potentially substantive special meetings.
                record['title_review_flag']=bool(re.search(r'ceremony|concert|graduation|holiday|announcement|recognition|promotional',record['title'],re.I))
                if len(tokens)<MIN_WORDS:record['status']='under_200_words';audit.append(record);continue
                digest=hashlib.sha256(' '.join(tokens).encode()).hexdigest();record['clean_sha256']=digest
                if digest in seen:record.update(status='exact_duplicate',duplicate_of=seen[digest]);audit.append(record);continue
                seen[digest]=vid
                # Hash of content prevents exact-content leakage, with common ~90/10 split for all K.
                split='test' if int(digest[:8],16)%10==0 else 'train'
                shard=int(hashlib.sha256((str(SEED)+vid).encode()).hexdigest()[:8],16)%SHARDS
                times=[(m.start(),3600*int(m[1])+60*int(m[2])+int(m[3])) for m in TS.finditer(text)]
                positions=[x[0] for x in times]
                def timestamp(pos):
                    j=bisect.bisect_right(positions,pos)-1
                    return float(times[j][1]) if j>=0 else None
                # Retain the final short tail by merging it into the preceding chunk.
                starts=list(range(0,len(tokens),CHUNK))
                if len(starts)>1 and len(tokens)-starts[-1]<MIN_WORDS:starts.pop()
                for i,start in enumerate(starts):
                    end=starts[i+1] if i+1<len(starts) else len(tokens)
                    buffers[shard].append(dict(id=vid,leaid=record['leaid'],year=int(record['year']),chunk=i,start_seconds=timestamp(matches[start].start()),end_seconds=timestamp(matches[end-1].start()),words=end-start,split=split,text=' '.join(tokens[start:end])))
                    counts[split+'_chunks']+=1;counts[split+'_words']+=end-start
                    if len(buffers[shard])>=128:flush(shard)
                record.update(status='retained',split=split,chunks=len(starts));counts[split+'_videos']+=1;audit.append(record)
                if len(audit)%5000==0:print('Cleaned entries',len(audit),'elapsed min',round((time.time()-started)/60,1),flush=True)
        print('Finished archive',archive.name,flush=True)
    for shard in buffers:flush(shard)
    for w in writers.values():w.close()
    for vid in set(meta)-found:audit.append(dict(id=vid,**meta[vid],status='missing_archive',chunks=0))
    a=pd.DataFrame(audit);a.to_csv(ROOT/'cleaning_manifest.csv',index=False)
    a.groupby(['year','status']).size().unstack(fill_value=0).to_csv(ROOT/'cleaning_by_year.csv')
    result=dict(metadata_in_period=len(d),statuses=a.status.value_counts().to_dict(),counts=dict(counts),title_flags_retained=int(a.loc[a.status=='retained','title_review_flag'].sum()),low_ascii_flags_retained=int((a.loc[a.status=='retained','ascii_letter_fraction']<.95).sum()),elapsed_seconds=time.time()-started,seed=SEED,chunk_words=CHUNK,min_words=MIN_WORDS,shards=SHARDS,split_rule='content SHA256 prefix modulo 10; 0 = test, otherwise train; exact duplicates removed first',stopwords=STOP)
    save_json('preparation.json',result);return result

def iter_frames(shard=None):
    paths=sorted((ROOT/'chunks').glob('*.parquet')) if shard is None else [ROOT/'chunks'/f'shard_{shard:02}.parquet']
    for p in paths:
        for b in pq.ParquetFile(p).iter_batches(batch_size=BATCH):yield b.to_pandas()

def vocabulary():
    if (ROOT/'vectorizer.joblib').exists():return joblib.load(ROOT/'vectorizer.joblib')
    base=CountVectorizer(stop_words=STOP,token_pattern=r"(?u)\b[a-z][a-z]+\b")
    analyzer=base.build_analyzer();freq=collections.Counter();docfreq=collections.Counter();n=0
    for i,frame in enumerate(iter_frames()):
        for text in frame.loc[frame.split=='train','text']:
            words=analyzer(text);freq.update(words);docfreq.update(set(words));n+=1
        if i%100==0:print('Vocabulary chunks',n,flush=True)
    allowed=[w for w,c in freq.most_common() if docfreq[w]>=25 and docfreq[w]<=.8*n][:VOCAB_MAX]
    vec=CountVectorizer(stop_words=STOP,token_pattern=r"(?u)\b[a-z][a-z]+\b",vocabulary={w:i for i,w in enumerate(sorted(allowed))})
    vec.fit([' '.join(allowed[:2])]);joblib.dump(vec,ROOT/'vectorizer.joblib')
    pd.DataFrame([{'term':w,'term_count':freq[w],'chunk_document_frequency':docfreq[w]} for w in vec.get_feature_names_out()]).to_csv(ROOT/'vocabulary.csv',index=False)
    save_json('vocabulary_info.json',dict(train_chunks=n,vocabulary=len(allowed),min_df=25,max_df=.8,max_features=VOCAB_MAX));return vec

def fit_models():
    """One complete online pass over all nonempty training chunks, all K together."""
    prep=prepare();vec=vocabulary();checkpoint=ROOT/'training_checkpoint.joblib'
    if (ROOT/'training.json').exists():return json.loads((ROOT/'training.json').read_text())
    if checkpoint.exists():state=joblib.load(checkpoint);models=state['models'];done=state['done'];timings=state['timings'];n_seen=state['n_seen'];empty=state['empty']
    else:
        models={k:LatentDirichletAllocation(n_components=k,doc_topic_prior=ALPHA,topic_word_prior=ETA,learning_method='online',learning_decay=.7,learning_offset=64,total_samples=prep['counts']['train_chunks'],batch_size=BATCH,max_doc_update_iter=20,mean_change_tol=.001,random_state=SEED,n_jobs=4) for k in TOPICS}
        done=[];timings={k:0. for k in TOPICS};n_seen=0;empty=0
    order=np.random.default_rng(SEED).permutation(SHARDS)
    for shard in order:
        shard=int(shard)
        if shard in done:continue
        for frame in iter_frames(shard):
            frame=frame.loc[frame.split=='train'].sample(frac=1,random_state=SEED)
            if frame.empty:continue
            X=vec.transform(frame.text);keep=X.getnnz(axis=1)>0;empty+=int((~keep).sum());X=X[keep]
            if X.shape[0]==0:continue
            for k,model in models.items():
                start=time.time();model.partial_fit(X);timings[k]+=time.time()-start
            n_seen+=X.shape[0]
        done.append(shard);joblib.dump(dict(models=models,done=done,timings=timings,n_seen=n_seen,empty=empty),checkpoint)
        print('Training shards',len(done),'/',SHARDS,'chunks',n_seen,'seconds by K',timings,flush=True)
    for k,m in models.items():joblib.dump(m,ROOT/f'lda_k{k}.joblib')
    result=dict(topic_counts=TOPICS,train_chunks_fitted=n_seen,empty_train_chunks=empty,passes=1,seed=SEED,alpha=ALPHA,eta=ETA,max_doc_update_iter=20,learning_decay=.7,learning_offset=64,n_jobs=4,training_seconds=timings,sklearn_version=sklearn.__version__,note='Full training corpus online pass; no convergence or seed stability claim. Shared held-out videos are never used to fit vocabulary or model.')
    save_json('training.json',result);return result

def evaluate():
    """Evaluate every nonempty test chunk by held-out token completion.

    Half the counts infer theta; the other half score predictive likelihood.
    Topic coherence uses binary co-occurrence across full test chunks.
    """
    if (ROOT/'comparison.csv').exists():return pd.read_csv(ROOT/'comparison.csv')
    vec=joblib.load(ROOT/'vectorizer.joblib');words=vec.get_feature_names_out()
    models={k:joblib.load(ROOT/f'lda_k{k}.joblib') for k in TOPICS}
    betas={k:m.components_/m.components_.sum(axis=1,keepdims=True) for k,m in models.items()}
    tops={k:np.argsort(b,axis=1)[:,-10:][:,::-1] for k,b in betas.items()}
    union=np.unique(np.concatenate([x.ravel() for x in tops.values()]))
    lookup={int(w):i for i,w in enumerate(union)};cooccur=np.zeros((len(union),len(union)),dtype=np.int64)
    rng=np.random.default_rng(SEED+1);ll={k:0. for k in TOPICS};score_tokens=0;test_chunks=0;scored_chunks=0;empty_chunks=0;unscorable=0
    heaps={k:[[] for _ in range(k)] for k in TOPICS};serial=0
    for fno,frame in enumerate(iter_frames()):
        frame=frame.loc[frame.split=='test'].reset_index(drop=True)
        if frame.empty:continue
        X=vec.transform(frame.text).tocsr();test_chunks+=X.shape[0]
        B=X[:,union].copy();B.data[:]=1;cooccur+=(B.T@B).toarray()
        observed=X.copy();observed.data=rng.binomial(X.data,.5);observed.eliminate_zeros()
        target=X-observed;target.eliminate_zeros()
        keep=(np.asarray(observed.sum(axis=1)).ravel()>0)&(np.asarray(target.sum(axis=1)).ravel()>0)
        empty_chunks+=int((X.getnnz(axis=1)==0).sum());unscorable+=int((~keep).sum())
        frame=frame.loc[keep].reset_index(drop=True);observed=observed[keep];target=target[keep]
        if not len(frame):continue
        scored_chunks+=len(frame);score_tokens+=int(target.sum());rows,cols=target.nonzero()
        for k,m in models.items():
            theta=m.transform(observed);beta=betas[k]
            for start in range(0,len(rows),50000):
                end=start+50000
                probs=np.einsum('ij,ij->i',theta[rows[start:end]],beta[:,cols[start:end]].T)
                ll[k]+=float(np.dot(target.data[start:end],np.log(np.maximum(probs,1e-300))))
            # Representatives use only the observed half, consistent across models.
            for topic in range(k):
                candidates=np.argsort(theta[:,topic])[-3:]
                for j in candidates:
                    row=frame.iloc[int(j)];serial+=1
                    item=(float(theta[j,topic]),serial,dict(id=row.id,chunk=int(row.chunk),start_seconds=row.start_seconds,end_seconds=row.end_seconds,leaid=row.leaid,year=int(row.year),excerpt=row.text[:1500]))
                    h=heaps[k][topic]
                    if len(h)<24:heapq.heappush(h,item)
                    elif item[0]>h[0][0]:heapq.heapreplace(h,item)
        if fno%30==0:print('Evaluated test chunks',test_chunks,flush=True)
    freq=np.diag(cooccur);results=[];topic_rows=[];examples={}
    for k in TOPICS:
        scores=[];examples[k]=[]
        for topic,indices in enumerate(tops[k]):
            ix=[lookup[int(w)] for w in indices];pairs=[]
            for i in range(len(ix)):
                for j in range(i):
                    both=cooccur[ix[i],ix[j]]
                    if not both:pairs.append(-1.);continue
                    p=both/test_chunks
                    pairs.append(float(np.log(p/((freq[ix[i]]/test_chunks)*(freq[ix[j]]/test_chunks)))/(-np.log(p))) if p<1 else 0.)
            coherence=float(np.mean(pairs));scores.append(coherence)
            topic_rows.append(dict(k=k,topic=topic+1,npmi=coherence,top_terms=', '.join(words[indices])))
            selected=[];seen=set()
            for weight,_,info in sorted(heaps[k][topic],reverse=True):
                if info['id'] in seen:continue
                seen.add(info['id']);selected.append(dict(observed_half_topic_weight=weight,**info))
                if len(selected)==3:break
            examples[k].append(dict(topic=topic+1,terms=list(words[indices]),npmi=coherence,examples=selected))
        results.append(dict(k=k,completion_perplexity=float(np.exp(-ll[k]/score_tokens)),heldout_log_likelihood=ll[k],scored_tokens=score_tokens,npmi_mean=float(np.mean(scores)),npmi_median=float(np.median(scores)),top10_diversity=len(np.unique(tops[k]))/(k*10)))
    pd.DataFrame(results).to_csv(ROOT/'comparison.csv',index=False);pd.DataFrame(topic_rows).to_csv(ROOT/'topics.csv',index=False)
    save_json('topic_examples.json',examples)
    save_json('evaluation.json',dict(test_chunks=test_chunks,scored_chunks=scored_chunks,empty_chunks=empty_chunks,unscorable_chunks=unscorable,scored_tokens=score_tokens,token_split_seed=SEED+1,token_observation_fraction=.5,coherence='Mean normalized PMI over 45 pairs of top 10 words, using held-out chunk co-occurrence. Zero pairs assigned -1.',representatives='Mixtures inferred from observed half of held-out word counts. Excerpts show beginning of full cleaned chunk; timestamp is approximate caption-line start.'))
    return pd.DataFrame(results)

if __name__=='__main__':
    import sys
    stage=sys.argv[1] if len(sys.argv)>1 else 'prepare'
    print(json.dumps({'prepare':prepare,'vocabulary':vocabulary,'fit':fit_models,'evaluate':evaluate}[stage](),default=str,indent=2),flush=True)
