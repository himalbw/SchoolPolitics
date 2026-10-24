"""Small diagnostic LDA run, not the final research model. Run after inspect_data.py."""
from pathlib import Path
import gzip,json,re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
import sklearn

OUT=Path(__file__).resolve().parent
records=[json.loads(line) for line in gzip.open(OUT/'sample_texts.jsonl.gz','rt')]
rng=np.random.default_rng(236)
# Reject very short text, then sample meetings equally; cap three chunks per meeting.
records=[r for r in records if len(r['text'].split())>=200]
records=[records[i] for i in rng.choice(len(records),min(600,len(records)),replace=False)]
train_ids,test_ids=train_test_split([r['id'] for r in records],test_size=.2,random_state=236)
train_set=set(train_ids); chunks=[]
for r in records:
    words=re.findall(r"[a-z]+(?:'[a-z]+)?",r['text'].lower())
    blocks=[words[i:i+1000] for i in range(0,len(words),1000) if len(words[i:i+1000])>=200]
    chosen=sorted(rng.choice(len(blocks),min(3,len(blocks)),replace=False)) if blocks else []
    for i in chosen:chunks.append({'id':r['id'],'chunk':int(i),'text':' '.join(blocks[i]),'train':r['id'] in train_set})
c=pd.DataFrame(chunks)
stop=sorted(set(ENGLISH_STOP_WORDS)|set('um uh yeah okay ok gonna wanna you know like really just actually sort kind thank thanks'.split()))
vec=CountVectorizer(stop_words=stop,min_df=5,max_df=.8,max_features=8000,token_pattern=r'(?u)\b[a-z][a-z]+\b')
X=vec.fit_transform(c.loc[c.train,'text']); H=vec.transform(c.loc[~c.train,'text'])
lda=LatentDirichletAllocation(n_components=12,learning_method='online',max_iter=8,batch_size=128,max_doc_update_iter=30,random_state=236,n_jobs=1)
lda.fit(X)
allX=vec.transform(c.text); theta=lda.transform(allX); words=vec.get_feature_names_out();topics=[]
for k,w in enumerate(lda.components_):
    order=theta[:,k].argsort()[::-1];examples=[];seen=set()
    for i in order:
        row=c.iloc[i]
        if row.id in seen:continue
        seen.add(row.id);examples.append({'id':row.id,'chunk':int(row.chunk),'weight':float(theta[i,k]),'excerpt':row.text[:900]})
        if len(examples)==3:break
    topics.append({'topic':k+1,'terms':list(words[w.argsort()[-15:][::-1]]),'examples':examples})
results={'purpose':'Diagnostic pilot, not validated topics or population estimates','meetings':len(records),'chunks':len(c),'train_meetings':len(train_ids),'test_meetings':len(test_ids),'train_chunks':int(c.train.sum()),'test_chunks':int((~c.train).sum()),'vocabulary':len(words),'k':12,'seed':236,'iterations':8,'sklearn_version':sklearn.__version__,'heldout_perplexity':float(lda.perplexity(H[H.getnnz(axis=1)>0])),'heldout_zero_vectors':int((H.getnnz(axis=1)==0).sum()),'topics':topics}
(OUT/'pilot_results.json').write_text(json.dumps(results,indent=2))
c.drop(columns='text').assign(top_topic=theta.argmax(axis=1)+1).to_csv(OUT/'pilot_chunks.csv',index=False)
print(json.dumps({k:v for k,v in results.items() if k!='topics'},indent=2),flush=True)
for item in topics:print(item['topic'],', '.join(item['terms']),flush=True)
