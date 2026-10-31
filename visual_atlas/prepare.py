"""Rebuild full-cohort v2 phrase counts without modifying source data."""
from pathlib import Path
import os, json, gzip, re, hashlib, time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
ROOT=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('SCHOOL_BOARD_PROJECT',str(ROOT/'local_data'/'School Board Project')))
V2=PROJECT/'data/modeling_v2_2018_2023'
OLD=PROJECT/'analysis/eligibility_hc_2018_2019_vs_2022_2023/cleaning_followup'
DEFS=json.loads((ROOT/'phrase_definitions.json').read_text())
RXS={k:re.compile(r'\b(?:'+pattern+r')\b') for k,pattern in DEFS.items()}
THEMES=list(DEFS)

def scan(row):
    vid,path=row
    with gzip.open(V2/path,'rt') as f: tokens=f.read().split()
    # Fixed 1,000-core-token blocks, small final block merged, shared across themes.
    chunks=[tokens[i:i+1000] for i in range(0,len(tokens),1000)]
    if len(chunks)>1 and len(chunks[-1])<200: chunks[-2].extend(chunks.pop())
    out={'id':vid,'words_all':len(tokens),'words_no_covid':0}
    out.update({k+'_'+m:0 for k in THEMES for m in ['all','no_covid']})
    examples=[]
    for ci,chunk in enumerate(chunks):
        text=' '.join(chunk).replace("'",'').replace('’','')
        # Independent searches preserve legitimate overlaps between subject dictionaries.
        hits={k:list(rx.finditer(text)) for k,rx in RXS.items()}
        counts={k:len(h) for k,h in hits.items()}
        keep=counts['covid_explicit']==0
        if keep: out['words_no_covid']+=len(chunk)
        for k,n in counts.items():
            out[k+'_all']+=n
            if keep:out[k+'_no_covid']+=n
        # Deterministic audit sample per recording; not a validation set.
        for k in ['book_restrictions','parents_rights']:
            if hits[k] and not any(e['theme']==k for e in examples):
                h=hits[k][0]
                examples.append(dict(id=vid,theme=k,chunk=ci,match=h.group(),context=text[max(0,h.start()-180):h.end()+220]))
    return out,examples

def main():
    (ROOT/'tables').mkdir(exist_ok=True)
    m=pd.read_parquet(V2/'model_manifest.parquet')
    m=m[m.include_primary_lda.eq(True)].copy()
    m['leaid']=m.leaid.astype(str).str.replace(r'\.0$','',regex=True).str.zfill(7)
    m['date']=pd.to_datetime(m.meeting_date,errors='coerce')
    m['quarter']=m.date.dt.to_period('Q').astype(str)
    def conflict(r):
        years=set(re.findall(r'\b(?:19|20)\d{2}\b',str(r.title)))
        return len(years)==1 and int(next(iter(years)))!=int(r.year)
    m['title_year_conflict']=m.apply(conflict,axis=1)
    m[['id','leaid','year','date','quarter','title','title_year_conflict','core_tokens_path']].to_parquet(ROOT/'tables/cohort.parquet',index=False)
    dest=ROOT/'tables/phrase_counts_v2.parquet'
    if not dest.exists():
        start=time.time();rows=[];examples=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for i,(r,e) in enumerate(pool.map(scan,m[['id','core_tokens_path']].itertuples(index=False,name=None),chunksize=40),1):
                rows.append(r);examples.extend(e)
                if i%2000==0:print(f'{i}/{len(m)} recordings counted; {time.time()-start:.0f}s',flush=True)
        pd.DataFrame(rows).to_parquet(dest,index=False)
        pd.DataFrame(examples).to_csv(ROOT/'tables/phrase_context_review.csv',index=False)
    d=m.merge(pd.read_parquet(dest),on='id',validate='one_to_one')
    assert len(d)==83366 and not d.id.duplicated().any()
    assert (d.words_no_covid<=d.words_all).all()
    d[d.title_year_conflict].to_csv(ROOT/'tables/date_conflicts.csv',index=False)
    (ROOT/'tables/preparation.json').write_text(json.dumps({'recordings':len(d),'title_year_conflicts':int(d.title_year_conflict.sum()),'definitions':DEFS,'source_manifest_sha256':hashlib.sha256((V2/'model_manifest.parquet').read_bytes()).hexdigest(),'chunk_rule':'1000 core tokens; tail under 200 merged; may differ from original preprocessing chunk boundaries','denominator':'all core tokens including stopwords and numeric tokens; not comparable in level with earlier alphabetic-token denominator','date_rule':'new substantive panels exclude titles with exactly one explicit four-digit year that disagrees with recorded meeting year; not a full date validation','covid_rule':'exclude entire chunks matching explicit candidate COVID lexicon; not causal adjustment'},indent=2))
    print('PREPARATION COMPLETE',len(d),flush=True)

if __name__=='__main__':main()
