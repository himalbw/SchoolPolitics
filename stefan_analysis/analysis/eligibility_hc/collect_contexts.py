from pathlib import Path
import pandas as pd, pyarrow.parquet as pq, re
import workflow
ROOT=Path(__file__).resolve().parent
m=pd.read_csv(ROOT/'analysis_videos.csv',dtype={'id':str}).set_index('id');allowed=set(m.index)
terms=['covet','coven','principles','eighty','thousand','foreign','esser','shortages','inflation']
counts={};out=[]
for f in sorted((workflow.LDA/'chunks').glob('*.parquet')):
 for batch in pq.ParquetFile(f).iter_batches(batch_size=2048,columns=['id','year','chunk','text']):
  df=batch.to_pandas();df=df[df.id.isin(allowed)]
  for row in df.itertuples():
   period='early' if row.year<2020 else 'late'
   for term in terms:
    key=(term,period)
    if len(counts.get(key,set()))>=2 or row.id in counts.get(key,set()):continue
    match=re.search(r'\b'+term+r'\b',row.text)
    if match:
     counts.setdefault(key,set()).add(row.id)
     out.append({'word':term,'period':period,'id':row.id,'year':row.year,'chunk':row.chunk,'title':m.loc[row.id,'title'],'context':row.text[max(0,match.start()-180):match.end()+240]})
 if all(len(counts.get((t,p),set()))>=2 for t in terms for p in ['early','late']):break
 print(f.name,len(out),flush=True)
pd.DataFrame(out).to_csv(ROOT/'word_context_examples.csv',index=False)
