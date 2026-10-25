"""Enrich summary after the full availability census."""
from pathlib import Path
import os
import pandas as pd,json,gzip
P=Path('Datastore/DistrictView/derived/initial_eda/')  # generated outputs (local only; may contain transcript text)
P.mkdir(parents=True, exist_ok=True)
RAW=Path(os.environ.get('STAT236_RAW_DIR', 'Datastore/DistrictView/districtview_share_draft')).expanduser()
for filename in ['meta_districtview.parquet'] + [f'transcripts_districtview_part{i}.tar' for i in (1,2,3)]:
    if not (RAW / filename).is_file():
        raise FileNotFoundError(f'Missing {RAW / filename}. See source/analysis/lda_2018_2023/README.md for data setup.')

d=pd.read_parquet(RAW/'meta_districtview.parquet');s=json.loads((P/'summary.json').read_text())
s['empty_strings']={c:int(d[c].astype(str).str.strip().eq('').sum()) for c in d.columns}
s['core_2018_2023_records']=int(d.meeting_date.dt.year.between(2018,2023).sum())
s['core_2018_2023_districts']=int(d.loc[d.meeting_date.dt.year.between(2018,2023),'leaid'].nunique())
s['sample_placeholders']=sum(json.loads(l)['text'].strip()=='<No transcript available>' for l in gzip.open(P/'sample_texts.jsonl.gz','rt'))
s['district_records_quantiles']=d.groupby('leaid').size().quantile([0,.25,.5,.75,.95,1]).to_dict()
a=pd.read_csv(P/'transcript_availability.csv');d=d.merge(a,on='id',how='left',validate='one_to_one')
d['nonplaceholder']=d.text_characters.gt(0)&d.placeholder.eq(False)&d.error.isna()
d['year']=d.meeting_date.dt.year
annual=d.groupby('year',dropna=False).agg(records=('id','size'),nonplaceholder_text=('nonplaceholder','sum'),districts=('leaid','nunique')).reset_index()
annual.to_csv(P/'annual_availability.csv',index=False)
s.update(placeholder_files=int(a.placeholder.sum()),read_errors=int(a.error.notna().sum()),nonplaceholder_files=int(d.nonplaceholder.sum()),annual_availability=annual.fillna('Missing').to_dict('records'),nonplaceholder_with_date=int((d.nonplaceholder&d.meeting_date.notna()).sum()),nonplaceholder_core=int((d.nonplaceholder&d.year.between(2018,2023)).sum()))
s['placeholder_by_archive']=pd.read_csv(P/'transcript_index.csv').merge(a,on='id').groupby('archive').placeholder.sum().to_dict()
(P/'summary.json').write_text(json.dumps(s,indent=2))
