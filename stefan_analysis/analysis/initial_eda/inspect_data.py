"""Reproducible initial audit: full metadata/archive index; seeded text sample."""
from pathlib import Path
import os
import tarfile, gzip, re, json, hashlib, collections
import numpy as np
import pandas as pd

RAW = Path(os.environ.get('STAT236_RAW_DIR', Path(__file__).resolve().parents[2] / 'data' / 'raw')).expanduser().resolve()
for filename in ['meta_districtview.parquet'] + [f'transcripts_districtview_part{i}.tar' for i in (1,2,3)]:
    if not (RAW / filename).is_file():
        raise FileNotFoundError(f'Missing {RAW / filename}. See stefan_analysis/README.md for data setup.')

OUT = Path(__file__).resolve().parent
df = pd.read_parquet(RAW/'meta_districtview.parquet')
df['year'] = df.meeting_date.dt.year
df['upload_date'] = pd.to_datetime(df.published_at, utc=True, errors='coerce').dt.tz_localize(None)
rows=[]; members={}
for path in sorted(RAW.glob('*.tar')):
    with tarfile.open(path) as t:
        ms=[m for m in t.getmembers() if m.isfile()]
        members[path.name]=ms
        for m in ms:
            stem=Path(m.name).name.removesuffix('.txt.gz')
            channel, sep, vid=stem.partition('__')
            rows.append(dict(archive=path.name,member=m.name,id=vid if sep else None,archive_channel=channel,compressed_bytes=m.size))
    print('Indexed',path.name,len(ms),flush=True)
idx=pd.DataFrame(rows)
idx.to_csv(OUT/'transcript_index.csv',index=False)
df['has_transcript']=df.id.isin(idx.id)
missing=df.isna().sum().rename('missing').to_frame();missing['percent']=100*missing['missing']/len(df)
missing.to_csv(OUT/'missingness.csv')
annual=df.groupby('year',dropna=False).agg(records=('id','size'),transcripts=('has_transcript','sum'),districts=('leaid','nunique'),channels=('channel_id','nunique')).reset_index()
annual.to_csv(OUT/'annual_coverage.csv',index=False)
districts=df.groupby('leaid',dropna=False).agg(records=('id','size'),transcripts=('has_transcript','sum'),first=('meeting_date','min'),last=('meeting_date','max')).sort_values('records',ascending=False)
districts.to_csv(OUT/'district_coverage.csv')
df.loc[~df.has_transcript,['id','title','meeting_date','leaid']].to_csv(OUT/'missing_transcripts.csv',index=False)
# Simple random sample of archive entries; all text statistics below are sample estimates.
rng=np.random.default_rng(236)
sample=idx.iloc[rng.choice(len(idx),min(1500,len(idx)),replace=False)]
selected=set(sample.member); texts={}; stats=[]; errors=[]
ts=re.compile(r'^\[\d+:\d{2}:\d{2}(?:\.\d+)?\]\s*',re.M)
token=re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
for archive,ms in members.items():
    with tarfile.open(RAW/archive) as t:
        for m in ms:
            if m.name not in selected: continue
            try:
                raw=gzip.decompress(t.extractfile(m).read()).decode('utf-8')
                clean=ts.sub('',raw); words=token.findall(clean.lower()); lines=clean.splitlines()
                vid=Path(m.name).name.removesuffix('.txt.gz').partition('__')[2]
                texts[vid]=clean
                stats.append(dict(id=vid,words=len(words),characters=len(clean),lines=len(lines),timestamp_lines=len(ts.findall(raw)),unique_word_ratio=len(set(words))/max(1,len(words)),adjacent_repeated_lines=sum(a.strip()==b.strip() for a,b in zip(lines,lines[1:])),sha256=hashlib.sha256(clean.encode()).hexdigest()))
            except Exception as e: errors.append({'member':m.name,'error':str(e)})
    print('Sampled',archive,flush=True)
s=pd.DataFrame(stats).merge(df[['id','year','leaid','duration']],on='id',how='left',validate='one_to_one')
s.to_csv(OUT/'text_sample_metrics.csv',index=False)
# Keep local sample for reproducible pilot, no full extraction.
with gzip.open(OUT/'sample_texts.jsonl.gz','wt') as f:
    for vid,txt in texts.items(): f.write(json.dumps({'id':vid,'text':txt})+'\n')
q=s.words.quantile([0,.05,.25,.5,.75,.95,1]).to_dict()
duration=pd.to_timedelta(df.duration,errors='coerce').dt.total_seconds()/60
merged=idx.merge(df[['id','channel_id']],on='id',how='inner')
summ={
 'metadata_rows':len(df),'unique_ids':df.id.nunique(),'districts':df.leaid.nunique(),'channels':df.channel_id.nunique(),
 'archives':idx.groupby('archive').size().to_dict(),'archive_bytes':{p.name:p.stat().st_size for p in RAW.glob('*.tar')},
 'transcript_entries':len(idx),'unique_transcript_ids':idx.id.nunique(),'duplicate_transcript_ids':int(idx.id.duplicated().sum()),
 'metadata_without_transcripts':int((~df.has_transcript).sum()),'transcripts_without_metadata':int((~idx.id.isin(df.id)).sum()),
 'channel_mismatches':int((merged.archive_channel!=merged.channel_id).sum()),
 'date_min':str(df.meeting_date.min()),'date_max':str(df.meeting_date.max()),
 'upload_date_equal_meeting_date':int((df.meeting_date.dt.normalize()==df.upload_date.dt.normalize()).sum()),
 'meeting_after_upload_over7days':int(((df.meeting_date-df.upload_date).dt.total_seconds()>7*86400).sum()),
 'duration_minutes_quantiles':duration.quantile([0,.05,.5,.95,1]).to_dict(),
 'sample_n':len(s),'sample_seed':236,'sample_errors':errors,'sample_word_quantiles':q,
 'sample_under200_words':int((s.words<200).sum()),'sample_over50000_words':int((s.words>50000).sum()),
 'sample_exact_duplicate_texts':int(s.sha256.duplicated().sum()),
 'sample_timestamp_line_fraction':float(s.timestamp_lines.sum()/s.lines.sum()),
 'sample_adjacent_repeat_fraction':float(s.adjacent_repeated_lines.sum()/s.lines.sum()),
 'top10_district_record_share':float(districts.head(10).records.sum()/len(df)),
 'top_words':collections.Counter(w for txt in texts.values() for w in token.findall(txt.lower())).most_common(40),
 'annual':annual.fillna('Missing').to_dict('records'),
 'missingness':missing.loc[df.columns[:17]].reset_index().rename(columns={'index':'field'}).to_dict('records')}
(OUT/'summary.json').write_text(json.dumps(summ,indent=2,default=str))
print(json.dumps(summ,indent=2,default=str),flush=True)
