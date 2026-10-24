"""Check corpus accounting and train/test separation without loading text."""
from pathlib import Path
import json
import pandas as pd
import pyarrow.parquet as pq

P=Path(__file__).resolve().parent
p=json.loads((P/'preparation.json').read_text())
m=pd.read_csv(P/'cleaning_manifest.csv',dtype={'id':str,'leaid':str})
assert len(m)==p['metadata_in_period'] and m.id.is_unique
r=m.loc[m.status=='retained'];assert r.clean_sha256.is_unique
c=pd.concat([pq.read_table(f,columns=['id','split','chunk','words','start_seconds','end_seconds']).to_pandas() for f in sorted((P/'chunks').glob('*.parquet'))],ignore_index=True)
assert set(c.id)==set(r.id)
assert c.groupby('id').split.nunique().eq(1).all()
assert c.words.between(200,1199).all()
assert not c.duplicated(['id','chunk']).any()
for split in ['train','test']:
    z=c.loc[c.split==split]
    assert len(z)==p['counts'][split+'_chunks']
    assert int(z.words.sum())==p['counts'][split+'_words']
    assert z.id.nunique()==p['counts'][split+'_videos']
out=dict(checked_records=len(m),checked_chunks=len(c),unique_retained_videos=len(r),train_test_video_overlap=0,missing_start_timestamp_chunks=int(c.start_seconds.isna().sum()),reversed_timestamp_chunks=int((c.end_seconds<c.start_seconds).sum()),accounting_checks='passed')
(P/'verification.json').write_text(json.dumps(out,indent=2));print(out)
