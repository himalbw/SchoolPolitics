"""Full gzip readability and placeholder census; no extraction to disk."""
from pathlib import Path
import os
import tarfile,gzip,json
import pandas as pd
RAW=Path(os.environ.get('STAT236_RAW_DIR', Path(__file__).resolve().parents[2] / 'data' / 'raw')).expanduser().resolve()
for filename in ['meta_districtview.parquet'] + [f'transcripts_districtview_part{i}.tar' for i in (1,2,3)]:
    if not (RAW / filename).is_file():
        raise FileNotFoundError(f'Missing {RAW / filename}. See stefan_analysis/README.md for data setup.')

OUT=Path(__file__).resolve().parent
rows=[]
for p in sorted(RAW.glob('*.tar')):
    with tarfile.open(p,mode='r|') as t:
        for m in t:
            if not m.isfile():continue
            vid=Path(m.name).name.removesuffix('.txt.gz').partition('__')[2]
            try:
                text=gzip.decompress(t.extractfile(m).read()).decode('utf-8')
                rows.append((vid,text.strip()=='<No transcript available>',len(text),None))
            except Exception as e:rows.append((vid,False,0,str(e)))
    print('Checked',p.name,len(rows),flush=True)
pd.DataFrame(rows,columns=['id','placeholder','text_characters','error']).to_csv(OUT/'transcript_availability.csv',index=False)
print('Finished',flush=True)
