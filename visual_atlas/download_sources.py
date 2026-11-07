"""Fetch public inputs; cached local files make ordinary notebook runs offline."""
from pathlib import Path
import requests,zipfile,io,hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parent;E=ROOT/'external'
URLS={
 'GRF20.zip':'https://nces.ed.gov/programs/edge/data/GRF20.zip',
 'counties.zip':'https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip',
 'county_returns_2000_2016.csv':'https://raw.githubusercontent.com/MEDSL/county-returns/master/countypres_2000-2016.csv'}
def main():
 E.mkdir(exist_ok=True)
 for name,url in URLS.items():
  path=E/name
  if not path.exists():
   with requests.get(url,stream=True,timeout=120) as response:
    response.raise_for_status()
    tmp=path.with_suffix(path.suffix+'.partial')
    with tmp.open('wb') as f:
     for part in response.iter_content(1024*1024):f.write(part)
    tmp.replace(path)
 if not (E/'district_county_2020.csv').exists():
  with zipfile.ZipFile(E/'GRF20.zip') as z:
   data=z.read('grf20_lea_county.sas7bdat')
  pd.read_sas(io.BytesIO(data),format='sas7bdat',encoding='utf-8').to_csv(E/'district_county_2020.csv',index=False)
 manifest={name:{'url':url,'sha256':hashlib.sha256((E/name).read_bytes()).hexdigest(),'bytes':(E/name).stat().st_size} for name,url in URLS.items()}
 manifest['notes']='Retrieved 2026-10-31. Official MEDSL GitHub release (file version 20190722), 2016 baseline. Newer Dataverse release not used because download requires a guestbook submission. NCES and Census geography fixed at 2020; boundary changes not modeled.'
 (E/'sources.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
