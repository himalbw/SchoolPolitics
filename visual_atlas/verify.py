"""Check delivered counts, joins, figure files and executed notebook outputs."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np,nbformat
from PIL import Image
from prepare import ROOT,PROJECT,THEMES

def main():
 c=pd.read_parquet(ROOT/'tables/cohort.parquet');p=pd.read_parquet(ROOT/'tables/phrase_counts_v2.parquet')
 expected=pd.read_parquet(PROJECT/'analysis/lda_cleaning_comparison/cohort.parquet')
 assert len(c)==len(p)==83366 and set(c.id)==set(p.id)==set(expected.id)
 assert p.id.is_unique and c.id.is_unique
 source=pd.read_parquet(PROJECT/'data/modeling_v2_2018_2023/model_manifest.parquet').set_index('id')
 assert np.array_equal(p.set_index('id').words_all.sort_index().values,source.loc[p.id].core_token_count.sort_index().values)
 assert (p.words_no_covid<=p.words_all).all()
 for theme in THEMES:assert (p[theme+'_no_covid']<=p[theme+'_all']).all()
 assert p.covid_explicit_no_covid.eq(0).all()
 conflicts=int(c.title_year_conflict.sum());assert conflicts==38
 counts=pd.read_csv(ROOT/'tables/district_quarter_coverage.csv',index_col=0);assert counts.to_numpy().sum()==83366-conflicts
 complete=pd.read_csv(ROOT/'tables/complete_quarter_panel.csv',dtype=str).leaid;assert len(complete)==186
 assert counts.loc[complete.astype(int)].gt(0).all().all()
 maps=pd.read_csv(ROOT/'tables/single_county_district_elections.csv',dtype={'leaid':str,'county_fips':str});assert maps.leaid.is_unique
 assert maps.county_fips.str.fullmatch(r'\d{5}').all()
 assert maps.rep_share_2016.dropna().between(0,1).all()
 catalog=json.loads((ROOT/'figure_catalog.json').read_text());assert len(catalog)==13
 for item in catalog:
  for ext in ['png','pdf']:assert (ROOT/'figures'/f'{item["name"]}.{ext}').stat().st_size>10000
  with Image.open(ROOT/'figures'/f'{item["name"]}.png') as im:im.verify()
 notebook=nbformat.read(ROOT/'School_board_visual_atlas.ipynb',as_version=4);code=[c for c in notebook.cells if c.cell_type=='code']
 assert all(c.execution_count is not None for c in code)
 assert not any(o.output_type=='error' for c in code for o in c.outputs)
 hashes=json.loads((ROOT/'output_checksums.json').read_text())
 for name,digest in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 result={'recordings_counted':len(p),'date_conflicts':conflicts,'new_analysis_recordings':int(counts.to_numpy().sum()),'complete_quarter_panel':len(complete),'figures':len(catalog),'formats':['PNG','PDF'],'executed_code_cells':len(code),'notebook_errors':0,'verified_checksums':len(hashes),'same_recording_ids_as_paired_LDA':True,'core_token_counts_match_saved_v2':True}
 (ROOT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
