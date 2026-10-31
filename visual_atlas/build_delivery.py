"""Build and execute the notebook, then render a browsable figure gallery."""
from pathlib import Path
import json,html,sys,hashlib,importlib.metadata
import nbformat as nb
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parent

METHODS='''These figures combine three explicitly labeled sources: new version-2 phrase counts for all 83,366 primary-cohort recordings; existing controlled before/after LDA results; and existing earlier-pipeline vocabulary/HC results. No new LDA fitting or v2 HC rerun is claimed.

New phrase measurement uses v2 core tokens, including stopwords and numbers. Counts are within fixed 1,000-token blocks, with a final block under 200 tokens merged. Dictionaries are searched independently so subjects may overlap. Rates are mentions per 10,000 core tokens, not the earlier alphabetic-token denominator. No-COVID excludes whole blocks with candidate COVID terms; it is not causal control. Dictionaries are exploratory, not validated policy or stance classifiers. Ads, repeated-caption artifacts, played media and ASR confusions remain possible.

38 recordings with exactly one explicit title year contradicting the recorded meeting year are excluded from new substantive panels (83,328 recordings remain). Dates otherwise retain the source meeting_date; this is not full date validation. The period panel requires at least four recordings and three observed quarters in each of 2018, 2019, 2022 and 2023. The quarterly panel is fixed to districts present in every one of the 24 quarters. Weighting is equal meetings within district-year/quarter, equal years within period, and equal districts. Intervals are pointwise and exploratory, not simultaneous discovery guarantees.

Geography uses exact zero-padded NCES LEAID matches to the 2020 relationship file. County comparisons use only single-county districts; multi-county/unmatched districts are audited. A single fixed crosswalk does not resolve historical boundary changes or original district attribution errors. State summaries allow multi-county districts whose counties all lie in one state. Maps show the contiguous US; tables retain other matched states. Gray is missing/unrepresented, not zero. Rates represent observed districts, not every resident or meeting in the county/state.

Political context is 2016 Republican votes divided by Democratic plus Republican presidential votes from MEDSL's official GitHub release. It is a pre-period county measure, not district election returns or speaker ideology. Scatterplot regression intervals cluster by state; group trajectories resample states within each political group (499 draws, fixed seed). These are unadjusted descriptive associations; group composition and unobserved confounders remain. Each political trajectory uses the same districts across all quarters. Few-cluster intervals can be unstable.

LDA perplexity is compared only within a representation. NPMI uses the existing shared held-out reference. HC plots use earlier processing and show exploratory standardized changes; all original global-test p-values reached the 0.0005 bootstrap resolution floor. Neither global-test statistic ratios nor raw p-values rank test power.'''

def build():
 catalog=list({x['name']:x for x in json.loads((ROOT/'figure_catalog.json').read_text())}.values())
 notebook=nb.v4.new_notebook()
 notebook.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
 cells=[nb.v4.new_markdown_cell('# School-board meetings: visual atlas\n\nAll eight proposed visualization families, with additional companion panels. Generated figures are saved as PNG and PDF in `figures/`; plot inputs and linkage audits are in `tables/`.\n\n## Methods and interpretation\n\n'+METHODS),
 nb.v4.new_code_cell("from pathlib import Path\nimport sys, os, json\nfrom IPython.display import display, Image, Markdown\nROOT = Path.cwd()\nassert (ROOT / 'prepare.py').exists(), 'Run this notebook from its own folder.'\nsys.path.insert(0, str(ROOT))\nos.environ.setdefault('MPLCONFIGDIR', '/tmp/stat236-mpl')\nimport pandas as pd\nimport download_sources, prepare, figures, geography\n# Set SCHOOL_BOARD_PROJECT before importing these modules to relocate source data.\n"),
 nb.v4.new_markdown_cell('## Reproduce the complete figure set\nThe first step uses cached count files when available. To rebuild counts after changing tokenization or dictionaries, remove only `tables/phrase_counts_v2.parquet` and rerun. Source datasets remain read-only. Public external files are cached with SHA-256 checksums.'),
 nb.v4.new_code_cell('download_sources.main()\nprepare.main()\nfigures.main()\ngeography.main()'),
 nb.v4.new_markdown_cell('## Audit the samples and joins'),
 nb.v4.new_code_cell("display(pd.read_csv(ROOT/'tables/theme_changes_v2.csv'))\ndisplay(pd.Series(json.loads((ROOT/'tables/geography_summary.json').read_text())))\ndisplay(pd.read_csv(ROOT/'tables/political_associations_state_clustered.csv'))")]
 for item in catalog:
  cells.append(nb.v4.new_markdown_cell('## '+item['title']+'\n\n'+item['caption']))
  cells.append(nb.v4.new_code_cell(f"display(Image(filename=str(ROOT/'figures/{item['name']}.png')))"))
 cells.append(nb.v4.new_markdown_cell('## Sources\n- [NCES geographic relationships](https://nces.ed.gov/programs/edge/geographic/relationshipfiles)\n- [Census 2020 county boundaries](https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_us_county_20m.zip)\n- [MEDSL official county-return release](https://github.com/MEDSL/county-returns)\n- Local input notebooks: `analysis/lda_cleaning_comparison` and `analysis/eligibility_hc_2018_2019_vs_2022_2023`.\n\nAll new files are additive; raw transcripts and earlier results are unchanged.'))
 notebook.cells=cells
 path=ROOT/'School_board_visual_atlas.ipynb';nb.write(notebook,path)
 NotebookClient(notebook,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
 nb.write(notebook,path)
 sections=[]
 for i in catalog:
  n=i['name'];sections.append(f'<section id="{n}"><h2>{html.escape(i["title"])}</h2><p>{html.escape(i["caption"])}</p><a href="figures/{n}.pdf">PDF</a> · <a href="figures/{n}.png">PNG</a><img loading="lazy" src="figures/{n}.png" alt="{html.escape(i["title"])}"></section>')
 nav=''.join(f'<li><a href="#{i["name"]}">{html.escape(i["title"])}</a></li>' for i in catalog)
 page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>School-board visual atlas</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#263238;background:#fafafa;margin:0}main{max-width:1200px;margin:auto;padding:40px 28px}h1{font-size:32px;line-height:1.2}h2{font-size:23px}a{color:#27647b}section{border-top:1px solid #ccd3d6;padding:28px 0}img{width:100%;height:auto;background:white;margin-top:16px}p{max-width:1000px}summary{font-weight:600;cursor:pointer}.tag{color:#59656a;font-size:14px}</style><main><p class="tag">STAT 236 · Generated 31 October 2026 · Reproducible analysis</p><h1>School-board meetings: visual atlas</h1><p>Thirteen figures cover coverage, policy mentions, time trends, cleaning effects, vocabulary tests, geography and baseline voting. All figures have PNG/PDF exports and reproducible scripts.</p><p><a href="School_board_visual_atlas.ipynb">Jupyter notebook</a> · <a href="README.md">Reproduction guide</a> · <a href="tables/">Data tables and audits</a></p><details><summary>Read methods and limitations before interpreting</summary>'+''.join('<p>'+html.escape(p)+'</p>' for p in METHODS.split('\n\n'))+'</details><ol>'+nav+'</ol>'+''.join(sections)+'</main></html>'
 (ROOT/'index.html').write_text(page)
 pd=None
 versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib','statsmodels','pyarrow','nbformat','nbclient']}
 (ROOT/'environment.json').write_text(json.dumps({'python':sys.version,'packages':versions},indent=2))
 checksums={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.rglob('*') if f.is_file() and f.suffix in ['.png','.pdf','.csv','.parquet','.ipynb','.py'] and 'vendor' not in f.parts and '__pycache__' not in f.parts and 'superseded' not in f.name}
 (ROOT/'output_checksums.json').write_text(json.dumps(checksums,indent=2))
 print('DELIVERY BUILT: executed notebook and gallery',flush=True)

if __name__=='__main__':build()
