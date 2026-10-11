"""Download Emily Oster's COVID School Data Hub state files (links scraped from covidschooldatahub.com).
Keeps district-level files: DistrictsByShare (share of year by learning model), Districts_LearningModelData,
MaskingData, Demographics, AssmtOutcomesByInPersonAccess, plus data-detail PDFs. Output: Datastore/Covid School Data Hub/raw/"""
import re, urllib.request, urllib.parse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / "Datastore" / "Covid School Data Hub"
RAW = ROOT / "raw"; RAW.mkdir(parents=True, exist_ok=True)
html = urllib.request.urlopen("https://www.covidschooldatahub.com/", timeout=60).read().decode("utf-8", "ignore")
urls = sorted(set(re.findall(r"https://assets\.ctfassets\.net/[^\"\\ ]+", html)))
keep = re.compile(r"(DistrictsByShare|Districts_LearningModelData|MaskingData|Demographics_|AssmtOutcomes|CSDH_.*_COVID_Districts|Data_Details|State_Overall_Shares)", re.I)
seen = set()
for u in urls:
    name = urllib.parse.unquote(u.rsplit("/", 1)[1])
    if not keep.search(name) or name in seen: continue
    # skip xlsx twin when a csv exists
    if name.endswith(".xlsx") and any(x.rsplit("/",1)[1] == name[:-5] + ".csv" for x in urls): continue
    seen.add(name)
    f = RAW / name
    if not f.exists():
        try: f.write_bytes(urllib.request.urlopen(u, timeout=120).read())
        except Exception as e: print("FAIL", name, e)
print(len(seen), "files")
