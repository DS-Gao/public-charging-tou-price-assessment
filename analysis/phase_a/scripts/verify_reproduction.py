"""Compare deterministic scientific rerun outputs with delivered originals."""
from pathlib import Path
import argparse,hashlib,json
R=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--repeat',type=Path,required=True);a=ap.parse_args()
repeat=a.repeat.resolve()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in repeat.rglob('*') if p.is_file()]
results=[]
for p in sorted(files):
 rel=p.relative_to(repeat)
 if '__pycache__' in rel.parts or p.name=='run_environment.json':continue
 primary=R/rel
 results.append({'file':rel.as_posix(),'exact_match':primary.is_file() and sha(primary)==sha(p)})
bad=[r for r in results if not r['exact_match']]
print(json.dumps({'scientific_files_compared':len(results),'exact_matches':len(results)-len(bad),'mismatches':bad,'environment_metadata_excluded':'run_environment.json can contain executable paths on another machine'},ensure_ascii=False,indent=2))
raise SystemExit(1 if bad or not results else 0)

