"""Compare every published paired-analysis output with a separately reproduced copy."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]/'analysis/phase_a'
parser=argparse.ArgumentParser();parser.add_argument('--repeat',type=Path,required=True);a=parser.parse_args()
checks=[]
for folder in ('task1','task2','task5'):
    for original in sorted((ROOT/folder).rglob('*')):
        if not original.is_file():continue
        rel=original.relative_to(ROOT);repeat=a.repeat/rel
        ok=repeat.is_file() and hashlib.sha256(original.read_bytes()).digest()==hashlib.sha256(repeat.read_bytes()).digest()
        checks.append({'file':rel.as_posix(),'exact_match':ok})
bad=[x for x in checks if not x['exact_match']]
print(json.dumps({'scientific_outputs':len(checks),'mismatches':bad,'scope':'Published scientific outputs only. Execution and previous-version comparison logs are not distributed.'},indent=2))
raise SystemExit(1 if bad or not checks else 0)
