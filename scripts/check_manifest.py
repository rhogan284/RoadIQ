import json
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker

schema = json.loads(Path("src/edgecv/contracts/schemas/capture_manifest.schema.json").read_text())
manifest = json.loads(Path("data/rdd2022/manifest.json").read_text())
records = manifest["clean"] + manifest["defect"]

validator = Draft202012Validator(schema, format_checker=FormatChecker())
bad = [(r["seq"], e.message) for r in records for e in validator.iter_errors(r)]

print(f"{len(records)} records, {len(bad)} errors")
for seq, msg in bad[:5]:
    print(seq, msg)