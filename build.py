"""Build a portable, self-contained learning handbook from audited source snapshots."""
from pathlib import Path
import json, hashlib, re, ast

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "docs"
SNAPSHOTS = ROOT / "source"
OUT.mkdir(exist_ok=True)
SNAPSHOTS.mkdir(exist_ok=True)
files = []
for part in ("data", "model", "training"):
    items = json.loads((ROOT / f"content-{part}.json").read_text(encoding="utf-8-sig"))
    for item in items:
        relative = Path(item["path"])
        original = PROJECT / relative
        snapshot = SNAPSHOTS / relative
        # On a stand-alone checkout, use committed snapshots.
        if original.is_file():
            source = original.read_text(encoding="utf-8-sig")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_text(source, encoding="utf-8")
        elif snapshot.is_file():
            source = snapshot.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(item["path"])
        lines = source.splitlines()
        previous = 0
        for seg in item["segments"]:
            assert 1 <= seg["start"] <= seg["end"] <= len(lines), (item["path"], seg["title"])
            assert seg["start"] > previous, (item["path"], "overlapping segments", seg["title"])
            previous = seg["end"]
            for key in ("title","principle","syntax","exercise","answer"):
                assert seg.get(key), (item["path"], key, seg["title"])
        if relative.suffix == ".py":
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef)):
                    assert any(s["start"]<=node.lineno<=s["end"] for s in item["segments"]), (item["path"],"uncovered function",node.name,node.lineno)
        item.update(source=source,lines=lines,sha256=hashlib.sha256(source.encode()).hexdigest())
        files.append(item)
assert len({f["path"] for f in files}) == len(files)
report = (ROOT / "report.html").read_text(encoding="utf-8-sig")
assert all(link in {f["path"] for f in files} for link in re.findall(r'href="#code/([^"?]+)',report))
manifest = {"created":"2026-11-14","files":files}
payload = json.dumps(manifest,ensure_ascii=False,separators=(",",":")).replace("<","\\u003c")
html = (ROOT / "template.html").read_text(encoding="utf-8-sig")
for key,value in {
    "STYLE":(ROOT / "style.css").read_text(encoding="utf-8-sig"),
    "REPORT":report,
    "DATA":payload,
    "APP":(ROOT / "app.js").read_text(encoding="utf-8-sig")
}.items():
    html = html.replace("/*__"+key+"__*/",value)
assert "/*__" not in html
(OUT / "index.html").write_text(html,encoding="utf-8")
(OUT / ".nojekyll").touch()
summary = {"files":len(files),"segments":sum(len(f["segments"]) for f in files),"source_lines":sum(len(f["lines"]) for f in files),"html_bytes":len(html.encode()),"hashes":{f["path"]:f["sha256"] for f in files}}
(ROOT / "build-manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k!="hashes"},ensure_ascii=False))
