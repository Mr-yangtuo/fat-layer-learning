"""Build a portable, self-contained learning handbook from audited source snapshots."""
from pathlib import Path
import json, hashlib, re, ast, base64, html as html_module, xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "docs"
SNAPSHOTS = ROOT / "source"
OUT.mkdir(exist_ok=True)
SNAPSHOTS.mkdir(exist_ok=True)
files = []

def validate_flow(flow, context):
    if not flow:
        return
    assert flow.get("title") and flow.get("caption"), context
    nodes = flow["nodes"]
    ids = {n["id"] for n in nodes}
    assert len(ids) == len(nodes) and len(nodes) > 1, context
    for node in nodes:
        assert node.get("title") and node.get("detail"), (context, node)
    for edge in flow["edges"]:
        assert edge["from"] in ids and edge["to"] in ids, (context, edge)

def source_inventory(tree, segments):
    definitions = []
    def walk(body, scope=""):
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = next((s for s in segments if s["start"] <= node.lineno <= s["end"]), None)
                assert owner, ("definition without explanation", scope, node.name)
                is_class = isinstance(node, ast.ClassDef)
                signature = ("class " + node.name + "(" + ", ".join(ast.unparse(b) for b in node.bases) + ")") if is_class else node.name + "(" + ast.unparse(node.args) + ")"
                definitions.append({"name":node.name, "scope":scope, "signature":signature, "line":node.lineno, "end":node.end_lineno, "segmentStart":owner["start"], "kind":"class" if is_class else "function"})
                walk(node.body, (scope + "." if scope else "") + node.name)
    walk(tree.body)
    return definitions

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
            for key in ("title","summary","principle","syntax","steps","returns"):
                assert seg.get(key), (item["path"], key, seg["title"])
            assert not any(key in seg for key in ("exercise", "answer", "diagram")), (item["path"], "obsolete teaching schema")
            for key in ("parameters", "variables"):
                assert isinstance(seg.get(key), list), (item["path"], key, seg["title"])
                for row in seg[key]:
                    assert all(row.get(k) for k in ("name", "type", "meaning")), (item["path"], row)
            validate_flow(seg.get("flow"), (item["path"], seg["title"]))
        for key in ("overview", "role", "inputs", "outputs", "components", "readingGuide"):
            assert item.get(key), (item["path"], key)
        for component in item["components"]:
            assert all(component.get(k) for k in ("name", "kind", "summary", "segmentStart")), (item["path"], component)
            assert component["segmentStart"] in {s["start"] for s in item["segments"]}, (item["path"], component)
        validate_flow(item.get("pipeline"), item["path"])
        if relative.suffix == ".py":
            tree = ast.parse(source)
            item["definitions"] = source_inventory(tree, item["segments"])
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef)):
                    assert any(s["start"]<=node.lineno<=s["end"] for s in item["segments"]), (item["path"],"uncovered function",node.name,node.lineno)
                    relevant = [s for s in item["segments"] if s["start"] <= node.end_lineno and s["end"] >= node.lineno]
                    names = " ".join(p["name"] for s in relevant for p in s["parameters"])
                    params = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                    params += [p for p in (node.args.vararg, node.args.kwarg) if p]
                    for param in params:
                        if param.arg not in ("self", "cls"):
                            assert re.search(r"(?<![A-Za-z_0-9])" + re.escape(param.arg) + r"(?![A-Za-z_0-9])", names), (item["path"], node.name, "missing parameter explanation", param.arg)
        item.update(source=source,lines=lines,sha256=hashlib.sha256(source.encode()).hexdigest())
        files.append(item)
assert len({f["path"] for f in files}) == len(files)
for item in files:
    dependencies = []
    if item["path"].endswith(".py"):
        tree = ast.parse(item["source"])
        package = item["path"].split("/")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = package[:len(package)-node.level+1] if node.level else []
                module = base + (node.module.split(".") if node.module else [])
                candidates = ["/".join(module) + ".py", "/".join(module + ["__init__.py"])]
                for candidate in candidates:
                    if candidate != item["path"] and candidate in {f["path"] for f in files} and candidate not in dependencies:
                        dependencies.append(candidate)
    item["dependencies"] = dependencies
report = (ROOT / "report.html").read_text(encoding="utf-8-sig")
figure_info = {
    "interpolation": ("图 3 · 离散点到逐列标签的完整计算", "示例使用原图8×20、新图16×10。先反查原图横坐标，线性插值，再缩放纵坐标；数值存在和标签有效是两件事。"),
    "projection": ("图 7 · 三次卷积，三种不同职责", "拼接后的融合卷积先压缩描述，预测头再扩展隐藏特征，最后用两组权重生成两条边界分数。128是隐藏宽度，2是边界数量。"),
    "workflow": ("图 1 · 从示范、训练到独立预测", "训练路用人工答案计算误差并更新网络；验证路不更新参数，只观察表现和选择模型。部署时新图像不需要人工答案。"),
    "annotation": ("图 2 · 从人工边界到逐列监督", "线由坐标点描述，缺少标注的列需另行记录。示意图显示原始标注关系；实际重采样可能通过插值填补部分空缺。"),
    "preprocess": ("图 4 · 两种视野，分别同步缩放", "原图始终保留，符合条件时额外加入底部裁剪版。两者各自缩放到256×256，边界按对应比例变化。"),
    "convolution": ("图 5 · 卷积怎样产生多张特征图", "局部窗口滑动得到一张响应图；不同滤波器得到不同通道。数字例子用于说明运算，并非真实网络权重。"),
    "model": ("图 6 · 完整的双尺度边界预测网络", "沿编号与箭头阅读。叠层表示一组特征图，叠层数量仅示意，实际通道数以文字标注为准。默认输入为单张256×256灰度图。"),
    "block": ("图 8 · 骨干内部的倒残差积木", "这是骨干内部的通用单元，不是整个边界预测网络。通道门控与残差连接只在满足相应配置和尺寸条件时使用。"),
    "heatmap": ("图 9 · 热图监督与概率加权坐标解码", "人工边界构造目标分布用于训练，模型评分沿高度归一化后用于解码。所有色带、概率条和曲线均为示意。"),
}
def embed_figure(match):
    key = match.group(1)
    title, caption = figure_info[key]
    svg = (ROOT / "figures" / (key + ".svg")).read_bytes()
    tree = ET.fromstring(svg)
    assert tree.tag.endswith("svg")
    view = tree.attrib["viewBox"].split()
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg).decode()
    label = html_module.escape(title)
    return ('<figure class="paper-figure" id="figure-' + key + '"><div class="figure-top"><span>' + label +
        '</span><button type="button" data-zoom-figure="' + key + '">放大看图 ↗</button></div><div class="figure-canvas">' +
        '<img src="' + uri + '" alt="' + label + '" width="' + view[2] + '" height="' + view[3] +
        '" loading="lazy"></div><figcaption>' + caption + '</figcaption><div class="figure-bottom">' +
        '<span>可横向滑动 · 点击放大查看细节</span><a download="' + key + '.svg" href="' + uri +
        '">下载矢量原图 ↓</a></div></figure>')
report = re.sub(r"<!--FIG:([a-z]+)-->", embed_figure, report)
assert "<!--FIG:" not in report
assert all(link in {f["path"] for f in files} for link in re.findall(r'href="#code/([^"?]+)',report))
guide = (ROOT / "reader-guide.html").read_text(encoding="utf-8-sig")
for path, line in re.findall(r'href="#code/([^"?]+)(?:\?line=(\d+))?"', guide):
    file = next((f for f in files if f["path"] == path), None)
    assert file, ("guide links to unknown file", path)
    if line:
        assert any(s["start"] <= int(line) <= s["end"] for s in file["segments"]), (path, line)
visuals = json.loads((ROOT / "reader-visuals.json").read_text(encoding="utf-8-sig"))
assert len({v["id"] for v in visuals}) == len(visuals)
for visual in visuals:
    svg = (ROOT / "figures" / (visual["id"] + ".svg")).read_bytes()
    xml = ET.fromstring(svg)
    assert xml.tag.endswith("svg"), visual["id"]
    box = xml.attrib["viewBox"].split()
    visual.update(uri="data:image/svg+xml;base64," + base64.b64encode(svg).decode(), width=float(box[2]), height=float(box[3]))
    for link in visual["links"]:
        file = next(f for f in files if f["path"] == link["path"])
        segment = next(s for s in file["segments"] if s["start"] == link["start"])
        segment.setdefault("visuals", []).append(visual["id"])
manifest = {"created":"2026-11-14","updated":"2026-11-22","reader_version":5,"files":files,"visuals":visuals}
payload = json.dumps(manifest,ensure_ascii=False,separators=(",",":")).replace("<","\\u003c")
html = (ROOT / "template.html").read_text(encoding="utf-8-sig")
for key,value in {
    "STYLE":(ROOT / "style.css").read_text(encoding="utf-8-sig") + "\n" + (ROOT / "reader.css").read_text(encoding="utf-8-sig"),
    "REPORT":report,
    "GUIDE":guide,
    "DATA":payload,
    "APP":(ROOT / "app.js").read_text(encoding="utf-8-sig").replace("/*__READER__*/", (ROOT / "reader.js").read_text(encoding="utf-8-sig"))
}.items():
    html = html.replace("/*__"+key+"__*/",value)
assert "/*__" not in html
(OUT / "index.html").write_text(html,encoding="utf-8")
(OUT / ".nojekyll").touch()
summary = {"files":len(files),"segments":sum(len(f["segments"]) for f in files),"source_lines":sum(len(f["lines"]) for f in files),"html_bytes":len(html.encode()),"hashes":{f["path"]:f["sha256"] for f in files}}
summary.update(definitions=sum(len(f.get("definitions",[])) for f in files), parameter_entries=sum(len(s["parameters"]) for f in files for s in f["segments"]), variable_entries=sum(len(s["variables"]) for f in files for s in f["segments"]), flowcharts=sum(bool(f.get("pipeline")) + sum(bool(s.get("flow")) for s in f["segments"]) for f in files))
summary.update(concept_illustrations=len(visuals), illustrated_segments=sum(bool(s.get("visuals")) for f in files for s in f["segments"]))
(ROOT / "build-manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k!="hashes"},ensure_ascii=False))
