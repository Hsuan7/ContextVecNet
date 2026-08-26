"""Local browser UI for annotating image-text inconsistency cases.

Run:
    /home/angle/miniconda3/envs/contextvecnet/bin/python annotation_case_review_server.py

Then open:
    http://127.0.0.1:8765

The server reads case_review_samples.csv and writes annotations to
case_review_samples_annotated.csv by default.
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import pandas as pd


DEFAULT_INPUT = Path(
    "results/final_preprocessing_v2/image_text_alignment_analysis/case_review_samples.csv"
)
DEFAULT_OUTPUT = Path(
    "results/final_preprocessing_v2/image_text_alignment_analysis/case_review_samples_annotated.csv"
)

MANUAL_COLUMNS = [
    "manual_text_image_consistency",
    "manual_image_extra_signal",
    "manual_image_effect",
    "manual_inconsistency_taxonomy_A_to_H",
    "manual_identifiable_info",
    "manual_notes",
]


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value)


def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value):
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


class AnnotationStore:
    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        load_path = output_path if output_path.exists() else input_path
        if not load_path.exists():
            raise FileNotFoundError(f"Cannot find case CSV: {load_path}")
        self.df = pd.read_csv(load_path)
        if "case_type" not in self.df.columns and "minimal_review_reason" in self.df.columns:
            self.df.insert(0, "case_type", self.df["minimal_review_reason"].fillna("").astype(str))
        for col in MANUAL_COLUMNS:
            if col not in self.df.columns:
                self.df[col] = ""
            self.df[col] = self.df[col].fillna("").astype(str)
        self.save()

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.output_path.with_suffix(".tmp.csv")
        self.df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, self.output_path)

    def count(self):
        return len(self.df)

    def annotated_count(self):
        if not len(self.df):
            return 0
        return int((self.df["manual_text_image_consistency"].fillna("") != "").sum())

    def case_types(self):
        return sorted(clean_value(v) for v in self.df["case_type"].dropna().unique())

    def row_to_case(self, index: int):
        row = self.df.iloc[index]
        image_candidates = [
            {
                "label": "Lowest similarity image",
                "caption": clean_value(row.get("lowest_similarity_caption", "")),
                "image_path": clean_value(row.get("lowest_similarity_image_path", "")),
                "similarity": safe_float(row.get("lowest_post_similarity", "")),
            },
            {
                "label": "Highest similarity image",
                "caption": clean_value(row.get("highest_similarity_caption", "")),
                "image_path": clean_value(row.get("highest_similarity_image_path", "")),
                "similarity": safe_float(row.get("highest_post_similarity", "")),
            },
        ]
        rater_choices = {}
        for col in MANUAL_COLUMNS:
            rater1 = clean_value(row.get(f"{col}_rater1", ""))
            rater2 = clean_value(row.get(f"{col}_rater2", ""))
            if rater1 or rater2:
                rater_choices[col] = {"rater1": rater1, "rater2": rater2}

        return {
            "index": index,
            "total": self.count(),
            "progress": self.annotated_count(),
            "case_type": clean_value(row.get("case_type", "")),
            "fold": safe_int(row.get("fold", "")),
            "author": clean_value(row.get("author", "")),
            "label": safe_int(row.get("label", "")),
            "text_probability": safe_float(row.get("text_probability", "")),
            "multimodal_probability": safe_float(row.get("multimodal_probability", "")),
            "delta_p": safe_float(row.get("delta_p", "")),
            "user_alignment_group": clean_value(row.get("user_alignment_group", "")),
            "multimodal_error_type": clean_value(row.get("multimodal_error_type", "")),
            "mean_similarity": safe_float(row.get("mean_similarity", "")),
            "valid_post_count": safe_int(row.get("valid_post_count", "")),
            "images": image_candidates,
            "manual": {col: clean_value(row.get(col, "")) for col in MANUAL_COLUMNS},
            "rater_choices": rater_choices,
            "case_types": self.case_types(),
        }

    def update_case(self, index: int, values: dict):
        if index < 0 or index >= self.count():
            raise IndexError(f"Case index out of range: {index}")
        for col in MANUAL_COLUMNS:
            self.df.at[index, col] = clean_value(values.get(col, ""))
        self.save()

    def next_unannotated(self, start: int = 0):
        if not len(self.df):
            return 0
        start = max(0, min(start, len(self.df) - 1))
        for offset in range(len(self.df)):
            idx = (start + offset) % len(self.df)
            if clean_value(self.df.iloc[idx].get("manual_text_image_consistency", "")) == "":
                return idx
        return start

    def filtered_indices(self, case_type: str):
        if not case_type or case_type == "ALL":
            return list(range(len(self.df)))
        return self.df.index[self.df["case_type"].astype(str) == case_type].tolist()


HTML_PAGE = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <title>圖文不一致案例人工標註</title>
  <style>
    body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 0; background: #f6f7f9; color: #1f2937; }
    header { background: #111827; color: white; padding: 14px 22px; position: sticky; top: 0; z-index: 1; }
    main { padding: 18px 22px 40px; max-width: 1300px; margin: 0 auto; }
    .bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 14px 0; }
    button, select, input, textarea { font: inherit; }
    button { border: 0; padding: 8px 12px; border-radius: 8px; background: #2563eb; color: white; cursor: pointer; }
    button.secondary { background: #4b5563; }
    button.good { background: #059669; }
    button.warn { background: #dc2626; }
    button:disabled { opacity: .5; cursor: not-allowed; }
    select, input, textarea { border: 1px solid #cbd5e1; border-radius: 8px; padding: 7px 9px; background: white; }
    textarea { min-height: 76px; width: 100%; resize: vertical; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
    .meta { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .pill { display: inline-block; padding: 3px 8px; border-radius: 999px; background: #e0ecff; color: #1e40af; margin-right: 6px; }
    .metric { background: #f8fafc; padding: 8px; border-radius: 8px; }
    .metric b { display: block; font-size: 12px; color: #64748b; margin-bottom: 2px; }
    .imageBox img { max-width: 100%; max-height: 520px; border-radius: 8px; border: 1px solid #e5e7eb; background: #111; display: block; margin: 8px auto; }
    .caption { white-space: pre-wrap; line-height: 1.55; max-height: 260px; overflow: auto; background: #f8fafc; border-radius: 8px; padding: 10px; }
    .formgrid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .field label { display: block; font-size: 13px; color: #475569; margin-bottom: 4px; }
    .taxonomy { font-size: 13px; line-height: 1.5; background: #fffbeb; border: 1px solid #fde68a; border-radius: 8px; padding: 10px; }
    .status { color: #e0f2fe; font-size: 13px; margin-top: 4px; }
    .small { font-size: 13px; color: #64748b; }
    @media (max-width: 980px) { .grid, .formgrid, .meta { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <div><b>圖文不一致案例人工標註</b></div>
    <div id="status" class="status">Loading...</div>
  </header>
  <main>
    <div class="bar">
      <button class="secondary" onclick="go(-1)">上一筆</button>
      <button class="secondary" onclick="go(1)">下一筆</button>
      <button onclick="jumpUnannotated()">下一筆未標註</button>
      <span>跳到第 <input id="jumpIndex" type="number" min="1" style="width:80px" /> 筆</span>
      <button class="secondary" onclick="jumpTo()">跳轉</button>
      <span>案例類型 <select id="caseTypeFilter" onchange="applyFilter()"></select></span>
      <button class="good" onclick="saveCase()">儲存</button>
    </div>

    <section class="card">
      <div id="caseTitle"></div>
      <div class="meta" id="metrics"></div>
      <div id="raterChoices" class="small" style="margin-top:10px"></div>
    </section>

    <div class="grid" style="margin-top:16px">
      <section class="card imageBox">
        <h3>最低 similarity 貼文</h3>
        <div id="lowInfo" class="small"></div>
        <img id="lowImg" alt="lowest similarity image" />
        <div id="lowCaption" class="caption"></div>
      </section>
      <section class="card imageBox">
        <h3>最高 similarity 貼文</h3>
        <div id="highInfo" class="small"></div>
        <img id="highImg" alt="highest similarity image" />
        <div id="highCaption" class="caption"></div>
      </section>
    </div>

    <section class="card" style="margin-top:16px">
      <h3>人工標註</h3>
      <div class="formgrid">
        <div class="field">
          <label>圖文是否一致</label>
          <select id="manual_text_image_consistency">
            <option value=""></option><option>一致</option><option>部分一致</option><option>不一致</option>
          </select>
        </div>
        <div class="field">
          <label>圖片是否提供額外心理健康相關線索</label>
          <select id="manual_image_extra_signal">
            <option value=""></option><option>有</option><option>不明確</option><option>無</option>
          </select>
        </div>
        <div class="field">
          <label>圖片對模型可能的影響</label>
          <select id="manual_image_effect">
            <option value=""></option><option>幫助</option><option>干擾</option><option>無明顯影響</option>
          </select>
        </div>
        <div class="field">
          <label>不一致類型 A-H，可複選如 A;D</label>
          <input id="manual_inconsistency_taxonomy_A_to_H" />
        </div>
        <div class="field">
          <label>是否涉及可識別資訊</label>
          <select id="manual_identifiable_info">
            <option value=""></option><option>是</option><option>否</option>
          </select>
        </div>
      </div>
      <div class="field" style="margin-top:12px">
        <label>備註</label>
        <textarea id="manual_notes"></textarea>
      </div>
      <div class="taxonomy" style="margin-top:12px">
        <b>Taxonomy:</b>
        A Caption 高風險、圖片中性；
        B Caption 中性、圖片低落氛圍；
        C 圖片為裝飾或無關內容；
        D 圖片含文字截圖；
        E 圖文反諷或語氣不一致；
        F 多圖或影片代表性不足；
        G 缺圖或低品質圖片；
        H 平台風格影響。
      </div>
    </section>
  </main>

<script>
let current = 0;
let total = 0;
let caseTypes = [];

function enc(path) { return encodeURIComponent(path || ""); }
function fnum(x) { return x === null || x === undefined || x === "" ? "" : Number(x).toFixed(4); }

async function loadCase(index) {
  const resp = await fetch(`/api/case?index=${index}`);
  if (!resp.ok) { alert(await resp.text()); return; }
  const data = await resp.json();
  current = data.index; total = data.total; caseTypes = data.case_types || [];
  document.getElementById("status").textContent = `已標註 ${data.progress} / ${data.total}，目前第 ${current + 1} 筆`;
  document.getElementById("jumpIndex").value = current + 1;
  fillCaseTypeFilter(data.case_type);
  document.getElementById("caseTitle").innerHTML =
    `<span class="pill">${data.case_type}</span><span class="pill">${data.user_alignment_group}</span><span class="pill">${data.multimodal_error_type}</span>` +
    ` fold=${data.fold} author=${escapeHtml(data.author)} label=${data.label}`;
  const metrics = [
    ["Text prob", fnum(data.text_probability)],
    ["Multimodal prob", fnum(data.multimodal_probability)],
    ["delta_p", fnum(data.delta_p)],
    ["Mean similarity", fnum(data.mean_similarity)],
    ["Valid posts", data.valid_post_count ?? ""],
  ];
  document.getElementById("metrics").innerHTML = metrics.map(([k,v]) => `<div class="metric"><b>${k}</b>${v}</div>`).join("");
  const raterChoices = data.rater_choices || {};
  const raterHtml = Object.entries(raterChoices)
    .filter(([_, v]) => (v.rater1 || v.rater2))
    .map(([k, v]) => `<div><b>${escapeHtml(k)}</b>: Rater1=${escapeHtml(v.rater1 || "")} / Rater2=${escapeHtml(v.rater2 || "")}</div>`)
    .join("");
  document.getElementById("raterChoices").innerHTML = raterHtml ? `<b>兩位評分者原始標註</b>${raterHtml}` : "";
  const low = data.images[0], high = data.images[1];
  setImageBlock("low", low);
  setImageBlock("high", high);
  for (const [key, value] of Object.entries(data.manual)) {
    const el = document.getElementById(key);
    if (el) el.value = value || "";
  }
}

function setImageBlock(prefix, item) {
  document.getElementById(prefix + "Info").textContent = `similarity=${fnum(item.similarity)} path=${item.image_path || ""}`;
  document.getElementById(prefix + "Caption").textContent = item.caption || "";
  const img = document.getElementById(prefix + "Img");
  img.src = item.image_path ? `/image?path=${enc(item.image_path)}` : "";
}

function fillCaseTypeFilter(currentType) {
  const sel = document.getElementById("caseTypeFilter");
  if (sel.options.length) return;
  sel.innerHTML = `<option value="ALL">ALL</option>` + caseTypes.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join("");
}

async function saveCase() {
  const manual = {};
  ["manual_text_image_consistency","manual_image_extra_signal","manual_image_effect","manual_inconsistency_taxonomy_A_to_H","manual_identifiable_info","manual_notes"].forEach(id => {
    manual[id] = document.getElementById(id).value;
  });
  const resp = await fetch("/api/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({index: current, manual})
  });
  if (!resp.ok) { alert(await resp.text()); return; }
  document.getElementById("status").textContent = `已儲存第 ${current + 1} 筆`;
}

async function go(step) {
  await saveCase();
  let next = Math.max(0, Math.min(total - 1, current + step));
  loadCase(next);
}

async function jumpTo() {
  await saveCase();
  const idx = Number(document.getElementById("jumpIndex").value || 1) - 1;
  loadCase(Math.max(0, Math.min(total - 1, idx)));
}

async function jumpUnannotated() {
  await saveCase();
  const resp = await fetch(`/api/next_unannotated?start=${current + 1}`);
  const data = await resp.json();
  loadCase(data.index);
}

async function applyFilter() {
  await saveCase();
  const type = document.getElementById("caseTypeFilter").value;
  const resp = await fetch(`/api/first_of_type?case_type=${encodeURIComponent(type)}`);
  const data = await resp.json();
  loadCase(data.index);
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

loadCase(0);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    store: AnnotationStore

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_text(self, text, status=200, content_type="text/plain; charset=utf-8"):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                return self.send_text(HTML_PAGE, content_type="text/html; charset=utf-8")
            if parsed.path == "/api/case":
                index = int(qs.get("index", ["0"])[0])
                index = max(0, min(self.store.count() - 1, index))
                return self.send_json(self.store.row_to_case(index))
            if parsed.path == "/api/next_unannotated":
                start = int(qs.get("start", ["0"])[0])
                return self.send_json({"index": self.store.next_unannotated(start)})
            if parsed.path == "/api/first_of_type":
                case_type = qs.get("case_type", ["ALL"])[0]
                indices = self.store.filtered_indices(case_type)
                return self.send_json({"index": indices[0] if indices else 0})
            if parsed.path == "/image":
                image_path = Path(unquote(qs.get("path", [""])[0]))
                if not image_path.exists() or not image_path.is_file():
                    return self.send_text(f"Image not found: {image_path}", status=404)
                content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
                data = image_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return self.send_text("Not found", status=404)
        except Exception as exc:
            return self.send_text(str(exc), status=500)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/save":
            return self.send_text("Not found", status=404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.store.update_case(int(payload["index"]), payload.get("manual", {}))
            return self.send_json({"ok": True, "progress": self.store.annotated_count()})
        except Exception as exc:
            return self.send_text(str(exc), status=500)

    def log_message(self, fmt, *args):
        return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    store = AnnotationStore(args.input, args.output)
    Handler.store = store
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Annotation UI: {url}")
    print(f"Input : {args.input}")
    print(f"Output: {args.output}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
