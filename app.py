# -*- coding: utf-8 -*-
"""ระบบ ABSA ภาษาไทย — Flask web UI
กรอกข้อความรีวิว -> ตรวจแง่มุม (aspect) + อารมณ์ (sentiment) รายแง่มุม

โครงสร้างนี้ reuse logic เดียวกับที่ใช้ตอนเทรน/ประเมิน TEST มาทั้งสัปดาห์
(clean_text, aspect-conditioned sentiment pairing, threshold ต่อแง่มุม)
เพื่อให้ผลลัพธ์ตรงกับตัวเลขที่วัดไว้ ไม่ใช่ logic ใหม่ที่อาจให้ผลต่างออกไป

วิธีรัน (ดู README.md สำหรับรายละเอียดเต็ม):
    python app.py
    เปิดเบราว์เซอร์ไปที่ http://127.0.0.1:5000
"""
import html
import json
import os
import re

import torch
from flask import Flask, render_template, request

ASPECT_MODEL_DIR = os.environ.get(
    "ASPECT_MODEL_DIR", "./models/Aspect/aspect_extraction_11.07.2026")
ASPECT_TOKENIZER_DIR = os.environ.get(
    "ASPECT_TOKENIZER_DIR", "./models/Aspect/aspect_extraction_11.07.2026")
SENTIMENT_MODEL_DIR = os.environ.get(
    "SENTIMENT_MODEL_DIR", "./models/Sentiment/sentiment_extraction_v2")
SENTIMENT_TOKENIZER_DIR = os.environ.get(
    "SENTIMENT_TOKENIZER_DIR", "./models/Sentiment/sentiment_extraction_v2")
ASPECT_THRESHOLDS_PATH = os.environ.get(
    "ASPECT_THRESHOLDS_PATH", "./models/Aspect/aspect_extraction_11.07.2026/aspect_thresholds.json"
)
DEFAULT_ASPECT_THRESHOLD = 0.5  

MAX_LEN = 128
LOW_CONF_THRESHOLD = 0.55
ASPECTS = ["ห้องพัก", "บริการ", "ทำเลที่ตั้ง",
           "สิ่งอำนวยความสะดวก", "อาหารและเครื่องดื่ม", "ราคา"]
SENT_LABELS = ["Negative", "Neutral", "Positive"]
OVERALL_ASPECT_NAME = "ไม่พบแง่มุม"

_re_rep = re.compile(r"(.)\1{2,}")
_re_brk = re.compile(r"\(\)|\{\}|\[\]")

try:
    from pythainlp.tokenize import word_tokenize
    _HAS_PYTHAINLP_TOK = True
except ImportError:
    _HAS_PYTHAINLP_TOK = False

CONTRASTIVE_MARKERS = ["แต่ว่า", "แต่", "ทว่า", "ในทางกลับกัน", "ทั้งที่", "ถึงแม้", "แม้ว่า", "อย่างไรก็ตาม", "อย่างไรก็ดี"]
CAUSAL_MARKERS = ["เพราะว่า", "เพราะ", "เนื่องจาก", "เนื่องด้วย", "อันเนื่องมาจาก", "ด้วยเหตุที่", "จึง", "ดังนั้น", "ทำให้", "ส่งผลให้"]

def find_markers(text, markers):
    if _HAS_PYTHAINLP_TOK:
        tokens = set(word_tokenize(text, engine="newmm"))
        return [m for m in markers if m in tokens]
    return [m for m in markers if m in text]

try:
    from thaixtransformers.preprocess import process_transformers
    _HAS_WCB_PREP = True
except Exception:
    _HAS_WCB_PREP = False
    print("[คำเตือน] ไม่พบ thaixtransformers — ใช้ preprocessing แบบพื้นฐานแทน "
          "(ผลอาจต่างจากตอนเทรนเล็กน้อย ถ้าต้องการผลตรงเป๊ะให้ pip install thaixtransformers)")

try:
    from pythainlp.util import normalize as th_normalize
except Exception:
    th_normalize = lambda t: t 
    print("[คำเตือน] ไม่พบ pythainlp — ข้าม normalize ภาษาไทย")

try:
    import emoji
    _HAS_EMOJI = True
except Exception:
    _HAS_EMOJI = False


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    if _HAS_EMOJI:
        t = emoji.demojize(t).replace(":", " ").replace("_", " ")
    t = th_normalize(t)
    t = _re_brk.sub(" ", t)
    t = _re_rep.sub(r"\1", t)
    t = " ".join(t.split())
    if _HAS_WCB_PREP:
        t = process_transformers(t)
    return t

def _check_path(name, path):
    ok = os.path.exists(path)
    # status = "Found" if ok else "Not Found"
    # print(f"[{status}] {name} = {path}")
    return ok


print("=" * 60)
# print("กำลังตรวจสอบ path โมเดล...")
# print("=" * 60)

paths_ok = all([
    _check_path("ASPECT_MODEL_DIR", ASPECT_MODEL_DIR),
    _check_path("ASPECT_TOKENIZER_DIR", ASPECT_TOKENIZER_DIR),
    _check_path("SENTIMENT_MODEL_DIR", SENTIMENT_MODEL_DIR),
    _check_path("SENTIMENT_TOKENIZER_DIR", SENTIMENT_TOKENIZER_DIR),
])
if not paths_ok:
    raise SystemExit(
        "\nตำแหน่ง path ไม่ถูกต้อง — แก้ไข path ให้ตรงกับตำแหน่งจริง "
        "save ไว้ก่อนรันใหม่\n(ดูวิธีตั้งค่าใน README.md)"
    )
# aspect_thresholds.json — ถ้าไม่เจอ ใช้ค่า default แทนชั่วคราว
_has_thresholds_file = _check_path("ASPECT_THRESHOLDS_PATH", ASPECT_THRESHOLDS_PATH)

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nใช้งานอุปกรณ์: {device}")

print("กำลังโหลดโมเดล aspect ...")
aspect_tok = AutoTokenizer.from_pretrained(ASPECT_TOKENIZER_DIR)
aspect_mdl = AutoModelForSequenceClassification.from_pretrained(ASPECT_MODEL_DIR).to(device).eval()

print("กำลังโหลดโมเดล sentiment ...")
sent_tok = AutoTokenizer.from_pretrained(SENTIMENT_TOKENIZER_DIR)
sent_mdl = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_DIR).to(device).eval()

if _has_thresholds_file:
    with open(ASPECT_THRESHOLDS_PATH, encoding="utf-8") as f:
        aspect_thresholds = json.load(f)
else:
    aspect_thresholds = {a: DEFAULT_ASPECT_THRESHOLD for a in ASPECTS}
    print(f"\n⚠️  ไม่พบ aspect_thresholds.json — ใช้ค่า default {DEFAULT_ASPECT_THRESHOLD} "
          f"ทุกแง่มุมไปก่อน (ผลลัพธ์จะแม่นน้อยกว่าค่าที่จูนไว้จริง)")
    print(f"   ไปหาไฟล์นี้ใน Google Drive มาวางที่ {ASPECT_THRESHOLDS_PATH} "
          f"แล้วรันใหม่ เพื่อผลลัพธ์ที่แม่นยำตามที่วัด TEST ไว้จริง\n")

print("โมเดลพร้อมใช้งาน")
print("=" * 60)

@torch.no_grad()
def analyze_text(raw_text: str) -> dict:
    text_clean = clean_text(raw_text)
    if not text_clean:
        return {"original_text": raw_text, "aspects": [], "error": "ข้อความว่างเปล่าหลังทำความสะอาด"}

    # 1) ตรวจแง่มุมทั้งหมดที่พบในประโยค
    enc = aspect_tok(text_clean, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
    probs = torch.sigmoid(aspect_mdl(**enc).logits).squeeze(0).cpu().numpy()
    detected = [a for j, a in enumerate(ASPECTS) if probs[j] > aspect_thresholds[a]]
    
    if not detected:
        detected = [OVERALL_ASPECT_NAME]

    # ฟังก์ชันช่วยทาย Aspect รายท่อน
    def _get_aspect_probs(txt):
        e = aspect_tok(txt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
        return torch.sigmoid(aspect_mdl(**e).logits).squeeze(0).cpu().numpy()

    # ฟังก์ชันช่วยทาย Sentiment
    def _predict_sent(asp, txt):
        enc_sent = sent_tok(f"แง่มุม: {asp}", txt, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(device)
        p = torch.softmax(sent_mdl(**enc_sent).logits, dim=-1).squeeze(0).cpu().numpy()
        conf = float(p.max())
        label = SENT_LABELS[int(p.argmax())]
        return label, conf

    results = []
    contrastive = find_markers(text_clean, CONTRASTIVE_MARKERS)
    causal = find_markers(text_clean, CAUSAL_MARKERS)

    # เงื่อนไขการตัดประโยคขัดแย้ง
    if contrastive and not causal and text_clean.count(contrastive[0]) == 1:
        marker = contrastive[0]
        before, after = text_clean.split(marker, 1)
        after = marker + after
        before, after = before.strip(), after.strip()

        # =========================================================================
        # กรณี 1 แง่มุมแต่ขัดแย้งในตัวเอง (เช่น "ห้องกว้างดี แต่ห้องเก่ามาก")
        # =========================================================================
        if len(detected) == 1 and detected[0] != OVERALL_ASPECT_NAME:
            asp = detected[0]
            lbl_b, conf_b = _predict_sent(asp, before)
            lbl_a, conf_a = _predict_sent(asp, after)

            if conf_b >= 0.6 and conf_a >= 0.6 and lbl_b != lbl_a:
                aspect_prob = round(float(probs[ASPECTS.index(asp)]), 3) if asp in ASPECTS else None
                results.append({
                    "aspect": asp, "sentiment": lbl_b, "confidence": round(conf_b, 3),
                    "low_confidence": conf_b < LOW_CONF_THRESHOLD, "aspect_prob": aspect_prob,
                    "text_segment": before.replace("<_>", " "), "split_info": f"ตัดท่อนหน้า ({marker})"
                })
                results.append({
                    "aspect": asp, "sentiment": lbl_a, "confidence": round(conf_a, 3),
                    "low_confidence": conf_a < LOW_CONF_THRESHOLD, "aspect_prob": aspect_prob,
                    "text_segment": after.replace("<_>", " "), "split_info": f"ตัดท่อนหลัง ({marker})"
                })
                return {"original_text": raw_text, "text_clean": text_clean, "aspects": results, "error": None}

        # =========================================================================
        # กรณีหลายแง่มุม (Multi-Aspect เช่น "ห้องสะอาดดี แต่ราคาแพง")
        # ใช้ Aspect Model ตรวจว่าท่อนหน้า/หลัง พูดถึงแง่มุมไหนมากกว่ากัน
        # =========================================================================
        probs_before = _get_aspect_probs(before)
        probs_after = _get_aspect_probs(after)

        for a in detected:
            aspect_prob = round(float(probs[ASPECTS.index(a)]), 3) if a in ASPECTS else None
            
            if a in ASPECTS:
                asp_idx = ASPECTS.index(a)
                score_b = probs_before[asp_idx]
                score_a = probs_after[asp_idx]

                # เลือกท่อนที่มีความน่าจะเป็นของ Aspect นั้นสูงกว่า
                if score_b > score_a and score_b >= 0.3:
                    chosen_seg, split_lbl = before, f"ตัดท่อนหน้า ({marker})"
                elif score_a > score_b and score_a >= 0.3:
                    chosen_seg, split_lbl = after, f"ตัดท่อนหลัง ({marker})"
                else:
                    chosen_seg, split_lbl = text_clean, "เต็มประโยค"
            else:
                chosen_seg, split_lbl = text_clean, "เต็มประโยค"

            lbl, conf = _predict_sent(a, chosen_seg)

            results.append({
                "aspect": a, "sentiment": lbl, "confidence": round(conf, 3),
                "low_confidence": conf < LOW_CONF_THRESHOLD, "aspect_prob": aspect_prob,
                "text_segment": chosen_seg.replace("<_>", " "), "split_info": split_lbl
            })

        return {"original_text": raw_text, "text_clean": text_clean, "aspects": results, "error": None}

    # กรณีทั่วไป (ไม่มีคำขัดแย้ง) -> วิเคราะห์แบบเต็มประโยค
    for a in detected:
        aspect_prob = round(float(probs[ASPECTS.index(a)]), 3) if a in ASPECTS else None
        lbl_full, conf_full = _predict_sent(a, text_clean)
        results.append({
            "aspect": a, "sentiment": lbl_full, "confidence": round(conf_full, 3),
            "low_confidence": conf_full < LOW_CONF_THRESHOLD, "aspect_prob": aspect_prob,
            "text_segment": text_clean.replace("<_>", " "), "split_info": "เต็มประโยค"
        })

    return {"original_text": raw_text, "text_clean": text_clean, "aspects": results, "error": None}

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("review_text", "").strip()
        if input_text:
            result = analyze_text(input_text)
    return render_template("index.html", result=result, input_text=input_text)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """API endpoint แบบ JSON เผื่ออนาคตอยากต่อกับหน้าเว็บอื่น หรือระบบอื่น
    ตัวอย่างเรียกใช้: curl -X POST -H "Content-Type: application/json" \
        -d '{"text": "ห้องสะอาดดีแต่ราคาแพง"}' http://127.0.0.1:5000/api/analyze
    """
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return {"error": "กรุณาส่งข้อความในฟิลด์ 'text'"}, 400
    return analyze_text(text)


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
