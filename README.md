# ระบบ ABSA ภาษาไทย — Flask Web UI

กรอกข้อความรีวิว → ระบบตรวจแง่มุม (aspect) + วิเคราะห์อารมณ์ (sentiment) แยกตามแง่มุม
ใช้โมเดล WangchanBERTa 2 ตัว (aspect + sentiment) ที่เทรนไว้แล้ว

## โครงสร้างโปรเจกต์

```
absa_flask_app/
├── app.py                 ← Flask backend (โหลดโมเดล + วิเคราะห์)
├── requirements.txt
├── templates/
│   └── index.html         ← หน้าเว็บ
├── static/
│   └── style.css
└── models/                ← ต้อง(สร้างเอง (ดูขั้นตอนที่ 2))
    ├── aspect_final/
    └── sentiment_final_v2/
```

---

## ขั้นตอนที่ 1 — เตรียมเครื่องใน VS Code

1. เปิด VS Code → ติดตั้ง extension **Python** (ของ Microsoft) ถ้ายังไม่มี
2. เปิดโฟลเดอร์ `absa_flask_app` นี้ใน VS Code (**File → Open Folder**)
3. เปิด Terminal ใน VS Code (**Terminal → New Terminal**) แล้วสร้าง virtual environment:

```bash
python -m venv venv
```

4. เปิดใช้งาน venv:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

   เมื่อเปิดสำเร็จ จะเห็น `(venv)` ขึ้นหน้า prompt ใน terminal — **VS Code มักถามว่าจะใช้ interpreter นี้เป็นค่าเริ่มต้นไหม กด "Yes"**

5. ติดตั้งไลบรารีทั้งหมด:

```bash
pip install -r requirements.txt
```

ขั้นตอนนี้จะใช้เวลาสักครู่ (โดยเฉพาะ `torch` ที่ไฟล์ใหญ่) รอจนเสร็จ

---

## ขั้นตอนที่ 2 — เอาโมเดลที่เทรนไว้มาไว้ในเครื่อง

โมเดลตอนนี้อยู่บน Google Drive (ที่ใช้กับ Colab) ต้อง**ดาวน์โหลดมาไว้ในเครื่องท้องถิ่น** ก่อน เพราะ Flask รันในเครื่องคุณเอง ไม่ได้เชื่อมกับ Drive อัตโนมัติ

**สำคัญ — ฝั่ง Aspect: tokenizer แยกโฟลเดอร์จากโมเดล** จากที่เห็นใน Drive จะมี 2 โฟลเดอร์แยกกันใต้ `Data/Model/Aspect/`:
```
Data/Model/Aspect/
├── aspect_extraction_11.07.2026/           ← ตัวโมเดล (config.json, model.safetensors)
└── tokenizer_aspect_extraction_11.07.2026/ ← ตัว tokenizer (tokenizer_config.json, tokenizer.json)
```
**ต้องดาวน์โหลดทั้ง 2 โฟลเดอร์นี้แยกกัน** (คลิกขวาแต่ละโฟลเดอร์ → Download) แล้วแตกไฟล์ทั้งคู่

ฝั่ง Sentiment ปกติจะรวมอยู่โฟลเดอร์เดียว (เช่น `sentiment_extraction_12...` ที่มีทั้ง `model.safetensors` และ `tokenizer.json` ในที่เดียวกัน) ดาวน์โหลดโฟลเดอร์เดียวพอ — **แต่เช็คให้แน่ใจก่อน** ว่าโฟลเดอร์ sentiment ของคุณเป็นแบบนี้จริงไหม (เปิดดูใน Drive ว่ามีไฟล์ `tokenizer_config.json` อยู่ในโฟลเดอร์เดียวกับ `model.safetensors` หรือแยกอยู่คนละที่แบบ aspect)

วางไว้ในโปรเจกต์ **โดยไม่ต้องเปลี่ยนชื่อโฟลเดอร์เลย** (สร้างแค่โฟลเดอร์ `models/Aspect/` และ `models/Sentiment/` มาครอบ):

```
absa_flask_app/
└── models/
    ├── Aspect/
    │   ├── aspect_extraction_11.07.2026/              ← จาก Drive ตรงๆ ไม่เปลี่ยนชื่อ
    │   ├── tokenizer_aspect_extraction_11.07.2026/    ← จาก Drive ตรงๆ ไม่เปลี่ยนชื่อ
    │   └── aspect_thresholds.json                      ← ก็อปมาวางเพิ่ม (ดูหมายเหตุด้านล่าง)
    └── Sentiment/
        └── sentiment_extraction_v2/                     ← ชื่อจริงของคุณอาจต่างจากนี้ เช่น sentiment_extraction_12...
```

`app.py` ตั้งค่าเริ่มต้นให้ตรงกับโครงสร้างนี้ไว้แล้ว **ถ้าโฟลเดอร์ของคุณอยู่ตำแหน่ง/ชื่ออื่น ให้แก้ path ใน `app.py` แทนการเปลี่ยนชื่อโฟลเดอร์** (แก้โค้ดปลอดภัยกว่าเปลี่ยนชื่อไฟล์ที่ดาวน์โหลดมา เสี่ยงพิมพ์ผิด/ย้ายไฟล์ไม่ครบน้อยกว่า):
```python
ASPECT_MODEL_DIR = "./models/Aspect/aspect_extraction_11.07.2026"
ASPECT_TOKENIZER_DIR = "./models/Aspect/tokenizer_aspect_extraction_11.07.2026"
SENTIMENT_MODEL_DIR = "./models/Sentiment/sentiment_extraction_v2"       # แก้ให้ตรงชื่อจริง
SENTIMENT_TOKENIZER_DIR = "./models/Sentiment/sentiment_extraction_v2"   # โฟลเดอร์เดียวกันถ้ารวมกันอยู่
```

**ถ้า `aspect_thresholds.json` ไม่ได้อยู่ในโฟลเดอร์ไหนเลยข้างต้น** (แยกเก็บไว้คนละที่ตามที่เจอปัญหาระหว่างสัปดาห์) ให้คัดลอกมาวางไว้ในโฟลเดอร์ `models/Aspect/` เพิ่ม หรือแก้ `ASPECT_THRESHOLDS_PATH` ให้ชี้ path จริง

**เช็คก่อนรันเสมอ** — เข้าไปดูในแต่ละโฟลเดอร์ด้วยตาว่ามี `model.safetensors` อยู่จริง (จากที่เจอปัญหาไฟล์นี้หายมาก่อนหน้านี้ในสัปดาห์)

---


## ขั้นตอนที่ 3 — รันแอป

```bash
python app.py
```

ถ้า path ถูกต้องครบ จะเห็น:
```
✅ ASPECT_MODEL_DIR = ./models/aspect_final
✅ ASPECT_TOKENIZER_DIR = ./models/aspect_tokenizer
✅ SENTIMENT_MODEL_DIR = ./models/sentiment_final_v2
✅ SENTIMENT_TOKENIZER_DIR = ./models/sentiment_final_v2
✅ ASPECT_THRESHOLDS_PATH = ./models/aspect_final/aspect_thresholds.json
โหลดโมเดลครบทั้งหมด พร้อมใช้งาน ✅
 * Running on http://127.0.0.1:5000
```

เปิดเบราว์เซอร์ไปที่ **http://127.0.0.1:5000** — จะเห็นหน้ากรอกข้อความพร้อมใช้งาน

**ถ้าเห็น `❌`** — แปลว่า path ผิด กลับไปเช็คขั้นตอนที่ 2 ให้ path ใน `app.py` ตรงกับตำแหน่งโฟลเดอร์จริงในเครื่องคุณ (แก้ตรง `ASPECT_MODEL_DIR`, `SENTIMENT_MODEL_DIR`, `ASPECT_THRESHOLDS_PATH` บนสุดของไฟล์)

---

## กด F5 ดีบักใน VS Code (ทางเลือก)

แทนที่จะพิมพ์ `python app.py` ใน terminal ทุกครั้ง ตั้งค่าให้กด F5 รันได้เลย:

1. กด `Ctrl+Shift+D` (Run and Debug panel) → **create a launch.json file** → เลือก **Flask**
2. VS Code จะสร้าง `.vscode/launch.json` ให้อัตโนมัติ
3. กด F5 ได้เลยครั้งต่อไป — วางจุด breakpoint ในโค้ดได้ด้วยถ้าต้องการดีบัก

---

## ใช้งานผ่าน API (เผื่อจะต่อกับระบบอื่น)

นอกจากหน้าเว็บ มี endpoint แบบ JSON ให้เรียกจากที่อื่นได้ด้วย:

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "ห้องสะอาดดีแต่ราคาแพง"}'
```

จะได้ผลลัพธ์กลับมาเป็น JSON แทนหน้า HTML — เหมาะถ้าจะเอาไปต่อกับ MongoDB pipeline (Part 9) หรือ frontend อื่นในอนาคต

---

## ปัญหาที่พบบ่อย

**`ModuleNotFoundError: No module named 'thaixtransformers'`**
ไม่บังคับต้องมี — โค้ดมี fallback ให้ใช้ preprocessing แบบพื้นฐานแทนได้ (จะขึ้นคำเตือนแต่รันต่อได้) แต่ถ้าอยากได้ผลตรงกับตอนเทรนเป๊ะที่สุด ติดตั้งด้วย `pip install thaixtransformers`

**เครื่องไม่มี GPU** — ไม่เป็นไร โค้ดตรวจจับอัตโนมัติแล้วใช้ CPU แทน (`torch.device("cuda" if ... else "cpu")`) ช้ากว่า GPU แต่สำหรับข้อความสั้นๆ ทีละข้อความ ใช้เวลาไม่ถึงวินาทีอยู่ดี

**พอร์ต 5000 ถูกใช้งานอยู่แล้ว** — แก้บรรทัดสุดท้ายของ `app.py` เปลี่ยน `port=5000` เป็นเลขอื่น เช่น `port=5001`

**อยากเปลี่ยน path โมเดลโดยไม่แก้โค้ด** — ตั้ง environment variable ก่อนรันแทนได้:
```bash
export ASPECT_MODEL_DIR="/path/จริง/aspect_extraction_11.07.2026"
export ASPECT_TOKENIZER_DIR="/path/จริง/tokenizer_aspect_extraction_11.07.2026"
export SENTIMENT_MODEL_DIR="/path/จริง/sentiment_final_v2"
export SENTIMENT_TOKENIZER_DIR="/path/จริง/sentiment_final_v2"
python app.py
```
