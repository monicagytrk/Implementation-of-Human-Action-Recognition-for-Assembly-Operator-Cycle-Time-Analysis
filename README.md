# 🏭 HATRec Cycle Time Monitor v3

AI-powered cycle time monitoring untuk proses assembly manual.

## Struktur folder

```
hatrec_streamlit/
├── app.py                        ← Streamlit dashboard utama
├── inference.py                  ← Engine: sliding window, cycle time, Groq
├── style.css                     ← Tema biru muda
├── requirements.txt
├── .streamlit/
│   └── secrets.toml              ← Groq API key (jangan di-upload ke GitHub)
├── models/                       ← BUAT FOLDER INI, copy dari Google Drive
│   ├── hatrec_mobilenetv2.h5
│   ├── config.json
│   └── metrics.json
└── README.md
```

## Setup

### 1. Copy file model dari Google Drive
Download folder `HATRec_Model` dari Google Drive, lalu copy ke `models/`:
```
models/hatrec_mobilenetv2.h5
models/config.json
models/metrics.json
```

### 2. Install dependencies
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 3. (Opsional) Setup Groq API key
Edit file `.streamlit/secrets.toml`:
```toml
groq_api_key = "gsk_isi_api_key_kamu_disini"
```
Dapatkan API key gratis di https://console.groq.com

### 4. Jalankan
```bash
streamlit run app.py
```
Buka browser: http://localhost:8501

## Perubahan v3 dari v2
- Tidak ada sidebar — info model tampil di header
- Parameter inferensi hardcoded (window=3s, stride=1.5s, min_dur=1.5s)
- Groq API key di backend (secrets.toml), tidak di UI
- Preview video 1/3 ukuran, posisi center
- Fix error legend parameter di donut chart
