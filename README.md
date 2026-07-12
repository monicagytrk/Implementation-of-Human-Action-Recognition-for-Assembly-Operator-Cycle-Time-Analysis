# Implementation of Human Action Recognition for Assembly Operator Cycle Time Analysis

## Struktur folder

```
hatrec_streamlit/
├── app.py                        ← Streamlit dashboard utama
├── inference.py                  ← Engine: cycle time & Groq
├── style.css                     ← Desain UI
├── requirements.txt
├── .streamlit/
│   └── secrets.toml              ← Groq API key 
├── models/                       ← Buat folder dan upload hasil trained model dan labels
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
