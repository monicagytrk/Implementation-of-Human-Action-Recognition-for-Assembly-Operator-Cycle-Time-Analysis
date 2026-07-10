import json
import numpy as np
import cv2
from pathlib import Path

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"  

# Minimum confidence untuk dianggap prediksi yang valid
MIN_CONFIDENCE_THRESHOLD = 0.45

# Ukuran jendela majority-vote smoothing
SMOOTHING_WINDOW_SIZE = 3

# MODEL LOADERS

def load_config(config_path: str = "models/config.json") -> dict:
    """
    Baca config.json yang dihasilkan saat training.

    Returns dict berisi: n_classes, class_names, short_names,
                         label_map, n_frames, img_size.
    """
    with open(config_path) as f:
        return json.load(f)


def load_model(model_path: str = "models/hatrec_mobilenetv2.h5"):
    """
    Load Keras model dari file .h5.

    Dipanggil via @st.cache_resource agar hanya diload sekali
    selama session Streamlit aktif.
    """
    import tensorflow as tf
    return tf.keras.models.load_model(model_path, compile=False)


def load_metrics(metrics_path: str = "models/metrics.json") -> dict:
    """
    Baca metrics.json hasil evaluasi training.
    Return dict kosong jika file tidak ditemukan.
    """
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# VIDEO UTILITIES

def sample_frames(
    video_path: str,
    n_frames: int = 8,
    img_size: tuple = (112, 112)
) -> np.ndarray | None:
    """
    Sample n_frames secara merata dari satu video clip.

    Frame diambil dari indeks yang terdistribusi merata (linspace),
    bukan dari awal saja, agar mencakup seluruh durasi video.

    Parameters
    ----------
    video_path : path ke file video
    n_frames   : jumlah frame yang diambil
    img_size   : (width, height) target resize

    Returns
    -------
    np.ndarray shape (n_frames, H, W, 3), dtype float32 [0,1]
    None jika video tidak bisa dibuka atau jumlah frame tidak cukup.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return None

    
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            fallback = frames[-1].copy() if frames else np.zeros((*img_size, 3), dtype=np.float32)
            frames.append(fallback)
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, img_size)
        frames.append(frame.astype(np.float32) / 255.0)

    cap.release()

    if len(frames) != n_frames:
        return None

    return np.array(frames, dtype=np.float32)


def get_video_duration(video_path: str) -> tuple[float, float, int]:
    """
    Baca metadata durasi video tanpa membaca frame.

    Returns
    -------
    (duration_sec, fps, total_frames)
    """
    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames / fps, fps, total_frames


def extract_representative_frame(
    video_path: str,
    position: float = 0.5,
    display_size: tuple = (320, 240)
) -> np.ndarray:
    """
    Ekstrak satu frame dari posisi tertentu dalam video.

    Dipakai untuk pratinjau visual di UI — ukuran lebih besar dari
    frame training (112px) agar tampil jelas.

    Parameters
    ----------
    position : 0.0 = awal, 0.5 = tengah, 1.0 = akhir video

    Returns
    -------
    np.ndarray uint8 RGB shape (H, W, 3)
    """
    cap          = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = int(total_frames * position)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        # Return blank frame jika gagal baca
        return np.zeros((*display_size[::-1], 3), dtype=np.uint8)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, display_size)

# SLIDING WINDOW INFERENCE

def _extract_window_frames(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    n_frames: int,
    img_size: tuple
) -> list[np.ndarray]:
    """
    Ekstrak n_frames dari satu window [start_frame, end_frame].

    Helper internal untuk predict_video_sliding — memisahkan
    logika ekstraksi frame dari logika prediksi.
    """
    indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            fallback = frames[-1].copy() if frames else np.zeros((*img_size, 3), dtype=np.float32)
            frames.append(fallback)
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, img_size)
        frames.append(frame.astype(np.float32) / 255.0)

    return frames


def predict_video_sliding(
    video_path: str,
    model,
    config: dict,
    window_sec: float = 3.0,
    stride_sec: float = 1.5,
    progress_cb=None
) -> tuple[list[dict], float]:
    """
    Jalankan sliding window inference pada video panjang.

    Setiap window = window_sec detik → 1 prediksi label.
    Window digeser sebesar stride_sec detik setiap iterasi.

    Parameters
    ----------
    video_path  : path ke file video
    model       : Keras model yang sudah diload
    config      : dict dari config.json (n_frames, img_size, class_names)
    window_sec  : durasi tiap window (detik)
    stride_sec  : jarak antar window (detik); lebih kecil = lebih detail
                  tapi lebih lambat
    progress_cb : callback(float 0..1) untuk update progress bar Streamlit

    Returns
    -------
    predictions : list of dict [{timestamp_start, timestamp_end,
                  label_id, label_name, short_name, confidence, probs}]
    duration    : total durasi video (detik)
    """
    n_frames  = config["n_frames"]
    img_size  = tuple(config["img_size"])
    names     = config["class_names"]
    short     = config["short_names"]

    # Baca metadata video
    cap          = cv2.VideoCapture(str(video_path))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps

    # Konversi detik ke frame
    win_frames    = max(int(window_sec * fps), n_frames)
    stride_frames = max(int(stride_sec  * fps), 1)

    # Hitung semua posisi awal window
    window_starts = list(range(0, max(1, total_frames - win_frames + 1), stride_frames))
    n_windows     = len(window_starts)
    predictions   = []

    for w_idx, start_f in enumerate(window_starts):
        end_f  = min(start_f + win_frames, total_frames)
        frames = _extract_window_frames(cap, start_f, end_f, n_frames, img_size)

        if len(frames) < n_frames:
            continue

        # Bentuk tensor input (1, n_frames, H, W, 3) untuk batch size 1
        X     = np.array(frames, dtype=np.float32)[np.newaxis]
        probs = model.predict(X, verbose=0)[0]
        lid   = int(np.argmax(probs))
        conf  = float(probs[lid])

        predictions.append({
            "timestamp_start": round(start_f / fps, 2),
            "timestamp_end"  : round(end_f   / fps, 2),
            "label_id"       : lid,
            "label_name"     : names[lid],
            "short_name"     : short[lid],
            "confidence"     : round(conf, 4),
            "probs"          : probs.tolist()
        })

        if progress_cb:
            progress_cb((w_idx + 1) / n_windows)

    cap.release()
    return predictions, duration

# CYCLE TIME ANALYSIS

def _smooth_predictions(
    predictions: list[dict],
    class_names: list[str],
    short_names: list[str]
) -> list[dict]:
    """
    Haluskan stream prediksi dengan majority vote per 3 window.

    Tujuan: hilangkan prediksi noise sesaat (1-2 window) yang muncul
    di tengah-tengah task yang sebenarnya konsisten.
    """
    smoothed = []
    for i, pred in enumerate(predictions):
        window = predictions[max(0, i - 1): i + 2]
        votes  = [w["label_id"] for w in window]
        majority_label = max(set(votes), key=votes.count)

        smoothed.append({
            **pred,
            "label_id"  : majority_label,
            "label_name": class_names[majority_label],
            "short_name": short_names[majority_label]
        })
    return smoothed


def _build_segments(
    smoothed: list[dict],
    class_names: list[str],
    short_names: list[str],
    min_duration: float
) -> list[dict]:
    """
    Kelompokkan prediksi berurutan dengan label yang sama menjadi segment.

    Segment yang durasinya di bawah min_duration dibuang (noise).

    Returns list of segment dict: {label_id, label_name, short_name,
                                    start, end, duration, confidence, probs}
    """
    segments = []
    i = 0

    while i < len(smoothed):
        curr_label = smoothed[i]["label_id"]
        j = i

        while j < len(smoothed) and smoothed[j]["label_id"] == curr_label:
            j += 1

        seg_window    = smoothed[i:j]
        seg_start     = seg_window[0]["timestamp_start"]
        seg_end       = seg_window[-1]["timestamp_end"]
        seg_duration  = seg_end - seg_start
        seg_confidence= float(np.mean([s["confidence"] for s in seg_window]))
        seg_probs     = np.mean([s["probs"] for s in seg_window], axis=0).tolist()

        if seg_duration >= min_duration:
            segments.append({
                "label_id"  : curr_label,
                "label_name": class_names[curr_label],
                "short_name": short_names[curr_label],
                "start"     : round(seg_start, 2),
                "end"       : round(seg_end, 2),
                "duration"  : round(seg_duration, 2),
                "confidence": round(seg_confidence, 4),
                "probs"     : seg_probs
            })
        i = j

    return segments


def _compute_task_stats(
    segments: list[dict],
    class_names: list[str],
    short_names: list[str]
) -> dict:
    """
    Hitung statistik durasi per task dari daftar segment.

    Returns dict keyed by label_id, berisi count, avg, min, max, total.
    """
    task_stats = {}

    for label_id, label_name in enumerate(class_names):
        task_segs = [s for s in segments if s["label_id"] == label_id]
        if not task_segs:
            continue

        durations = [s["duration"] for s in task_segs]
        task_stats[label_id] = {
            "label_name": label_name,
            "short_name": short_names[label_id],
            "count"     : len(task_segs),
            "total_sec" : round(sum(durations), 2),
            "avg_sec"   : round(float(np.mean(durations)), 2),
            "min_sec"   : round(float(np.min(durations)),  2),
            "max_sec"   : round(float(np.max(durations)),  2),
            "segments"  : task_segs
        }

    return task_stats


def _count_complete_cycles(
    segments: list[dict],
    n_classes: int
) -> tuple[int, list[float]]:
    """
    Hitung jumlah cycle lengkap (task 0 → task n_classes-1).

    Menggunakan sliding window O(n) — setiap posisi dicek sekali.
    Satu cycle = semua n_classes task muncul dalam window 2× ukuran kelas.

    Returns
    -------
    n_cycles     : jumlah cycle lengkap yang terdeteksi
    cycle_starts : timestamp detik awal tiap cycle
    """
    task_order   = [s["label_id"] for s in segments]
    all_tasks    = set(range(n_classes))
    window_size  = n_classes * 2   # window cukup luas untuk satu cycle
    n_cycles     = 0
    cycle_starts = []
    i            = 0

    while i <= len(task_order) - n_classes:
        window = set(task_order[i: i + window_size])
        if all_tasks.issubset(window):
            n_cycles += 1
            cycle_starts.append(segments[i]["start"])
            i += n_classes
        else:
            i += 1

    return n_cycles, cycle_starts


def analyze_cycle_time(
    predictions: list[dict],
    class_names: list[str],
    short_names: list[str],
    min_task_duration: float = 1.0,
    idle_gap: float = 2.0
) -> tuple[list[dict], dict, list[dict]]:
    """
    Pipeline lengkap: stream prediksi → cycle time statistics.

    Langkah internal:
        1. Smooth  — majority vote per 3 window
        2. Segment — gabungkan prediksi label sama → segment
        3. Stats   — hitung avg/min/max per task
        4. Cycles  — deteksi cycle lengkap (O(n))

    Parameters
    ----------
    predictions       : output dari predict_video_sliding
    class_names       : nama kelas lengkap
    short_names       : nama kelas pendek untuk display
    min_task_duration : segment lebih pendek dari ini dibuang (detik)
    idle_gap          : tidak dipakai langsung, disiapkan untuk ekstensi

    Returns
    -------
    task_segments : list segment yang valid
    cycle_summary : dict ringkasan statistik
    timeline      : list {time, label_id, label_name, confidence} untuk plot
    """
    if not predictions:
        return [], {}, []

    # Step 1: Smoothing
    smoothed = _smooth_predictions(predictions, class_names, short_names)

    # Step 2: Segmentasi
    task_segments = _build_segments(smoothed, class_names, short_names, min_task_duration)

    # Step 3: Statistik per task
    task_stats = _compute_task_stats(task_segments, class_names, short_names)

    # Step 4: Hitung cycle lengkap
    n_cycles, cycle_starts = _count_complete_cycles(task_segments, len(class_names))

    # Hitung working vs idle time
    total_duration = predictions[-1]["timestamp_end"]
    working_time   = sum(s["duration"] for s in task_segments)
    idle_time      = max(0.0, total_duration - working_time)
    efficiency_pct = (working_time / total_duration * 100) if total_duration > 0 else 0.0

    cycle_summary = {
        "total_duration_sec": round(total_duration, 2),
        "working_time_sec"  : round(working_time,   2),
        "idle_time_sec"     : round(idle_time,       2),
        "efficiency_pct"    : round(efficiency_pct,  1),
        "n_cycles_detected" : n_cycles,
        "cycle_starts"      : cycle_starts,
        "n_task_segments"   : len(task_segments),
        "task_stats"        : task_stats
    }

    # Timeline untuk plotting
    timeline = [
        {
            "time"      : p["timestamp_start"],
            "label_id"  : p["label_id"],
            "label_name": p["short_name"],
            "confidence": p["confidence"]
        }
        for p in smoothed
    ]

    return task_segments, cycle_summary, timeline

# AI SUMMARY (GROQ)

def _build_groq_prompt(
    cycle_summary: dict,
    short_names: list[str]
) -> str:
    """
    Buat teks prompt untuk Groq LLaMA dari hasil cycle_summary.

    Dipisahkan dari generate_ai_summary agar mudah diuji secara mandiri.
    """
    task_stats = cycle_summary.get("task_stats", {})
    task_lines = "\n".join(
        f"  - {stats['label_name']}: avg {stats['avg_sec']:.1f}s "
        f"(min {stats['min_sec']:.1f}s, max {stats['max_sec']:.1f}s, "
        f"n={stats['count']})"
        for stats in task_stats.values()
    )

    return f"""Kamu adalah analis efisiensi manufaktur industrial.
Berikut hasil analisis cycle time seorang operator dari rekaman video:

RINGKASAN:
- Total durasi video : {cycle_summary['total_duration_sec']:.1f} detik
- Waktu bekerja      : {cycle_summary['working_time_sec']:.1f} detik
- Waktu idle         : {cycle_summary['idle_time_sec']:.1f} detik
- Efisiensi operator : {cycle_summary['efficiency_pct']:.1f}%
- Cycle terdeteksi   : {cycle_summary['n_cycles_detected']}

WAKTU PER TASK:
{task_lines}

Berikan dalam Bahasa Indonesia:
1. Ringkasan singkat performa operator (2-3 kalimat)
2. Task mana yang paling memakan waktu dan kemungkinan penyebabnya
3. 2-3 rekomendasi konkret untuk meningkatkan efisiensi
4. Penilaian keseluruhan: Excellent / Good / Needs Improvement

Format: singkat dan langsung to the point."""


def generate_ai_summary(
    cycle_summary: dict,
    class_names: list[str],
    short_names: list[str],
    baseline_metrics: dict,
    groq_api_key: str
) -> str:
    """
    Generate ringkasan & rekomendasi efisiensi via Groq LLaMA-3.

    Parameters
    ----------
    cycle_summary    : output dari analyze_cycle_time
    class_names      : nama kelas lengkap
    short_names      : nama kelas pendek
    baseline_metrics : dict dari metrics.json (untuk konteks)
    groq_api_key     : API key Groq (gratis di console.groq.com)

    Returns
    -------
    str — teks analisis dari LLaMA, atau pesan error jika gagal.
    """
    try:
        from groq import Groq

        client = Groq(api_key=groq_api_key)
        prompt = _build_groq_prompt(cycle_summary, short_names)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.4
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"AI summary tidak tersedia: {str(e)}"
