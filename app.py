import os
import time
import tempfile

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import cv2
from pathlib import Path

# Page config
st.set_page_config(
    page_title="HATRec — Cycle Time Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inference parameters
WINDOW_SEC  = 2.0   
STRIDE_SEC  = 0.5   
MIN_DUR_SEC = 0.5   

# Model paths
MODEL_PATHS = {
    "model"  : "models/hatrec_mobilenetv2.h5",
    "config" : "models/config.json",
    "metrics": "models/metrics.json"
}

# Task colors
TASK_COLORS = [
    "#1565C0", "#42A5F5", "#00897B", "#F4511E",
    "#8E24AA", "#43A047", "#FB8C00"
]


# SETUP & LOADERS

def _load_css(path: str = "style.css") -> None:
    """Inject CSS dari file eksternal."""
    try:
        with open(path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


_load_css()


@st.cache_resource
def load_model_cached(model_path: str):
    """Load Keras model sekali lalu cache selama session aktif."""
    from inference import load_model
    return load_model(model_path)


@st.cache_data
def load_config_cached(config_path: str) -> dict:
    from inference import load_config
    return load_config(config_path)


@st.cache_data
def load_metrics_cached(metrics_path: str) -> dict:
    from inference import load_metrics
    return load_metrics(metrics_path)


def _get_groq_key() -> str:
    """
    Ambil Groq API key dari Streamlit secrets (backend).
    File: .streamlit/secrets.toml
    Isi:  groq_api_key = "gsk_..."
    Return empty string jika tidak ada.
    """
    try:
        return st.secrets["groq_api_key"]
    except Exception:
        return ""


def check_model_files() -> list[str]:
    """Return list file model yang tidak ditemukan."""
    labels = {
        MODEL_PATHS["model"]  : "Model Keras (.h5)",
        MODEL_PATHS["config"] : "config.json",
        MODEL_PATHS["metrics"]: "metrics.json"
    }
    return [label for path, label in labels.items() if not Path(path).exists()]

# PLOT HELPERS

def _base_layout(**kwargs) -> dict:
    """Layout dasar Plotly — konsisten di semua chart."""
    return dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#212121"),
        margin=dict(l=40, r=20, t=30, b=40),
        **kwargs
    )


def plot_timeline(
    timeline: list[dict],
    short_names: list[str],
    total_dur: float
) -> go.Figure | None:
    """Bar chart durasi per task dari timeline prediksi."""
    if not timeline:
        return None

    # Hitung total durasi per task
    task_durations = {}
    for i, short in enumerate(short_names):
        task_durations[short] = 0.0

    # Hitung durasi dari selisih timestamp antar prediksi
    for j in range(len(timeline)):
        label = timeline[j]["label_name"]
        if j < len(timeline) - 1:
            dur = timeline[j+1]["time"] - timeline[j]["time"]
        else:
            dur = STRIDE_SEC
        if label in task_durations:
            task_durations[label] += dur

    labels   = list(task_durations.keys())
    durations = [round(task_durations[l], 2) for l in labels]
    colors   = [TASK_COLORS[i % 7] for i in range(len(labels))]

    fig = go.Figure(go.Bar(
        x=labels,
        y=durations,
        marker_color=colors,
        text=[f"{v:.1f}s" for v in durations],
        textposition="outside"
    ))
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#212121"),
        height=350,
        margin=dict(l=40, r=20, t=30, b=60),
        xaxis=dict(title="Jenis Task", gridcolor="#E3F2FD"),
        yaxis=dict(title="Total Durasi (detik)", gridcolor="#E3F2FD"),
        showlegend=False
    )
    return fig


def plot_task_duration_bar(task_stats: dict) -> go.Figure | None:
    """Bar chart rata-rata durasi per task."""
    if not task_stats:
        return None

    tids   = sorted(task_stats.keys())
    labels = [task_stats[t]["short_name"] for t in tids]
    avgs   = [task_stats[t]["avg_sec"]    for t in tids]

    fig = go.Figure(go.Bar(
        x=labels, y=avgs,
        marker_color=[TASK_COLORS[i % 7] for i in tids],
        text=[f"{v:.1f}s" for v in avgs],
        textposition="outside"
    ))
    fig.update_layout(
        **_base_layout(height=300),
        yaxis=dict(title="Detik", gridcolor="#E3F2FD"),
        showlegend=False
    )
    return fig


def plot_working_idle_donut(working_sec: float, idle_sec: float) -> go.Figure:
    total = working_sec + idle_sec
    eff   = working_sec / total * 100 if total > 0 else 0

    fig = go.Figure(go.Pie(
        values=[working_sec, idle_sec],
        labels=["Working", "Idle"],
        hole=0.62,
        marker_colors=["#1565C0", "#BBDEFB"],
        textinfo="percent+label",
        hovertemplate="%{label}: %{value:.1f}s<extra></extra>"
    ))
    fig.add_annotation(
        text=f"<b>{eff:.0f}%</b><br><span style='font-size:11px'>efisiensi</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="#1565C0")
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#212121"),
        height=280,
        margin=dict(l=20, r=20, t=20, b=40),
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=-0.2,
        legend_xanchor="center",
        legend_x=0.5
    )
    return fig


def plot_confidence_per_task(
    task_stats: dict,
    short_names: list[str]
) -> go.Figure | None:
    """Bar chart rata-rata confidence prediksi per task."""
    if not task_stats:
        return None

    tids   = sorted(task_stats.keys())
    confs  = [
        float(np.mean([s["confidence"] for s in task_stats[t]["segments"]]))
        for t in tids
    ]
    labels = [task_stats[t]["short_name"] for t in tids]
    colors = ["#43A047" if c >= 0.75 else "#FB8C00" if c >= 0.55 else "#E53935"
              for c in confs]

    fig = go.Figure(go.Bar(
        x=labels, y=confs,
        marker_color=colors,
        text=[f"{c:.0%}" for c in confs],
        textposition="outside"
    ))
    fig.add_hline(y=0.75, line_dash="dash", line_color="#1565C0",
                  annotation_text="75% threshold")
    fig.update_layout(
        **_base_layout(height=280),
        yaxis=dict(range=[0, 1.1], tickformat=".0%", gridcolor="#E3F2FD"),
        showlegend=False
    )
    return fig


def plot_f1_radar(metrics: dict, short_names: list[str]) -> go.Figure | None:
    """Radar chart F1 per class dari training metrics."""
    f1_dict = metrics.get("f1_per_class", {})
    if not f1_dict:
        return None

    labels = list(f1_dict.keys())
    values = list(f1_dict.values())
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed, theta=labels_closed,
        fill="toself",
        fillcolor="rgba(21,101,192,0.15)",
        line=dict(color="#1565C0", width=2),
        marker=dict(size=6, color="#1565C0")
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="#E3F2FD"),
            angularaxis=dict(gridcolor="#E3F2FD")
        ),
        paper_bgcolor="white",
        height=300,
        margin=dict(l=40, r=40, t=30, b=30),
        showlegend=False
    )
    return fig


def plot_prediction_distribution(
    predictions: list[dict],
    short_names: list[str]
) -> go.Figure | None:
    """Bar chart distribusi label yang diprediksi di video ini."""
    if not predictions:
        return None

    pred_names = [short_names[p["label_id"]] for p in predictions]
    dist_df    = pd.Series(pred_names).value_counts().reset_index()
    dist_df.columns = ["Task", "Count"]

    fig = px.bar(
        dist_df, x="Task", y="Count",
        color="Task",
        color_discrete_sequence=TASK_COLORS,
        text="Count"
    )
    fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#212121"),
    height=250,
    margin=dict(l=20, r=20, t=10, b=40),
    showlegend=False
    )
    return fig

# TAB RENDERERS


def _segment_card_html(seg: dict, badge_color: str) -> str:
    """HTML card untuk satu task segment di Tab 2."""
    conf      = seg["confidence"]
    bar_color = "#43A047" if conf >= 0.75 else "#FB8C00" if conf >= 0.55 else "#E53935"
    return f"""
    <div style="margin-top:6px;">
      <span style="background:{badge_color};color:white;border-radius:12px;
                   padding:2px 10px;font-size:0.75rem;font-weight:700;">
        Task {seg['label_id']}
      </span>
      <div style="font-size:0.8rem;color:#37474F;margin-top:4px;font-weight:600;">
        {seg['short_name']}
      </div>
      <div style="font-size:0.75rem;color:#78909C;">
        {seg['start']:.1f}s → {seg['end']:.1f}s ({seg['duration']:.1f}s)
      </div>
      <div style="font-size:0.75rem;color:#78909C;">Confidence: {conf:.1%}</div>
      <div class="conf-bar-wrap">
        <div class="conf-bar-fill"
             style="width:{conf*100:.0f}%;background:{bar_color};">
        </div>
      </div>
    </div>"""


def _extract_frame_at(video_path: str, timestamp_sec: float) -> "np.ndarray | None":
    """Ekstrak satu frame dari timestamp tertentu (detik)."""
    cap   = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_sec * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame, (280, 210))


def render_tab_cycle_time(
    timeline: list[dict],
    task_stats: dict,
    cycle_summary: dict,
    short_names: list[str],
    total_dur: float
) -> None:
    """Tab 1: Cycle Time Analysis."""

    # Timeline
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Timeline Prediksi Task</div>',
                unsafe_allow_html=True)
    fig_tl = plot_timeline(timeline, short_names, total_dur)
    if fig_tl:
        st.plotly_chart(fig_tl, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Bar + Donut
    col_bar, col_donut = st.columns([3, 2])
    with col_bar:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⏱️ Avg Cycle Time per Task</div>',
                    unsafe_allow_html=True)
        fig_bar = plot_task_duration_bar(task_stats)
        if fig_bar:
            st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_donut:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 Working vs Idle</div>',
                    unsafe_allow_html=True)
        fig_donut = plot_working_idle_donut(
            cycle_summary["working_time_sec"],
            cycle_summary["idle_time_sec"]
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tabel statistik
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Detail Statistik per Task</div>',
                unsafe_allow_html=True)

    if task_stats:
        rows = [
            {
                "Task"      : f"[{tid}] {s['label_name']}",
                "Count"     : s["count"],
                "Avg (s)"   : f"{s['avg_sec']:.2f}",
                "Min (s)"   : f"{s['min_sec']:.2f}",
                "Max (s)"   : f"{s['max_sec']:.2f}",
                "Total (s)" : f"{s['total_sec']:.2f}",
            }
            for tid, s in sorted(task_stats.items())
        ]
        df_table = pd.DataFrame(rows)
        st.dataframe(df_table, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download CSV",
            data=df_table.to_csv(index=False),
            file_name="cycle_time_results.csv",
            mime="text/csv"
        )
    else:
        st.info("Tidak ada task segment terdeteksi.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_tab_per_task(
    task_segs: list[dict],
    video_path: str
) -> None:
    """Tab 2: Per-Task Breakdown."""

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Frame per Task Segment</div>',
                unsafe_allow_html=True)
    st.caption("Frame diambil dari tengah setiap segment yang terdeteksi.")

    if not task_segs:
        st.info("Tidak ada segment terdeteksi.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    MAX_DISPLAY = 14
    segs_show = sorted(task_segs, key=lambda s: s["label_id"])[:MAX_DISPLAY]
    n_cols      = min(4, len(segs_show))
    cols        = st.columns(n_cols)

    for i, seg in enumerate(segs_show):
        with cols[i % n_cols]:
            mid_time = (seg["start"] + seg["end"]) / 2
            frame    = _extract_frame_at(video_path, mid_time)
            if frame is not None:
                st.image(frame, width=280)
            badge_color = TASK_COLORS[seg["label_id"] % 7]
            st.markdown(_segment_card_html(seg, badge_color), unsafe_allow_html=True)

    if len(task_segs) > MAX_DISPLAY:
        st.caption(f"Menampilkan {MAX_DISPLAY} dari {len(task_segs)} segment.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Tabel semua segment
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Semua Task Segment</div>',
                unsafe_allow_html=True)
    if task_segs:
        seg_rows = [
            {
                "#"           : i + 1,
                "Task"        : f"[{s['label_id']}] {s['short_name']}",
                "Start (s)"   : s["start"],
                "End (s)"     : s["end"],
                "Duration (s)": s["duration"],
                "Confidence"  : f"{s['confidence']:.1%}"
            }
            for i, s in enumerate(task_segs)
        ]
        st.dataframe(pd.DataFrame(seg_rows), use_container_width=True,
                     hide_index=True, height=300)
    st.markdown('</div>', unsafe_allow_html=True)


def render_tab_evaluation(
    metrics: dict,
    task_stats: dict,
    predictions: list[dict],
    short_names: list[str]
) -> None:
    """Tab 3: Model Evaluation."""

    col_r, col_c = st.columns(2)
    with col_r:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">F1 Score per Task - Training</div>',
                    unsafe_allow_html=True)
        st.caption("Dari evaluasi model di test set saat training.")
        fig_radar = plot_f1_radar(metrics, short_names)
        if fig_radar:
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Data f1_per_class tidak tersedia di metrics.json.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confidence per Task</div>',
                    unsafe_allow_html=True)
        st.caption("Rata-rata confidence prediksi model di video yang diupload.")
        fig_conf = plot_confidence_per_task(task_stats, short_names)
        if fig_conf:
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("Tidak ada task segment terdeteksi.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Training metrics summary
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Training Metrics Summary</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy",        f"{metrics.get('test_accuracy',    0):.1%}")
    m2.metric("Test F1 (weighted)",   f"{metrics.get('test_f1_weighted', 0):.4f}")
    m3.metric("Best Val Acc Phase 1", f"{metrics.get('best_val_acc_p1',  0):.1%}")
    m4.metric("Best Val Acc Phase 2", f"{metrics.get('best_val_acc_p2',  0):.1%}")

    f1_dict = metrics.get("f1_per_class", {})
    if f1_dict:
        st.markdown("**F1 Score per Task:**")
        f1_rows = [
            {
                "Task"    : name,
                "F1 Score": f"{val:.4f}",
                "Status"  : "Good" if val >= 0.8 else "⚠️ Not good" if val >= 0.6 else "❌"
            }
            for name, val in f1_dict.items()
        ]
        st.dataframe(pd.DataFrame(f1_rows), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Distribusi prediksi
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Distribusi Prediksi di Video Ini</div>',
                unsafe_allow_html=True)
    fig_dist = plot_prediction_distribution(predictions, short_names)
    if fig_dist:
        st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_tab_ai_summary(
    cycle_summary: dict,
    task_stats: dict,
    class_names: list,
    short_names: list,
    metrics: dict,
    groq_key: str
) -> None:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 AI Analysis — Groq LLaMA</div>',
                unsafe_allow_html=True)

    if not groq_key:
        st.warning(
            "Groq API key belum dikonfigurasi.\n\n"
            "Tambahkan di file `.streamlit/secrets.toml`:\n"
            "```\ngroq_api_key = \"gsk_...\"\n```"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Auto-generate — cache per video berbeda
    cache_key = f"ai_{cycle_summary['total_duration_sec']}_{cycle_summary['n_cycles_detected']}_{cycle_summary['working_time_sec']}"

    if cache_key not in st.session_state:
        with st.spinner("Menganalisis dengan Groq LLaMA..."):
            from inference import _build_groq_prompt
            try:
                import httpx
                from groq import Groq
                http_client = httpx.Client()
                client      = Groq(api_key=groq_key, http_client=http_client)
                prompt      = _build_groq_prompt(cycle_summary, short_names)
                response    = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=600,
                    temperature=0.4
                )
                st.session_state[cache_key] = response.choices[0].message.content
            except Exception as e:
                st.session_state[cache_key] = f"AI summary tidak tersedia: {str(e)}"

    ai_text = st.session_state[cache_key].replace('\n', '<br>')
    st.markdown(
        f'<div class="ai-box">{ai_text}</div>',
        unsafe_allow_html=True
    )

    with st.expander("📋 Data yang dianalisis AI"):
        st.json({
            "total_duration_sec": cycle_summary["total_duration_sec"],
            "working_time_sec"  : cycle_summary["working_time_sec"],
            "idle_time_sec"     : cycle_summary["idle_time_sec"],
            "efficiency_pct"    : cycle_summary["efficiency_pct"],
            "n_cycles"          : cycle_summary["n_cycles_detected"],
            "task_stats"        : {
                str(k): {"name": v["label_name"],
                         "avg_sec": v["avg_sec"],
                         "count": v["count"]}
                for k, v in task_stats.items()
            }
        })
    st.markdown('</div>', unsafe_allow_html=True)


# HEADER & METRIC CARDS

def _render_header(metrics: dict, config: dict) -> None:
    """
    Render header dengan info model — menggantikan sidebar.
    Semua informasi penting tampil di bawah judul.
    """
    test_acc   = metrics.get("test_accuracy", 0)
    n_classes  = config.get("n_classes", 7)
    n_frames   = config.get("n_frames",  4)
    img_size   = config.get("img_size",  [112, 112])

    st.markdown(f"""
    <div class="app-header">
      <h1>HATRec Cycle Time Monitor</h1>
      <p>Human Action Recognition for Assembly Operator Cycle Time Analysis</p>
      <div class="header-info">
        <div class="header-info-item">
          <span>Model</span>
          MobileNetV2 + LSTM
        </div>
        <div class="header-info-item">
          <span>Kelas</span>
          {n_classes} assembly tasks
        </div>
        <div class="header-info-item">
          <span>Input</span>
          {n_frames} frames × {img_size[0]}px
        </div>
        <div class="header-info-item">
          <span>Test accuracy</span>
          {test_acc:.1%}
        </div>
        <div class="header-info-item">
          <span>Window / Stride</span>
          {WINDOW_SEC}s / {STRIDE_SEC}s
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def _render_metric_cards(cycle_summary: dict) -> None:
    """Baris kartu metrik ringkasan di atas tabs."""
    wt  = cycle_summary["working_time_sec"]
    it  = cycle_summary["idle_time_sec"]
    eff = cycle_summary["efficiency_pct"]
    nc  = cycle_summary["n_cycles_detected"]
    td  = cycle_summary["total_duration_sec"]
    eff_cls = "green" if eff >= 75 else "orange" if eff >= 50 else "red"

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-val">{td:.1f}s</div>
        <div class="metric-label">⏱Total Durasi Video</div>
      </div>
      <div class="metric-card green">
        <div class="metric-val green">{wt:.1f}s</div>
        <div class="metric-label">Total Working Time</div>
      </div>
      <div class="metric-card orange">
        <div class="metric-val orange">{it:.1f}s</div>
        <div class="metric-label">Total Idle Time</div>
      </div>
      <div class="metric-card purple">
        <div class="metric-val purple">{nc}</div>
        <div class="metric-label">Cycle Terdeteksi</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# MAIN

def main() -> None:
    """Entry point utama Streamlit app."""

    # Load config & metrics
    missing = check_model_files()
    if missing:
        st.error("❌ File model tidak ditemukan:\n" + "\n".join(f"- {m}" for m in missing))
        st.info("Letakkan file model di folder `models/` lalu refresh halaman.")
        return

    config   = load_config_cached(MODEL_PATHS["config"])
    metrics  = load_metrics_cached(MODEL_PATHS["metrics"])
    groq_key = _get_groq_key()

    class_names = config["class_names"]
    short_names = config["short_names"]

    # Header
    _render_header(metrics, config)

    # Load model
    with st.spinner("Loading model..."):
        model = load_model_cached(MODEL_PATHS["model"])

    # Upload section 
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Video</div>',
                unsafe_allow_html=True)

    col_up, col_tips = st.columns([2, 1])
    with col_up:
        video_file = st.file_uploader(
            " ",
            type=["mp4", "avi", "mov", "mkv"]
        )
    with col_tips:
        st.markdown("""
        **Requirement agar dapat hasil analisa terbaik:**
        - Kamera dari atas atau bird-eye view
        - Resolusi minimal 480p
        - Tangan operator terlihat jelas dan kontras
        - Durasi maksimal 60 detik
        """)
    st.markdown('</div>', unsafe_allow_html=True)

    if not video_file:
        st.markdown("""
        <div style="text-align:center;padding:50px 20px;color:#90A4AE;">
          <div style="font-size:3.5rem;">🎬</div>
          <div style="font-size:1.05rem;font-weight:600;margin-top:10px;color:#546E7A;">
            Upload video untuk memulai analisis
          </div>
          <div style="font-size:0.88rem;margin-top:6px;">
            Sistem akan otomatis mendeteksi task dan menghitung cycle time
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Simpan ke tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    from inference import get_video_duration
    duration, fps, total_frames = get_video_duration(tmp_path)

    if duration > 65:
        st.warning(f"⚠️ Video terlalu panjang ({duration:.0f}s). Maks 60 detik.")
        os.unlink(tmp_path)
        return

    # Info & preview video
    st.markdown(f"""
    <div style="background:#E3F2FD;border-radius:10px;padding:10px 16px;
                margin-bottom:12px;display:flex;gap:24px;flex-wrap:wrap;
                font-size:0.88rem;">
      <span>📁 <b>{video_file.name}</b></span>
      <span>| Durasi: <b>{duration:.1f}s</b></span>
      <span>| FPS: <b>{fps:.0f}</b></span>
      <span>| Frames: <b>{total_frames:,}</b></span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Preview video", expanded=False):
        col_l, col_m, col_r = st.columns([1, 1, 1])
        with col_m:
            st.video(tmp_path)

    cache_key = f"results_{video_file.name}_{video_file.size}"

    if cache_key not in st.session_state:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analisis berjalan...</div>',
                    unsafe_allow_html=True)

        prog_bar  = st.progress(0)
        prog_text = st.empty()
        t_start   = time.time()

        def on_progress(val: float) -> None:
            prog_bar.progress(val)
            elapsed = time.time() - t_start
            eta     = elapsed / val * (1 - val) if val > 0 else 0
            prog_text.caption(f"Progress: {val:.0%} | ~{eta:.0f}s tersisa")

        from inference import predict_video_sliding, analyze_cycle_time

        predictions, total_dur = predict_video_sliding(
            tmp_path, model, config,
            window_sec=WINDOW_SEC,
            stride_sec=STRIDE_SEC,
            progress_cb=on_progress
        )

        task_segs, cycle_summary, timeline = analyze_cycle_time(
            predictions, class_names, short_names,
            min_task_duration=MIN_DUR_SEC
        )

        st.session_state[cache_key] = {
            "predictions"  : predictions,
            "task_segs"    : task_segs,
            "cycle_summary": cycle_summary,
            "timeline"     : timeline,
            "total_dur"    : total_dur,
            "tmp_path"     : tmp_path
        }

        elapsed = time.time() - t_start
        prog_bar.progress(1.0)
        prog_text.caption(f"Total waktu analisa: {elapsed:.1f}s")
        st.markdown('</div>', unsafe_allow_html=True)

    # Ambil hasil
    res           = st.session_state[cache_key]
    predictions   = res["predictions"]
    task_segs     = res["task_segs"]
    cycle_summary = res["cycle_summary"]
    timeline      = res["timeline"]
    total_dur     = res["total_dur"]
    task_stats    = cycle_summary.get("task_stats", {})

    st.markdown('<div class="blue-divider"></div>', unsafe_allow_html=True)
    _render_metric_cards(cycle_summary)

    # Tab
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cycle Time Analysis",
        "🎬 Per-Task Breakdown",
        "🔍 Model Evaluation",
        "🤖 AI Summary"
    ])

    with tab1:
        render_tab_cycle_time(
            timeline, task_stats, cycle_summary, short_names, total_dur
        )
    with tab2:
        render_tab_per_task(task_segs, tmp_path)
    with tab3:
        render_tab_evaluation(metrics, task_stats, predictions, short_names)
    with tab4:
        render_tab_ai_summary(
            cycle_summary, task_stats,
            class_names, short_names, metrics, groq_key
        )


if __name__ == "__main__":
    main()
