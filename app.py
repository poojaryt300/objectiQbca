"""
ObjectIQ — Smart Object Detector
See it. Scan it. Identify it.

A modern AI-powered object detection application using YOLOv8 + Streamlit.
Author: ObjectIQ Team
"""

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────
# Page Configuration (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ObjectIQ — Smart Object Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "ObjectIQ — Smart Object Detector v1.0\nPowered by YOLOv8 + Streamlit",
    },
)

# ─────────────────────────────────────────────
# Custom CSS — Dark Navy + Purple/Blue Theme
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-primary:    #050d1a;
    --bg-secondary:  #0a1628;
    --bg-card:       #0e1f38;
    --accent-purple: #7c3aed;
    --accent-blue:   #2563eb;
    --accent-cyan:   #06b6d4;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #64748b;
    --border:        rgba(124,58,237,0.25);
    --border-hover:  rgba(124,58,237,0.6);
    --shadow:        0 8px 32px rgba(0,0,0,0.5);
    --radius:        16px;
    --radius-sm:     10px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--accent-purple); border-radius: 3px; }

.hero-banner {
    background: linear-gradient(135deg, #0a0f2e 0%, #130d38 40%, #0d1f3c 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0;
    letter-spacing: -1px;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-top: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 500;
}
.hero-badge {
    display: inline-block;
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.4);
    color: #a78bfa;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 1rem;
    letter-spacing: 0.5px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-purple), var(--accent-blue));
    border-radius: var(--radius) var(--radius) 0 0;
}
.metric-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(124,58,237,0.2);
}
.metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.4rem;
    font-weight: 600;
}

.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow);
    transition: border-color 0.3s ease;
}
.section-card:hover { border-color: var(--border-hover); }
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.55) !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #0d9488) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(5,150,105,0.35) !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: var(--radius-sm) !important;
    gap: 4px;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.3s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue)) !important;
    color: white !important;
}

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: var(--text-muted);
}
.empty-state-icon { font-size: 4rem; margin-bottom: 1rem; }
.empty-state-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}
.empty-state-text { font-size: 0.9rem; line-height: 1.6; }

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(5,150,105,0.15);
    border: 1px solid rgba(5,150,105,0.4);
    color: #34d399;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}
.status-dot {
    width: 7px; height: 7px;
    background: #34d399;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.det-tag {
    display: inline-block;
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.35);
    color: #c4b5fd;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 3px;
    font-weight: 500;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    """Load YOLOv8n model once and cache it.
    Uses absolute path relative to this script so the model is always found
    regardless of which directory the user launches streamlit from.
    """
    try:
        from ultralytics import YOLO
        # Always look for yolov8n.pt next to app.py itself
        app_dir = Path(__file__).parent
        model_path = app_dir / "yolov8n.pt"
        model = YOLO(str(model_path))  # uses local file; downloads only if truly missing
        return model, None
    except ImportError:
        return None, "ultralytics package not installed. Run: pip install ultralytics"
    except Exception as e:
        return None, f"Model loading failed: {str(e)}"


# ─────────────────────────────────────────────
# Detection Engine
# ─────────────────────────────────────────────
def run_detection(image_pil: Image.Image, model, confidence: float) -> dict:
    """
    Run YOLOv8 detection on a PIL image.
    Returns a dict with: annotated_image, detections_df, error
    """
    try:
        img_np = np.array(image_pil.convert("RGB"))
        results = model(img_np, conf=confidence, verbose=False)

        detections = []
        annotated = img_np.copy()

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_score = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                detections.append({
                    "Object": label.title(),
                    "Confidence (%)": round(conf_score * 100, 1),
                    "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                })

                color = get_label_color(cls_id)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                text = f"{label.title()}  {conf_score*100:.0f}%"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                cv2.putText(annotated, text, (x1 + 5, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        df = pd.DataFrame(detections) if detections else pd.DataFrame(
            columns=["Object", "Confidence (%)", "X1", "Y1", "X2", "Y2"]
        )
        annotated_pil = Image.fromarray(annotated)
        return {"annotated_image": annotated_pil, "detections_df": df, "error": None}

    except Exception as e:
        return {"annotated_image": image_pil, "detections_df": pd.DataFrame(), "error": str(e)}


def get_label_color(cls_id: int) -> tuple:
    """Return a distinct BGR color for each class ID."""
    PALETTE = [
        (124, 58, 237), (37, 99, 235), (6, 182, 212), (236, 72, 153),
        (234, 179, 8),  (16, 185, 129), (239, 68, 68), (249, 115, 22),
        (99, 102, 241), (20, 184, 166),
    ]
    rgb = PALETTE[cls_id % len(PALETTE)]
    return (rgb[2], rgb[1], rgb[0])  # BGR for OpenCV


# ─────────────────────────────────────────────
# UI Helper Components
# ─────────────────────────────────────────────
def render_hero():
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🔍 ObjectIQ</div>
        <div class="hero-subtitle">See it. Scan it. Identify it.</div>
        <div class="hero-badge">⚡ Powered by YOLOv8 · Real-time AI Detection</div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_cards(df: pd.DataFrame):
    total    = len(df)
    unique   = df["Object"].nunique() if total > 0 else 0
    avg_conf = df["Confidence (%)"].mean() if total > 0 else 0.0
    top_obj  = df["Object"].mode()[0] if total > 0 else "—"

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-icon">📦</div>
            <div class="metric-value">{total}</div>
            <div class="metric-label">Objects Detected</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🏷️</div>
            <div class="metric-value">{unique}</div>
            <div class="metric-label">Unique Categories</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-value">{avg_conf:.1f}%</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🏆</div>
            <div class="metric-value" style="font-size:1.4rem">{top_obj}</div>
            <div class="metric-label">Top Object</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_bar_chart(df: pd.DataFrame):
    counts = df["Object"].value_counts().reset_index()
    counts.columns = ["Object", "Count"]

    fig = px.bar(
        counts, x="Object", y="Count",
        color="Count",
        color_continuous_scale=["#7c3aed", "#2563eb", "#06b6d4"],
        template="plotly_dark",
        text="Count",
        title="Object Detection Counts",
    )
    fig.update_traces(textposition="outside",
                      marker_line_color="rgba(255,255,255,0.1)",
                      marker_line_width=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,31,56,0.5)",
        font=dict(family="Inter", color="#94a3b8"),
        title_font=dict(family="Space Grotesk", size=16, color="#a78bfa"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#e2e8f0")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#e2e8f0")),
        coloraxis_showscale=False,
        margin=dict(t=50, b=30, l=10, r=10),
    )
    st.plotly_chart(fig, width="stretch")


def render_detection_tags(df: pd.DataFrame):
    if df.empty:
        return
    tags_html = "".join(
        f'<span class="det-tag">🏷 {obj}</span>'
        for obj in df["Object"].unique()
    )
    st.markdown(f"<div style='margin-bottom:0.8rem'>{tags_html}</div>", unsafe_allow_html=True)


def render_empty_state():
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">🕵️</div>
        <div class="empty-state-title">No Objects Detected</div>
        <div class="empty-state-text">
            Try lowering the confidence threshold,<br>
            use a clearer image, or capture a different scene.
        </div>
    </div>
    """, unsafe_allow_html=True)


def df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render_results_dashboard(original: Image.Image, result: dict):
    df = result["detections_df"]

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Detection Results</div>', unsafe_allow_html=True)

    render_metric_cards(df)
    render_detection_tags(df)

    col_orig, col_det = st.columns(2, gap="medium")
    with col_orig:
        st.markdown('<div class="section-title">🖼 Original Image</div>', unsafe_allow_html=True)
        st.image(original, width="stretch")
    with col_det:
        st.markdown('<div class="section-title">🤖 AI Annotated</div>', unsafe_allow_html=True)
        st.image(result["annotated_image"], width="stretch")

    if not df.empty:
        col_chart, col_table = st.columns([1.4, 1], gap="medium")
        with col_chart:
            render_bar_chart(df)
        with col_table:
            st.markdown('<div class="section-title">📋 Detection Table</div>', unsafe_allow_html=True)
            display_df = df[["Object", "Confidence (%)"]].copy()
            display_df = display_df.sort_values("Confidence (%)", ascending=False).reset_index(drop=True)
            display_df.index += 1
            st.dataframe(
                display_df.style.background_gradient(subset=["Confidence (%)"], cmap="Purples"),
                use_container_width=True,
                height=380,
            )

        st.markdown('<div class="section-title">💾 Export Report</div>', unsafe_allow_html=True)
        col_dl, col_scan = st.columns([1, 3])
        with col_dl:
            csv_bytes = df_to_csv(df[["Object", "Confidence (%)"]])
            st.download_button(
                label="📥  Download CSV",
                data=csv_bytes,
                file_name="objectiq_detections.csv",
                mime="text/csv",
            )
        with col_scan:
            if st.button("🔄  Scan Another Image"):
                for key in ["detection_result", "source_image", "detection_done"]:
                    st.session_state.pop(key, None)
                st.rerun()
    else:
        render_empty_state()
        if st.button("🔄  Try Again"):
            for key in ["detection_result", "source_image", "detection_done"]:
                st.session_state.pop(key, None)
            st.rerun()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
def render_sidebar(model_loaded: bool):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 1.5rem;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:800;
                        background:linear-gradient(90deg,#a78bfa,#60a5fa);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text;">
                🔍 ObjectIQ
            </div>
            <div style="color:#64748b; font-size:0.75rem; margin-top:4px; letter-spacing:1.5px; text-transform:uppercase;">
                Smart Object Detector
            </div>
        </div>
        """, unsafe_allow_html=True)

        if model_loaded:
            st.markdown("""
            <div class="status-pill">
                <div class="status-dot"></div>
                YOLOv8n Model Ready
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Model not loaded")

        st.markdown("---")
        st.markdown("**🎚️ Detection Settings**")
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.40,
            step=0.05,
            help="Lower = detects more objects (may include false positives).",
            format="%.0f%%",
        )
        st.caption(f"Sensitivity: **{int(confidence*100)}%**")

        st.markdown("---")
        st.markdown("**ℹ️ About ObjectIQ**")
        st.markdown("""
        <div style="color:#64748b; font-size:0.82rem; line-height:1.7;">
            • Model: <span style="color:#a78bfa">YOLOv8n</span><br>
            • Dataset: <span style="color:#60a5fa">COCO (80 classes)</span><br>
            • No API key required<br>
            • Runs 100% locally<br>
            <br>
            <span style="color:#475569;">Detectable objects include:</span><br>
            person, laptop, phone, bottle, cup, chair, book, mouse, keyboard, car, dog, cat...
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="color:#334155; font-size:0.75rem; text-align:center;">
            ObjectIQ v1.0 · College Mini Project<br>
            Powered by Ultralytics YOLOv8
        </div>
        """, unsafe_allow_html=True)

    return confidence


# ─────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────
def main():
    with st.spinner("🔄 Loading YOLOv8 model (downloads ~6 MB on first run)..."):
        model, model_error = load_yolo_model()

    model_loaded = model is not None
    confidence = render_sidebar(model_loaded)
    render_hero()

    if model_error:
        st.error(f"❌ **Model Error:** {model_error}")
        st.info("Run `pip install ultralytics` in your terminal then restart the app.")
        return

    # If detection already done, show results
    if st.session_state.get("detection_done") and "detection_result" in st.session_state:
        render_results_dashboard(
            st.session_state["source_image"],
            st.session_state["detection_result"],
        )
        return

    # Input Tabs
    tab_camera, tab_upload = st.tabs(["📷  Camera Scan", "🖼️  Upload Image"])

    # ── Camera Tab ──
    with tab_camera:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📷 Camera Scan</div>
            <p style="color:#94a3b8; font-size:0.9rem; margin:0;">
                Point your camera at any object and click <strong>Take Photo</strong>.
                ObjectIQ will instantly identify everything in the frame.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.info("💡 **Tip:** Allow camera access when your browser asks. If blocked, click the camera icon in the address bar → Allow → Refresh.", icon="ℹ️")

        camera_image = st.camera_input(
            label="Take a photo to detect objects",
            label_visibility="collapsed",
        )

        if camera_image is not None:
            with st.spinner("🤖 Analysing image with YOLOv8..."):
                time.sleep(0.3)
                pil_img = Image.open(camera_image).convert("RGB")
                result = run_detection(pil_img, model, confidence)

            if result["error"]:
                st.error(f"❌ Detection error: {result['error']}")
            else:
                st.session_state["source_image"] = pil_img
                st.session_state["detection_result"] = result
                st.session_state["detection_done"] = True
                st.rerun()

    # ── Upload Tab ──
    with tab_upload:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">🖼️ Upload Image</div>
            <p style="color:#94a3b8; font-size:0.9rem; margin:0;">
                Upload a JPG, JPEG, PNG, or WEBP image.
                ObjectIQ will detect all objects present in the scene.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            try:
                pil_img = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error(f"❌ Could not open image: {e}")
                pil_img = None

            if pil_img is not None:
                col_prev, col_btn = st.columns([2, 1])
                with col_prev:
                    st.image(pil_img, caption="Uploaded Image", width="stretch")
                with col_btn:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    img_w, img_h = pil_img.size
                    st.markdown(f"""
                    <div style="color:#94a3b8; font-size:0.85rem; margin-bottom:1rem;">
                        📐 <strong>{img_w} × {img_h} px</strong><br>
                        📄 {uploaded_file.name}<br>
                        💾 {uploaded_file.size // 1024} KB
                    </div>
                    """, unsafe_allow_html=True)
                    detect_btn = st.button(
                        "🚀  Detect Objects",
                        use_container_width=True,
                        key="detect_upload_btn",
                    )

                if detect_btn:
                    with st.spinner("🤖 Running YOLOv8 detection..."):
                        time.sleep(0.2)
                        result = run_detection(pil_img, model, confidence)

                    if result["error"]:
                        st.error(f"❌ Detection error: {result['error']}")
                    else:
                        st.session_state["source_image"] = pil_img
                        st.session_state["detection_result"] = result
                        st.session_state["detection_done"] = True
                        st.rerun()

    # Welcome placeholder
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3, gap="large")
    with col_a:
        st.markdown("""
        <div class="section-card" style="text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">📷</div>
            <div class="section-title" style="justify-content:center;">Step 1</div>
            <p style="color:#64748b;font-size:0.88rem;">Capture a photo with your webcam or upload an image file.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="section-card" style="text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">🤖</div>
            <div class="section-title" style="justify-content:center;">Step 2</div>
            <p style="color:#64748b;font-size:0.88rem;">YOLOv8 AI analyses the image and identifies all objects.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="section-card" style="text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">📊</div>
            <div class="section-title" style="justify-content:center;">Step 3</div>
            <p style="color:#64748b;font-size:0.88rem;">View annotated results, confidence scores, and download your report.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
