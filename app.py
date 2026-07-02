import streamlit as st
import requests
from PIL import Image
import io
import os
import zipfile
import textwrap
import plotly.graph_objects as go

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="✍️",
    layout="wide",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  –  Light / Glassmorphism Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #fdf2f8 50%, #f0fdfa 100%);
}

/* ── Hero Banner ── */
.hero {
    position: relative;
    background: linear-gradient(120deg, #0f172a 0%, #1e1b4b 55%, #312e81 100%);
    border-radius: 20px;
    padding: 2.4rem 2.6rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 20px 45px rgba(30,27,75,0.28);
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    flex-wrap: wrap;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60%;
    right: -10%;
    width: 420px;
    height: 420px;
    background: radial-gradient(circle, rgba(99,102,241,0.35) 0%, rgba(99,102,241,0) 70%);
    pointer-events: none;
}
.hero::after {
    content: "";
    position: absolute;
    bottom: -50%;
    left: 10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(236,72,153,0.22) 0%, rgba(236,72,153,0) 70%);
    pointer-events: none;
}
.hero-left {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 1.1rem;
}
.hero-icon {
    flex-shrink: 0;
    width: 58px;
    height: 58px;
    border-radius: 16px;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.7rem;
    box-shadow: 0 8px 20px rgba(99,102,241,0.4);
}
.hero h1.hero-title, .hero-title {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    margin: 0 !important;
    letter-spacing: -0.02em;
}
.hero p.hero-subtitle, .hero-subtitle {
    color: #94a3b8 !important;
    font-size: 0.94rem !important;
    margin: 0.25rem 0 0 0 !important;
}
.hero-tag {
    display: inline-block;
    background: rgba(99,102,241,0.18);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 999px;
    padding: 0.15rem 0.7rem;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Hero stat grid ── */
.hero-stats {
    position: relative;
    z-index: 1;
    display: grid;
    grid-template-columns: repeat(2, minmax(120px, 1fr));
    gap: 0.6rem;
}
.hero-stat {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 0.6rem 1rem;
    text-align: left;
}
.hero-stat .stat-value {
    color: #f8fafc;
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.2;
}
.hero-stat .stat-label {
    color: #94a3b8;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

/* ── Glass Card ── */
.glass-card {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.6);
    border-radius: 20px;
    padding: 1.6rem;
    box-shadow: 0 8px 24px rgba(99,102,241,0.08);
    margin-bottom: 1rem;
}
.glass-card h3 {
    margin-top: 0;
    color: #1e293b;
}

/* ── Result Card ── */
.result-card {
    background: linear-gradient(135deg, #ffffff, #f5f3ff);
    border: 1px solid #e0e7ff;
    border-radius: 22px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1rem;
    box-shadow: 0 12px 32px rgba(99,102,241,0.15);
}
.result-digit {
    font-size: 6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.result-label {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
}
.confidence-bar-wrap {
    background: #eef2ff;
    border-radius: 999px;
    height: 12px;
    margin: 1.1rem 0 0.4rem;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── File info pills ── */
.info-pill {
    display: flex;
    justify-content: space-between;
    background: white;
    border-radius: 10px;
    padding: 0.5rem 0.9rem;
    margin-bottom: 0.4rem;
    font-size: 0.82rem;
    color: #475569;
    border: 1px solid #f1f5f9;
}
.info-pill b { color: #1e293b; }

/* ── Sample grid ── */
.sample-label {
    text-align: center;
    color: #64748b;
    font-size: 0.75rem;
    margin-top: 0.3rem;
}

/* ── Section titles ── */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.6rem;
}

/* ── Force caption/help text visible on light bg ── */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
.stCaption, .stCaption p, small, small p {
    color: #000000 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(255,255,255,0.55);
    padding: 0.4rem;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.6);
    box-shadow: 0 4px 16px rgba(99,102,241,0.08);
}
.stTabs [data-baseweb="tab"] {
    height: 46px;
    border-radius: 12px;
    padding: 0 1.4rem;
    background: transparent;
    color: #64748b;
    font-weight: 600;
    font-size: 0.92rem;
    border: none;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(99,102,241,0.08);
    color: #4f46e5;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #ec4899) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.35);
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

API_URL = "http://127.0.0.1:8000/predict/"

SAMPLE_DIR = "sample_digits"

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def call_api(image_bytes: bytes, filename: str) -> dict | None:
    try:
        resp = requests.post(
            API_URL,
            files={"file": (filename, image_bytes, "image/png")},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def make_zip_of_samples() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(10):
            path = os.path.join(SAMPLE_DIR, f"digit_{i}.png")
            if os.path.exists(path):
                zf.write(path, f"digit_{i}.png")
    return buf.getvalue()


def confidence_html(conf: float) -> str:
    pct = round(conf * 100, 1)
    color = "#059669" if conf > 0.90 else "#d97706" if conf > 0.70 else "#dc2626"
    return (
        f'<div class="confidence-bar-wrap">'
        f'<div class="confidence-bar-fill" style="width:{pct}%; background:{color};"></div>'
        f'</div>'
        f'<p style="color:{color}; font-size:0.95rem; font-weight:700; margin:0;">{pct}% confidence</p>'
    )

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <div class="hero-icon">✍️</div>
        <div>
            <span class="hero-tag">Deep Learning · Computer Vision</span>
            <h1 class="hero-title">MNIST Digit Classifier</h1>
            <p class="hero-subtitle">Upload a handwritten digit and get an instant AI-powered prediction</p>
        </div>
    </div>
    <div class="hero-stats">
        <div class="hero-stat"><div class="stat-value">97.65%</div><div class="stat-label">Accuracy</div></div>
        <div class="hero-stat"><div class="stat-value">Dense NN</div><div class="stat-label">Architecture</div></div>
        <div class="hero-stat"><div class="stat-value">FastAPI</div><div class="stat-label">Backend</div></div>
        <div class="hero-stat"><div class="stat-value">TensorFlow</div><div class="stat-label">Framework</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN — TABS
# ─────────────────────────────────────────────
tab_predict, tab_samples = st.tabs(["🎯 Predict", "🖼️ Sample Images"])

# ── TAB 1: Upload & Predict ─────────────────
with tab_predict:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📤 Upload Your Image</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Choose a PNG / JPG of a handwritten digit",
            type=["png", "jpg", "jpeg"],
            help="Best results with white digit on black background, similar to MNIST format",
            label_visibility="collapsed",
        )

        if uploaded:
            img_bytes = uploaded.read()
            img = Image.open(io.BytesIO(img_bytes))

            st.image(img, width=180, caption=uploaded.name)

            st.markdown(textwrap.dedent(f"""
            <div class="info-pill"><span>📄 File</span><b>{uploaded.name}</b></div>
            <div class="info-pill"><span>📐 Size</span><b>{img.size[0]} × {img.size[1]} px</b></div>
            <div class="info-pill"><span>🎨 Mode</span><b>{img.mode}</b></div>
            <div class="info-pill"><span>💾 Bytes</span><b>{len(img_bytes):,}</b></div>
            """), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🚀 Predict Digit", use_container_width=True, type="primary")
        else:
            st.info("👆 Drop an image above to get started")
            predict_btn = False

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔮 Prediction Result</div>', unsafe_allow_html=True)

        if uploaded and predict_btn:
            with st.spinner("🧠 Thinking..."):
                result = call_api(img_bytes, uploaded.name)

            if "error" in result:
                st.error(f"❌ API Error: {result['error']}\n\nMake sure `uvicorn main:app --reload` is running.")
            else:
                digit = result["predicted_digit"]
                conf  = result["confidence"]
                fname = result["filename"]

                emoji_map = {
                    0:"0️⃣",1:"1️⃣",2:"2️⃣",3:"3️⃣",4:"4️⃣",
                    5:"5️⃣",6:"6️⃣",7:"7️⃣",8:"8️⃣",9:"9️⃣"
                }

                st.markdown(textwrap.dedent(f"""
                <div class="result-card">
                    <div style="font-size:2rem;">{emoji_map.get(digit,'🔢')}</div>
                    <div class="result-digit">{digit}</div>
                    <div class="result-label">Predicted Digit</div>
                    {confidence_html(conf)}
                    <p style="color:#94a3b8; font-size:0.75rem; margin-top:0.8rem;">File: {fname}</p>
                </div>
                """), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-title" style="font-size:0.92rem;">📊 Confidence Breakdown</div>', unsafe_allow_html=True)

                conf_pct   = round(conf * 100, 1)
                uncertain  = round(100 - conf_pct, 1)
                gauge_color = "#059669" if conf > 0.90 else "#d97706" if conf > 0.70 else "#dc2626"

                gauge_col, donut_col = st.columns(2)

                with gauge_col:
                    gauge_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=conf_pct,
                        number={"suffix": "%", "font": {"size": 34, "color": "#1e293b"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e1"},
                            "bar": {"color": gauge_color, "thickness": 0.3},
                            "bgcolor": "white",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 70],  "color": "#fef2f2"},
                                {"range": [70, 90], "color": "#fffbeb"},
                                {"range": [90, 100],"color": "#ecfdf5"},
                            ],
                        },
                    ))
                    gauge_fig.update_layout(
                        height=220,
                        margin=dict(l=20, r=20, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"family": "Poppins", "color": "#000000"},
                    )
                    st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False}, theme=None)
                    st.caption("Model certainty for the predicted digit")

                with donut_col:
                    donut_fig = go.Figure(go.Pie(
                        labels=["Confidence", "Uncertainty"],
                        values=[conf_pct, uncertain],
                        hole=0.65,
                        marker={"colors": [gauge_color, "#e2e8f0"]},
                        textinfo="none",
                        sort=False,
                    ))
                    donut_fig.update_layout(
                        height=220,
                        margin=dict(l=10, r=10, t=10, b=10),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.15, font={"color": "#000000"}),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={"family": "Poppins", "size": 12, "color": "#000000"},
                        annotations=[dict(
                            text=f"<b>{digit}</b>", x=0.5, y=0.5,
                            font_size=36, font_color=gauge_color, showarrow=False
                        )],
                    )
                    st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False}, theme=None)
                    st.caption("Predicted class vs. remaining uncertainty")

                st.caption(
                    f"ℹ️ The model returns confidence for the top predicted class only — "
                    f"per-class probabilities aren't exposed by the API, so no distribution across all 10 digits is shown."
                )
        elif uploaded and not predict_btn:
            st.info("✅ Image loaded — hit **Predict Digit** to run inference")
        else:
            st.markdown(textwrap.dedent("""
            <div style="text-align:center; padding:2.5rem 1rem; color:#94a3b8;">
                <div style="font-size:3rem;">🔢</div>
                <p>Results will appear here once you upload & predict</p>
            </div>
            """), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: Sample Downloads ─────────────────
with tab_samples:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🖼️ Sample Test Images</div>', unsafe_allow_html=True)
    st.caption("Download sample MNIST-style images to test the classifier.")

    if os.path.exists(SAMPLE_DIR):
        zip_bytes = make_zip_of_samples()
        st.download_button(
            label="📦 Download All Samples (ZIP)",
            data=zip_bytes,
            file_name="mnist_sample_digits.zip",
            mime="application/zip",
            use_container_width=True,
        )

        st.markdown("<br>**Or download individually:**", unsafe_allow_html=True)

        rows = [range(0, 5), range(5, 10)]
        for row in rows:
            cols = st.columns(5)
            for i, digit in enumerate(row):
                path = os.path.join(SAMPLE_DIR, f"digit_{digit}.png")
                if os.path.exists(path):
                    with cols[i]:
                        img_sample = Image.open(path)
                        display_img = img_sample.resize((90, 90), Image.NEAREST)
                        st.image(display_img, caption=f"Digit {digit}", use_container_width=False)
                        with open(path, "rb") as f:
                            st.download_button(
                                label=f"↓ {digit}",
                                data=f.read(),
                                file_name=f"digit_{digit}.png",
                                mime="image/png",
                                key=f"dl_{digit}",
                                use_container_width=True,
                            )
    else:
        st.warning("Sample images not found. Make sure `sample_digits/` folder is in the same directory as `app.py`.")

    st.markdown('</div>', unsafe_allow_html=True)


