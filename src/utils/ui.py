import base64
from pathlib import Path
import streamlit as st


@st.cache_resource(show_spinner=False)
def get_base64_file(file_path: str) -> str:
    """Read a local file and return its base64 string representation."""
    p = Path(file_path)
    if p.exists():
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""


def apply_video_background(
    video_path: str = "assets/bg.mp4",
    poster_path: str = "assets/bg_poster.jpg",
) -> None:
    """Embed a looping video background with dark overlay and accessible glassmorphism CSS."""
    video_p = Path(video_path)
    poster_p = Path(poster_path)

    has_video = video_p.exists() and video_p.stat().st_size > 0
    video_b64 = get_base64_file(str(video_p)) if has_video else ""
    poster_b64 = get_base64_file(str(poster_p)) if poster_p.exists() else ""

    if has_video and video_b64:
        poster_attr = f'poster="data:image/jpeg;base64,{poster_b64}"' if poster_b64 else ""
        bg_html = f"""
        <div class="video-bg-container">
            <video autoplay muted loop playsinline {poster_attr} class="bg-video">
                <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
            </video>
            <div class="video-overlay"></div>
        </div>
        """
    else:
        # Fallback to static dark gradient with no crash
        bg_html = '<div class="static-bg-overlay"></div>'

    style_html = f"""
    {bg_html}
    <style>
    /* Full-screen video container */
    .video-bg-container {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        z-index: -1;
        pointer-events: none;
    }}
    .bg-video {{
        width: 100vw;
        height: 100vh;
        object-fit: cover;
        position: absolute;
        top: 0;
        left: 0;
        z-index: -2;
    }}
    /* Dark gradient overlay on top (rgba(0,0,0,0.55)) for crisp text readability */
    .video-overlay {{
        position: absolute;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: linear-gradient(
            135deg,
            rgba(8, 6, 22, 0.68) 0%,
            rgba(18, 16, 42, 0.55) 50%,
            rgba(10, 8, 28, 0.68) 100%
        );
        z-index: -1;
    }}
    /* Static fallback gradient */
    .static-bg-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        z-index: -1;
        pointer-events: none;
    }}

    /* Streamlit chrome transparency */
    .stApp {{
        background: transparent !important;
    }}
    header[data-testid="stHeader"], footer, #MainMenu {{
        background: transparent !important;
    }}
    [data-testid="stDecoration"] {{
        display: none !important;
    }}

    /* Typography & Contrast */
    h1, h2, h3, h4, p, label, span, div {{
        color: #ffffff !important;
    }}

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {{
        background: rgba(18, 16, 38, 0.65) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.12) !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #f1f0ff !important;
    }}

    /* Glassmorphism Cards */
    .glass {{
        background: rgba(22, 20, 48, 0.60) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 18px !important;
        padding: 1.4rem 1.8rem !important;
        margin-bottom: 0.8rem !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45) !important;
        color: #ffffff !important;
    }}
    .glass-sm {{
        background: rgba(22, 20, 48, 0.60) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 14px !important;
        padding: 1rem 1.4rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
        color: #ffffff !important;
        height: 100%;
    }}

    /* Text Input Glassmorphism */
    [data-testid="stTextInput"] > div > div > input {{
        background: rgba(18, 16, 38, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        padding: 0.65rem 1rem !important;
    }}
    [data-testid="stTextInput"] > div > div > input::placeholder {{
        color: rgba(255, 255, 255, 0.55) !important;
    }}
    [data-testid="stTextInput"] > div > div > input:focus {{
        border-color: rgba(147, 112, 219, 0.85) !important;
        box-shadow: 0 0 0 2px rgba(147, 112, 219, 0.35) !important;
    }}

    /* Buttons with accessible contrast */
    .stButton > button {{
        background: rgba(147, 112, 219, 0.25) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover {{
        background: rgba(147, 112, 219, 0.55) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
        box-shadow: 0 0 12px rgba(147, 112, 219, 0.4) !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background: rgba(20, 18, 42, 0.55) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 14px !important;
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
    }}
    [data-testid="stExpander"] summary {{
        color: #ffffff !important;
        font-weight: 500 !important;
    }}

    /* Chunk cards */
    .chunk-card {{
        background: rgba(15, 13, 30, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.15rem !important;
        margin-bottom: 0.7rem !important;
    }}
    .chunk-meta {{
        color: rgba(190, 180, 255, 0.95) !important;
        font-size: 0.82rem !important;
        margin-bottom: 0.4rem !important;
    }}

    /* Source chips */
    .chip {{
        display: inline-block;
        background: rgba(130, 100, 255, 0.35);
        border: 1px solid rgba(130, 100, 255, 0.55);
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.78rem;
        margin: 3px 4px 3px 0;
        color: #e5dcff !important;
    }}

    /* Error banner */
    .err-box {{
        background: rgba(255, 80, 80, 0.2) !important;
        border: 1px solid rgba(255, 80, 80, 0.5) !important;
        border-radius: 12px !important;
        padding: 1rem 1.4rem !important;
        color: #ffcccc !important;
        margin-bottom: 1rem !important;
    }}

    /* File Uploader styling */
    [data-testid="stFileUploader"] section {{
        background: rgba(18, 16, 38, 0.65) !important;
        border: 1px dashed rgba(255, 255, 255, 0.3) !important;
        border-radius: 14px !important;
        padding: 0.8rem 1rem !important;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: rgba(147, 112, 219, 0.8) !important;
    }}
    [data-testid="stFileUploader"] * {{
        color: #ffffff !important;
    }}

    /* Scope Selector (Radio) */
    [data-testid="stRadio"] {{
        background: rgba(22, 20, 48, 0.55) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        border-radius: 14px !important;
        padding: 0.6rem 1.2rem !important;
        margin-bottom: 0.8rem !important;
    }}
    [data-testid="stRadio"] label {{
        color: #ffffff !important;
        font-weight: 500 !important;
        cursor: pointer !important;
    }}
    [data-testid="stRadio"] div[role="radiogroup"] {{
        gap: 1.5rem !important;
    }}

    /* Progress bar */
    [data-testid="stProgress"] > div > div > div > div {{
        background: linear-gradient(90deg, #9370db, #3498db) !important;
    }}
    [data-testid="stProgress"] > div > div {{
        background: rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
    }}

    /* Uploaded Document List Cards in Sidebar */
    .doc-card {{
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 10px !important;
        padding: 0.5rem 0.75rem !important;
        margin-bottom: 0.4rem !important;
    }}

    /* Source Tags */
    .source-tag {{
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 1px 7px;
        border-radius: 8px;
        margin-right: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .source-tag-course {{
        background: rgba(147, 112, 219, 0.25);
        border: 1px solid rgba(147, 112, 219, 0.6);
        color: #d8b4fe !important;
    }}
    .source-tag-upload {{
        background: rgba(52, 152, 219, 0.25);
        border: 1px solid rgba(52, 152, 219, 0.6);
        color: #7dd3fc !important;
    }}
    </style>
    """
    st.markdown(style_html, unsafe_allow_html=True)

