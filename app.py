import streamlit as st
import streamlit.components.v1 as components
from huggingface_hub import InferenceClient
from gtts import gTTS
import io
import base64
import re

# ============================================================
#  PAGE CONFIG — must be first Streamlit call
# ============================================================
st.set_page_config(
    page_title="CodeRefine Pro",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
#  GLOBAL CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800&family=Outfit:wght@300;400;600;700&display=swap');

:root {
  --bg:           #04070f;
  --surface:      #080e1c;
  --border:       rgba(0,255,200,0.15);
  --accent:       #00ffc8;
  --accent2:      #ff3cac;
  --accent3:      #ffd700;
  --text:         #c8d8e8;
  --muted:        #4a5a72;
  --font-mono:    'Space Mono', monospace;
  --font-display: 'Syne', sans-serif;
  --font-body:    'Outfit', sans-serif;
}

html, body, [class*="css"] {
  font-family: var(--font-mono);
  background: var(--bg);
  color: var(--text);
}

.stApp {
  background:
    repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,200,0.015) 2px, rgba(0,255,200,0.015) 4px),
    radial-gradient(ellipse 80% 60% at 50% -10%, rgba(0,255,200,0.08) 0%, transparent 70%),
    var(--bg);
  min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem; max-width: 1100px; margin: auto; }

/* ── HEADER ── */
.cr-header {
  display: flex; align-items: center; gap: 1rem;
  padding: 2.2rem 0 1.2rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.8rem;
}
.cr-logo {
  font-family: var(--font-display); font-size: 2rem; font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.cr-badge {
  font-size: 0.65rem; font-family: var(--font-mono); letter-spacing: 0.2em;
  text-transform: uppercase; padding: 0.2rem 0.6rem;
  border: 1px solid var(--accent); color: var(--accent); margin-top: 4px; display: inline-block;
}
.cr-subtitle { font-size: 0.75rem; color: var(--muted); letter-spacing: 0.05em; margin-left: auto; }
.cr-cursor {
  display: inline-block; width: 10px; height: 22px;
  background: var(--accent); margin-left: 6px; vertical-align: middle;
  animation: blink 1s step-end infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* ── MESSAGES ── */
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0 !important; margin: 0 !important; }

.user-message-container { display: flex; justify-content: flex-end; margin-bottom: 1rem; width: 100%; gap: 10px; }
.user-message-wrapper { display: flex; flex-direction: column; align-items: flex-end; }
.user-label { font-size: 0.65rem; color: var(--accent2); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
.user-message {
  background: linear-gradient(135deg, rgba(255,60,172,0.15), rgba(255,60,172,0.05));
  border: 1px solid rgba(255,60,172,0.3); padding: 1rem 1.25rem;
  max-width: 80%; color: var(--text); font-family: var(--font-mono);
  font-size: 0.88rem; line-height: 1.6; word-break: break-word;
}
.user-avatar {
  width: 32px; height: 32px;
  background: linear-gradient(135deg, var(--accent2), #c0009a);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.7rem; font-weight: 700; color: white; flex-shrink: 0;
}

.assistant-message-container { display: flex; justify-content: flex-start; margin-bottom: 1rem; width: 100%; gap: 10px; }
.assistant-message-wrapper { display: flex; flex-direction: column; align-items: flex-start; }
.assistant-label { font-size: 0.65rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
.assistant-message {
  background: linear-gradient(135deg, rgba(0,255,200,0.1), rgba(0,255,200,0.03));
  border: 1px solid var(--border); padding: 1rem 1.25rem;
  max-width: 80%; color: var(--text); font-family: var(--font-mono);
  font-size: 0.88rem; line-height: 1.6; word-break: break-word;
}
.assistant-avatar {
  width: 32px; height: 32px;
  background: linear-gradient(135deg, var(--accent), #00996e);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.7rem; font-weight: 700; color: #04070f; flex-shrink: 0;
}
.assistant-message code, .assistant-message pre {
  font-family: var(--font-mono) !important; background: rgba(0,0,0,0.4) !important;
  border: 1px solid rgba(0,255,200,0.1) !important; color: var(--accent3) !important; padding: 0.2rem 0.4rem;
}
.assistant-message pre { padding: 1rem !important; overflow-x: auto; }

.audio-wrapper { margin-left: 42px; margin-top: -0.5rem; margin-bottom: 1.5rem; width: calc(100% - 42px); }

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
  width: calc(100% - 40px); max-width: 1060px; z-index: 1000;
}
[data-testid="stChatInput"] textarea { font-family: var(--font-mono) !important; color: var(--text) !important; background: transparent !important; font-size: 0.88rem !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }
[data-testid="stChatInput"] button { background: linear-gradient(135deg, var(--accent), #00b89a) !important; color: #04070f !important; }

[data-testid="stStatus"] {
  background: rgba(0,255,200,0.04) !important; border: 1px solid var(--border) !important;
  font-family: var(--font-mono) !important; font-size: 0.82rem !important; color: var(--accent) !important;
}

audio { display: none; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent); }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }

.chat-container { margin-bottom: 100px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  HEADER
# ============================================================
st.markdown("""
<div class="cr-header">
  <div>
    <div class="cr-logo">CodeRefine<span style="color:#ff3cac">Pro</span><span class="cr-cursor"></span></div>
    <div class="cr-badge">⬡ Powered by Qwen2.5-Coder-32B</div>
  </div>
  <div class="cr-subtitle">AI Senior Architect · Audio Insight · Live Debug</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
#  SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "fix_count" not in st.session_state:
    st.session_state.fix_count = 0
if "char_count" not in st.session_state:
    st.session_state.char_count = 0


# ============================================================
#  INFERENCE CLIENT
# ============================================================
client = InferenceClient(api_key=st.secrets["HF_TOKEN"])


# ============================================================
#  HELPER FUNCTIONS
# ============================================================
def get_ai_fix(user_code: str) -> str:
    prompt = (
        "You are a Senior Software Architect. "
        "First, briefly explain the root cause of any bug or design flaw. "
        "Then provide the corrected, production-ready code with inline comments. "
        "Format your explanation with **bold** section headers.\n\n"
        f"Code to review:\n```\n{user_code}\n```"
    )
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=900,
    )
    return response.choices[0].message.content


def get_audio_summary(full_text: str) -> str:
    summary_prompt = (
        "Summarize the following code fix in exactly 3 short, clear, friendly sentences "
        "suitable for text-to-speech. Speak like a teacher explaining to a student. "
        "No markdown, no symbols, no code snippets:\n\n"
        f"{full_text}"
    )
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=150,
    )
    return response.choices[0].message.content


def get_error_highlights(full_text: str) -> list:
    """Extract key error points as short bullet-style strings for the teacher chalkboard."""
    highlight_prompt = (
        "From the following code review, extract exactly 3 key problems found. "
        "Each should be 5-8 words max. Return ONLY a JSON array of 3 strings, nothing else.\n\n"
        f"{full_text}"
    )
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        messages=[{"role": "user", "content": highlight_prompt}],
        max_tokens=100,
    )
    raw = response.choices[0].message.content.strip()
    # Parse JSON array
    match = re.search(r'\[.*?\]', raw, re.DOTALL)
    if match:
        import json
        try:
            items = json.loads(match.group())
            return items[:3]
        except Exception:
            pass
    # Fallback
    return ["Bug detected in logic", "Missing error handling", "Code needs refactoring"]


def generate_audio_b64(text: str) -> str:
    tts = gTTS(text=text, lang="en", tld="co.uk")
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return base64.b64encode(fp.read()).decode("utf-8")


# ============================================================
#  AI TEACHER PLAYER  (the centrepiece feature)
# ============================================================
def render_teacher_player(audio_b64: str, error_points: list, autoplay: bool = True, player_id: str = "player"):
    """
    Renders a full-screen-style AI Teacher character with:
    - Animated 3D-looking teacher (CSS art, clay-style)
    - Chalkboard with error bullet points that appear one by one
    - Lip-sync mouth animation while audio plays
    - Pointer/arm gesture
    - Cyberpunk audio controls
    - Smooth entrance animation
    """

    ep_js = str(error_points).replace("'", '"')

    # SVG teacher character embedded inline
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; overflow: hidden; font-family: 'Space Mono', monospace; }}

  /* ── MAIN SCENE ── */
  .scene {{
    width: 100%;
    background: linear-gradient(160deg, #060d1f 0%, #0a1428 50%, #060d1f 100%);
    border: 1px solid rgba(0,255,200,0.2);
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-height: 520px;
  }}

  /* scanlines */
  .scene::before {{
    content: '';
    position: absolute; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,255,200,0.012) 3px, rgba(0,255,200,0.012) 4px);
    pointer-events: none; z-index: 10;
  }}

  /* corner decorations */
  .corner {{ position: absolute; width: 20px; height: 20px; z-index: 20; }}
  .corner.tl {{ top: 8px; left: 8px; border-top: 1px solid #00ffc8; border-left: 1px solid #00ffc8; }}
  .corner.tr {{ top: 8px; right: 8px; border-top: 1px solid #00ffc8; border-right: 1px solid #00ffc8; }}
  .corner.bl {{ bottom: 8px; left: 8px; border-bottom: 1px solid #00ffc8; border-left: 1px solid #00ffc8; }}
  .corner.br {{ bottom: 8px; right: 8px; border-bottom: 1px solid #00ffc8; border-right: 1px solid #00ffc8; }}

  /* top label */
  .scene-label {{
    position: absolute; top: 14px; left: 50%; transform: translateX(-50%);
    font-size: 0.5rem; letter-spacing: 0.3em; color: rgba(0,255,200,0.45);
    text-transform: uppercase; z-index: 20;
  }}

  /* ── CONTENT ROW ── */
  .content-row {{
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 20px;
    padding: 40px 24px 16px;
    flex: 1;
  }}

  /* ── TEACHER CHARACTER ── */
  .teacher-wrap {{
    position: relative;
    flex-shrink: 0;
    animation: teacherFloat 3s ease-in-out infinite;
    transform-origin: bottom center;
  }}
  @keyframes teacherFloat {{
    0%, 100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-6px); }}
  }}

  /* Shadow under teacher */
  .teacher-shadow {{
    position: absolute;
    bottom: -8px; left: 50%; transform: translateX(-50%);
    width: 90px; height: 12px;
    background: radial-gradient(ellipse, rgba(0,255,200,0.25) 0%, transparent 70%);
    animation: shadowPulse 3s ease-in-out infinite;
  }}
  @keyframes shadowPulse {{
    0%, 100% {{ transform: translateX(-50%) scaleX(1); opacity: 0.6; }}
    50% {{ transform: translateX(-50%) scaleX(0.85); opacity: 0.3; }}
  }}

  /* CSS Teacher Figure */
  .teacher-svg {{
    width: 130px;
    filter: drop-shadow(0 0 18px rgba(0,255,200,0.3)) drop-shadow(0 0 40px rgba(0,255,200,0.1));
  }}

  /* ── ARM/POINTER ANIMATION ── */
  .arm-pointer {{
    transform-origin: 68px 85px; /* shoulder pivot */
    transition: transform 0.6s cubic-bezier(.34,1.56,.64,1);
  }}
  .arm-pointer.pointing {{
    transform: rotate(-25deg);
  }}
  .arm-pointer.resting {{
    transform: rotate(0deg);
  }}

  /* ── MOUTH ANIMATION ── */
  .mouth-open {{ animation: mouthTalk 0.18s ease-in-out infinite alternate; }}
  .mouth-closed {{ transform: scaleY(1); }}
  @keyframes mouthTalk {{
    from {{ transform: scaleY(0.3); }}
    to   {{ transform: scaleY(1.4); }}
  }}

  /* ── CHALKBOARD ── */
  .chalkboard-wrap {{
    flex: 1;
    max-width: 480px;
    position: relative;
  }}

  .chalkboard {{
    width: 100%;
    background: linear-gradient(135deg, #1a3520 0%, #1f3d24 40%, #183018 100%);
    border: 8px solid #5c3d1a;
    border-radius: 2px;
    box-shadow:
      inset 0 0 30px rgba(0,0,0,0.5),
      inset 0 0 60px rgba(0,0,0,0.3),
      0 8px 32px rgba(0,0,0,0.6),
      0 0 0 2px #7a5225,
      0 0 0 4px #4a2a0a;
    padding: 20px 22px 16px;
    position: relative;
    min-height: 200px;
  }}

  /* Chalk texture overlay */
  .chalkboard::before {{
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4'%3E%3Crect width='4' height='4' fill='none'/%3E%3Ccircle cx='1' cy='1' r='0.5' fill='rgba(255,255,255,0.015)'/%3E%3C/svg%3E");
    pointer-events: none;
  }}

  .board-title {{
    font-family: 'Outfit', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: rgba(255,255,200,0.5);
    text-transform: uppercase;
    margin-bottom: 12px;
    text-align: center;
    text-shadow: 0 0 8px rgba(255,255,200,0.3);
  }}

  .board-divider {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,200,0.3), transparent);
    margin-bottom: 14px;
  }}

  /* Chalk-written error items */
  .error-item {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 12px;
    opacity: 0;
    transform: translateX(-10px);
    transition: opacity 0.5s ease, transform 0.5s ease;
    font-family: 'Outfit', sans-serif;
  }}
  .error-item.revealed {{
    opacity: 1;
    transform: translateX(0);
  }}

  .error-bullet {{
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.6rem; font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
  }}
  .error-bullet.e1 {{ background: rgba(255, 80, 80, 0.3); color: #ff6060; border: 1px solid rgba(255,80,80,0.5); }}
  .error-bullet.e2 {{ background: rgba(255, 200, 0, 0.3); color: #ffd700; border: 1px solid rgba(255,200,0,0.5); }}
  .error-bullet.e3 {{ background: rgba(0, 255, 200, 0.3); color: #00ffc8; border: 1px solid rgba(0,255,200,0.5); }}

  .error-text {{
    font-size: 0.82rem;
    line-height: 1.4;
    color: rgba(230, 255, 230, 0.85);
    text-shadow: 0 0 6px rgba(200,255,200,0.2);
    /* Chalk handwritten feel */
    letter-spacing: 0.02em;
  }}

  /* Chalk underline animation */
  .chalk-underline {{
    position: relative;
  }}
  .chalk-underline::after {{
    content: '';
    position: absolute;
    bottom: -2px; left: 0;
    width: 0%; height: 1px;
    background: rgba(255,255,200,0.4);
    transition: width 1s ease 0.3s;
  }}
  .error-item.revealed .chalk-underline::after {{
    width: 100%;
  }}

  /* Pointer laser beam */
  .pointer-beam {{
    position: absolute;
    width: 0;
    height: 2px;
    background: linear-gradient(90deg, rgba(255,215,0,0.8), transparent);
    top: 0; left: 0;
    transform-origin: left center;
    opacity: 0;
    transition: opacity 0.3s;
    box-shadow: 0 0 8px rgba(255,215,0,0.5);
    pointer-events: none;
    z-index: 5;
  }}
  .pointer-beam.active {{
    opacity: 1;
    animation: beamSweep 1s ease-out forwards;
  }}
  @keyframes beamSweep {{
    from {{ width: 0; }}
    to   {{ width: 60px; }}
  }}

  /* ── SPEECH BUBBLE ── */
  .speech-bubble {{
    position: absolute;
    top: -55px;
    left: 60px;
    background: rgba(4,7,15,0.95);
    border: 1px solid rgba(0,255,200,0.4);
    padding: 8px 14px;
    font-size: 0.7rem;
    color: var(--accent, #00ffc8);
    white-space: nowrap;
    max-width: 200px;
    white-space: normal;
    line-height: 1.4;
    opacity: 0;
    transform: translateY(6px) scale(0.95);
    transition: opacity 0.4s ease, transform 0.4s ease;
    z-index: 50;
    font-family: 'Space Mono', monospace;
    box-shadow: 0 0 20px rgba(0,255,200,0.1);
  }}
  .speech-bubble::before {{
    content: '';
    position: absolute;
    bottom: -8px; left: 20px;
    width: 0; height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid rgba(0,255,200,0.4);
  }}
  .speech-bubble::after {{
    content: '';
    position: absolute;
    bottom: -6px; left: 21px;
    width: 0; height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 7px solid rgba(4,7,15,0.95);
  }}
  .speech-bubble.show {{
    opacity: 1;
    transform: translateY(0) scale(1);
  }}

  /* ── AUDIO CONTROLS BAR ── */
  .controls-bar {{
    background: rgba(0,0,0,0.4);
    border-top: 1px solid rgba(0,255,200,0.12);
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    position: relative;
  }}
  .controls-bar::before {{
    content: '🔊 AUDIO BRIEF';
    position: absolute;
    top: -9px; left: 20px;
    font-size: 0.48rem; letter-spacing: 0.22em; color: rgba(0,255,200,0.4);
    background: #060d1f; padding: 0 6px;
  }}

  #playBtn {{
    width: 38px; height: 38px;
    border: 2px solid #00ffc8; background: rgba(0,255,200,0.1);
    color: #00ffc8; font-size: 1rem; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; transition: all .2s; padding: 0; line-height: 1;
    font-family: monospace;
  }}
  #playBtn:hover {{ background: rgba(0,255,200,0.22); box-shadow: 0 0 18px rgba(0,255,200,0.4); }}
  #playBtn.on {{ animation: glowPulse 1.5s ease-in-out infinite; background: rgba(0,255,200,0.18); }}
  @keyframes glowPulse {{ 0%,100%{{box-shadow:0 0 6px rgba(0,255,200,0.5)}} 50%{{box-shadow:0 0 22px rgba(0,255,200,0.95)}} }}

  .wave {{ display: flex; align-items: center; gap: 2px; height: 28px; flex-shrink: 0; }}
  .bar  {{ width: 3px; background: #00ffc8; opacity: 0.25; transform-origin: center; border-radius: 1px; }}
  .wave.on .bar {{ opacity: 1; animation: waveDance var(--spd,.9s) ease-in-out infinite alternate; }}
  @keyframes waveDance {{ from{{transform:scaleY(0.15)}} to{{transform:scaleY(1)}} }}

  .track-wrap {{ flex: 1; display: flex; flex-direction: column; gap: 6px; }}
  .times {{ display: flex; justify-content: space-between; font-size: .6rem; color: rgba(200,216,232,.38); }}
  #curT  {{ color: #00ffc8; }}
  .track {{
    width: 100%; height: 4px;
    background: rgba(255,255,255,.07); cursor: pointer; position: relative;
  }}
  .fill {{
    height: 100%; width: 0%;
    background: linear-gradient(90deg, #00ffc8, #ff3cac);
    transition: width .15s linear; position: relative;
  }}
  .fill::after {{
    content: ''; position: absolute; right: -5px; top: 50%; transform: translateY(-50%);
    width: 8px; height: 8px; background: #00ffc8;
    box-shadow: 0 0 8px #00ffc8; opacity: 0; transition: opacity .2s; border-radius: 50%;
  }}
  .track:hover .fill::after {{ opacity: 1; }}

  /* Tap overlay */
  #tapOverlay {{
    position: absolute; inset: 0;
    background: rgba(4,7,15,0.82);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; z-index: 100;
    backdrop-filter: blur(2px);
  }}
  .tap-btn {{
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.2em; color: #00ffc8;
    text-transform: uppercase;
    border: 1px solid rgba(0,255,200,0.45);
    padding: 0.6rem 1.6rem;
    background: rgba(0,255,200,0.09);
    animation: tapPulse 1.6s ease-in-out infinite;
    display: flex; align-items: center; gap: 10px;
  }}
  @keyframes tapPulse {{
    0%,100%{{ box-shadow: 0 0 0 rgba(0,255,200,0); }}
    50%{{ box-shadow: 0 0 20px rgba(0,255,200,0.3); }}
  }}

  /* Entrance animation */
  .scene {{
    animation: sceneIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
  }}
  @keyframes sceneIn {{
    from {{ opacity: 0; transform: translateY(20px) scale(0.97); }}
    to   {{ opacity: 1; transform: translateY(0) scale(1); }}
  }}

  /* Glow ring around teacher when speaking */
  .teacher-glow {{
    position: absolute;
    bottom: -4px; left: 50%; transform: translateX(-50%);
    width: 110px; height: 110px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,255,200,0) 40%, rgba(0,255,200,0.15) 70%, transparent 100%);
    opacity: 0;
    transition: opacity 0.4s;
    pointer-events: none;
  }}
  .teacher-glow.active {{ opacity: 1; animation: glowRing 1.5s ease-in-out infinite; }}
  @keyframes glowRing {{
    0%,100% {{ transform: translateX(-50%) scale(1); opacity: 0.5; }}
    50% {{ transform: translateX(-50%) scale(1.1); opacity: 1; }}
  }}

</style>
</head>
<body>

<div class="scene" id="scene">
  <!-- Corner decorations -->
  <div class="corner tl"></div>
  <div class="corner tr"></div>
  <div class="corner bl"></div>
  <div class="corner br"></div>
  <div class="scene-label">⬡ AI TEACHER · LIVE ANALYSIS</div>

  <!-- Tap overlay (if not autoplay) -->
  {"" if autoplay else '''
  <div id="tapOverlay" onclick="startSession()">
    <div class="tap-btn">▶&nbsp;&nbsp;TAP TO START AI TEACHER</div>
  </div>
  '''}

  <!-- Content row: Teacher + Chalkboard -->
  <div class="content-row">

    <!-- TEACHER FIGURE -->
    <div class="teacher-wrap" id="teacherWrap">
      <div class="teacher-glow" id="teacherGlow"></div>
      <div class="teacher-shadow"></div>

      <!-- Speech bubble above teacher -->
      <div class="speech-bubble" id="speechBubble">
        Let me walk you through the issues I found...
      </div>

      <!-- SVG Teacher Character - Clay/3D Style -->
      <svg class="teacher-svg" id="teacherSvg" viewBox="0 0 130 280" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <!-- Skin gradient -->
          <radialGradient id="skinGrad" cx="45%" cy="35%" r="55%">
            <stop offset="0%" stop-color="#f5c5a3"/>
            <stop offset="60%" stop-color="#e8a882"/>
            <stop offset="100%" stop-color="#c8855a"/>
          </radialGradient>
          <!-- Hair gradient -->
          <radialGradient id="hairGrad" cx="50%" cy="20%" r="60%">
            <stop offset="0%" stop-color="#b5651d"/>
            <stop offset="100%" stop-color="#7a3b0d"/>
          </radialGradient>
          <!-- Jacket gradient -->
          <radialGradient id="jacketGrad" cx="40%" cy="30%" r="70%">
            <stop offset="0%" stop-color="#5b8fcc"/>
            <stop offset="60%" stop-color="#3a6aaa"/>
            <stop offset="100%" stop-color="#2a4f8a"/>
          </radialGradient>
          <!-- Skirt gradient -->
          <radialGradient id="skirtGrad" cx="50%" cy="20%" r="60%">
            <stop offset="0%" stop-color="#555"/>
            <stop offset="100%" stop-color="#333"/>
          </radialGradient>
          <!-- Shirt gradient -->
          <linearGradient id="shirtGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#f0f0f0"/>
            <stop offset="100%" stop-color="#d0d0d0"/>
          </linearGradient>
          <!-- Shadow filter -->
          <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="2" dy="4" stdDeviation="4" flood-color="rgba(0,0,0,0.4)"/>
          </filter>
          <!-- Glow filter -->
          <filter id="glow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <!-- Glasses lens gradient -->
          <radialGradient id="lensGrad" cx="40%" cy="35%" r="65%">
            <stop offset="0%" stop-color="rgba(100,180,255,0.3)"/>
            <stop offset="100%" stop-color="rgba(50,120,200,0.1)"/>
          </radialGradient>
        </defs>

        <!-- ═══ BODY ═══ -->

        <!-- Legs (behind skirt) -->
        <rect x="43" y="218" width="18" height="50" rx="4" fill="#2a2a35" filter="url(#shadow)"/>
        <rect x="67" y="218" width="18" height="50" rx="4" fill="#2a2a35" filter="url(#shadow)"/>
        <!-- Shoes -->
        <ellipse cx="52" cy="268" rx="12" ry="5" fill="#1a1a20" filter="url(#shadow)"/>
        <ellipse cx="76" cy="268" rx="12" ry="5" fill="#1a1a20" filter="url(#shadow)"/>

        <!-- SKIRT -->
        <path d="M 30 175 Q 28 210 38 218 L 88 218 Q 98 210 96 175 Z"
              fill="url(#skirtGrad)" filter="url(#shadow)"/>
        <!-- Skirt highlight -->
        <path d="M 45 178 Q 50 200 55 215" stroke="rgba(255,255,255,0.1)" stroke-width="2" fill="none"/>

        <!-- TORSO / SHIRT -->
        <path d="M 38 118 Q 35 150 36 175 L 90 175 Q 92 150 88 118 Z"
              fill="url(#shirtGrad)" filter="url(#shadow)"/>

        <!-- JACKET -->
        <path d="M 22 112 Q 18 140 20 175 L 40 175 Q 38 145 40 118 Z"
              fill="url(#jacketGrad)" filter="url(#shadow)"/>
        <path d="M 104 112 Q 108 140 106 175 L 86 175 Q 88 145 86 118 Z"
              fill="url(#jacketGrad)" filter="url(#shadow)"/>
        <!-- Jacket lapels -->
        <path d="M 40 118 L 55 130 L 60 118" fill="url(#jacketGrad)"/>
        <path d="M 86 118 L 71 130 L 66 118" fill="url(#jacketGrad)"/>
        <!-- Jacket collar highlight -->
        <path d="M 22 115 Q 30 120 40 118" stroke="rgba(100,150,220,0.5)" stroke-width="1.5" fill="none"/>
        <path d="M 104 115 Q 96 120 86 118" stroke="rgba(100,150,220,0.5)" stroke-width="1.5" fill="none"/>

        <!-- Belt/waist -->
        <rect x="36" y="170" width="54" height="6" rx="1" fill="#222"/>
        <rect x="58" y="171" width="10" height="4" rx="1" fill="#888"/>

        <!-- NECK -->
        <rect x="55" y="96" width="18" height="26" rx="4" fill="url(#skinGrad)"/>

        <!-- Book held in left hand -->
        <g transform="translate(10, 148) rotate(-5)">
          <rect x="0" y="0" width="28" height="36" rx="2" fill="#2d7a2d" filter="url(#shadow)"/>
          <rect x="2" y="0" width="4" height="36" rx="1" fill="#1a5a1a"/>
          <line x1="8" y1="8" x2="24" y2="8" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
          <line x1="8" y1="12" x2="24" y2="12" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
          <line x1="8" y1="16" x2="20" y2="16" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
          <rect x="0" y="0" width="28" height="36" rx="2" fill="none"
                stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/>
        </g>

        <!-- LEFT ARM (holding book) -->
        <path d="M 22 115 Q 14 135 12 162" stroke="url(#jacketGrad)" stroke-width="16" fill="none"
              stroke-linecap="round" filter="url(#shadow)"/>
        <!-- Left hand -->
        <ellipse cx="12" cy="164" rx="9" ry="8" fill="url(#skinGrad)"/>

        <!-- RIGHT ARM (pointer arm) - animated -->
        <g id="armPointer" class="arm-pointer resting">
          <!-- Upper arm -->
          <path d="M 104 115 Q 114 130 118 150"
                stroke="url(#jacketGrad)" stroke-width="16" fill="none"
                stroke-linecap="round" filter="url(#shadow)"/>
          <!-- Forearm -->
          <path d="M 118 150 Q 122 162 120 172"
                stroke="url(#jacketGrad)" stroke-width="13" fill="none"
                stroke-linecap="round"/>
          <!-- Hand -->
          <ellipse cx="120" cy="174" rx="8" ry="7" fill="url(#skinGrad)"/>
          <!-- Pointing finger -->
          <path d="M 122 170 L 128 160 L 130 158"
                stroke="url(#skinGrad)" stroke-width="5" fill="none"
                stroke-linecap="round" id="fingerPointer"/>
          <!-- Pencil/pointer -->
          <line id="pencilPointer" x1="129" y1="158" x2="138" y2="145"
                stroke="#ffd700" stroke-width="2.5" stroke-linecap="round"
                filter="url(#glow)" opacity="0"/>
          <!-- Pencil tip glow -->
          <circle id="pencilTip" cx="138" cy="145" r="3" fill="#ffd700" opacity="0"
                  filter="url(#glow)"/>
        </g>

        <!-- SHOULDERS -->
        <ellipse cx="26" cy="116" rx="16" ry="12" fill="url(#jacketGrad)" filter="url(#shadow)"/>
        <ellipse cx="100" cy="116" rx="16" ry="12" fill="url(#jacketGrad)" filter="url(#shadow)"/>

        <!-- HEAD -->
        <ellipse cx="63" cy="68" rx="32" ry="36" fill="url(#skinGrad)" filter="url(#shadow)"/>
        <!-- Head highlight -->
        <ellipse cx="55" cy="52" rx="12" ry="10" fill="rgba(255,220,190,0.3)"/>

        <!-- HAIR -->
        <path d="M 32 60 Q 28 30 40 20 Q 55 8 75 10 Q 90 12 95 28 Q 98 42 94 56
                 Q 85 38 80 35 Q 70 28 63 28 Q 50 28 44 35 Q 38 42 36 58 Z"
              fill="url(#hairGrad)" filter="url(#shadow)"/>
        <!-- Hair strands -->
        <path d="M 94 56 Q 100 48 98 38" stroke="#a04010" stroke-width="3" fill="none" opacity="0.6"/>
        <!-- Side hair -->
        <path d="M 33 62 Q 28 80 30 95" stroke="url(#hairGrad)" stroke-width="8" fill="none"
              stroke-linecap="round"/>
        <path d="M 93 62 Q 98 80 96 100" stroke="url(#hairGrad)" stroke-width="8" fill="none"
              stroke-linecap="round"/>

        <!-- EYES (with glasses) -->
        <!-- Glasses frame -->
        <rect x="36" y="60" width="22" height="16" rx="3" fill="none"
              stroke="#2a1a0a" stroke-width="2"/>
        <rect x="64" y="60" width="22" height="16" rx="3" fill="none"
              stroke="#2a1a0a" stroke-width="2"/>
        <!-- Lens fill -->
        <rect x="37" y="61" width="20" height="14" rx="2" fill="url(#lensGrad)"/>
        <rect x="65" y="61" width="20" height="14" rx="2" fill="url(#lensGrad)"/>
        <!-- Bridge -->
        <line x1="58" y1="67" x2="64" y2="67" stroke="#2a1a0a" stroke-width="2"/>
        <!-- Temples -->
        <line x1="36" y1="67" x2="30" y2="65" stroke="#2a1a0a" stroke-width="2"/>
        <line x1="86" y1="67" x2="92" y2="65" stroke="#2a1a0a" stroke-width="2"/>
        <!-- Eyeballs -->
        <circle cx="47" cy="68" r="6" fill="white"/>
        <circle cx="75" cy="68" r="6" fill="white"/>
        <!-- Irises -->
        <circle cx="48" cy="68.5" r="4" fill="#4a2800"/>
        <circle cx="76" cy="68.5" r="4" fill="#4a2800"/>
        <!-- Pupils -->
        <circle cx="48.5" cy="68.5" r="2.2" fill="#1a0a00"/>
        <circle cx="76.5" cy="68.5" r="2.2" fill="#1a0a00"/>
        <!-- Eye shine -->
        <circle cx="50" cy="67" r="1.2" fill="white" opacity="0.9"/>
        <circle cx="78" cy="67" r="1.2" fill="white" opacity="0.9"/>
        <!-- Blink animation via JS (class toggle) -->
        <rect id="blinkL" x="37" y="61" width="20" height="0" rx="2" fill="#e8a882"/>
        <rect id="blinkR" x="65" y="61" width="20" height="0" rx="2" fill="#e8a882"/>

        <!-- Eyebrows -->
        <path d="M 38 58 Q 47 54 57 57" stroke="#7a3b0d" stroke-width="2.5" fill="none"
              stroke-linecap="round" id="browL"/>
        <path d="M 65 57 Q 75 54 85 58" stroke="#7a3b0d" stroke-width="2.5" fill="none"
              stroke-linecap="round" id="browR"/>

        <!-- NOSE -->
        <path d="M 60 72 Q 58 80 60 84 Q 63 87 66 84 Q 68 80 66 72"
              fill="rgba(180,100,60,0.3)" stroke="rgba(180,100,60,0.4)" stroke-width="0.5"/>
        <!-- Nostrils -->
        <ellipse cx="60" cy="84" rx="3" ry="2" fill="rgba(150,80,40,0.4)"/>
        <ellipse cx="66" cy="84" rx="3" ry="2" fill="rgba(150,80,40,0.4)"/>

        <!-- MOUTH -->
        <g id="mouthGroup" transform="translate(63, 92)">
          <!-- Lips -->
          <path d="M -12 0 Q -6 -3 0 -2 Q 6 -3 12 0 Q 6 4 0 5 Q -6 4 -12 0 Z"
                fill="#c8706a"/>
          <!-- Upper lip line -->
          <path d="M -12 0 Q -6 -3 0 -2 Q 6 -3 12 0" stroke="#a05050" stroke-width="0.8" fill="none"/>
          <!-- Mouth opening (animated) -->
          <ellipse id="mouthOpening" cx="0" cy="2" rx="7" ry="0" fill="#2a0a0a"/>
          <!-- Teeth (visible when mouth open) -->
          <rect id="teeth" x="-5" y="1" width="10" height="3" rx="1" fill="white" opacity="0"/>
          <!-- Smile lines -->
          <path id="smileL" d="M -13 0 Q -16 3 -14 6" stroke="rgba(180,100,60,0.4)" stroke-width="1" fill="none"/>
          <path id="smileR" d="M 13 0 Q 16 3 14 6" stroke="rgba(180,100,60,0.4)" stroke-width="1" fill="none"/>
        </g>

        <!-- Cheek blush -->
        <ellipse cx="38" cy="82" rx="8" ry="5" fill="rgba(255,150,130,0.2)"/>
        <ellipse cx="88" cy="82" rx="8" ry="5" fill="rgba(255,150,130,0.2)"/>

        <!-- Collar / shirt visible at neck -->
        <path d="M 50 108 L 57 120 L 63 115 L 69 120 L 76 108"
              fill="url(#shirtGrad)" stroke="rgba(200,200,200,0.3)" stroke-width="0.5"/>

        <!-- Cyber scan line on teacher (decorative) -->
        <line id="scanLine" x1="30" y1="50" x2="96" y2="50"
              stroke="rgba(0,255,200,0.3)" stroke-width="1" opacity="0">
          <animate attributeName="y1" values="50;200;50" dur="4s" repeatCount="indefinite"/>
          <animate attributeName="y2" values="50;200;50" dur="4s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="0;0.4;0" dur="4s" repeatCount="indefinite"/>
        </line>
      </svg>
    </div>

    <!-- CHALKBOARD -->
    <div class="chalkboard-wrap">
      <div class="chalkboard" id="chalkboard">
        <div class="board-title">★ ISSUES DETECTED ★</div>
        <hr class="board-divider">
        <div id="errorList">
          <!-- Error items injected by JS -->
        </div>
      </div>
    </div>

  </div>

  <!-- AUDIO CONTROLS BAR -->
  <div class="controls-bar">
    <audio id="aud" preload="auto">
      <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
    </audio>

    <button id="playBtn" onclick="toggle()">▶</button>

    <div class="wave" id="wave">
      {''.join(f'<div class="bar" style="height:{h}px;--spd:{s}s"></div>' for h,s in [(10,0.70),(18,0.50),(26,0.90),(14,0.60),(22,0.80),(10,0.55),(28,0.75),(16,0.65),(20,0.85),(12,0.70),(24,0.60),(18,0.90),(8,0.50),(22,0.70),(14,0.80)])}
    </div>

    <div class="track-wrap">
      <div class="times"><span id="curT">0:00</span><span id="durT">–:––</span></div>
      <div class="track" onclick="seek(event)"><div class="fill" id="fill"></div></div>
    </div>
  </div>

</div>

<script>
// ── ERROR POINTS ──
var errorPoints = {ep_js};

// ── DOM refs ──
var aud      = document.getElementById('aud');
var btn      = document.getElementById('playBtn');
var wave     = document.getElementById('wave');
var fill     = document.getElementById('fill');
var curT     = document.getElementById('curT');
var durT     = document.getElementById('durT');
var glow     = document.getElementById('teacherGlow');
var speech   = document.getElementById('speechBubble');
var armEl    = document.getElementById('armPointer');
var mouthOp  = document.getElementById('mouthOpening');
var teethEl  = document.getElementById('teeth');
var blinkL   = document.getElementById('blinkL');
var blinkR   = document.getElementById('blinkR');
var browL    = document.getElementById('browL');
var browR    = document.getElementById('browR');
var errorList= document.getElementById('errorList');

// ── Build error items on chalkboard ──
var bulletClasses = ['e1','e2','e3'];
var bulletLabels  = ['01','02','03'];
errorPoints.forEach(function(txt, i) {{
  var div = document.createElement('div');
  div.className = 'error-item';
  div.id = 'ei' + i;
  div.innerHTML =
    '<div class="error-bullet ' + bulletClasses[i] + '">' + bulletLabels[i] + '</div>' +
    '<div class="error-text chalk-underline">' + txt + '</div>';
  errorList.appendChild(div);
}});

// ── Utility ──
function fmt(s) {{
  if (!isFinite(s)) return '–:––';
  return Math.floor(s/60) + ':' + ('0'+Math.floor(s%60)).slice(-2);
}}

// ── UI state ──
function setPlaying(playing) {{
  btn.innerHTML  = playing ? '⏸' : '▶';
  playing ? btn.classList.add('on')    : btn.classList.remove('on');
  playing ? wave.classList.add('on')   : wave.classList.remove('on');
  glow.classList.toggle('active', playing);
  if (!playing) stopMouth();
}}

// ── Audio events ──
aud.addEventListener('loadedmetadata', function() {{ durT.textContent = fmt(aud.duration); }});
aud.addEventListener('timeupdate', function() {{
  curT.textContent = fmt(aud.currentTime);
  fill.style.width = (aud.duration ? aud.currentTime/aud.duration*100 : 0) + '%';
  updateTeacherAnim();
}});
aud.addEventListener('ended', function() {{
  setPlaying(false); fill.style.width='0%'; curT.textContent='0:00';
  retractAll();
}});
aud.addEventListener('play',  function() {{ setPlaying(true);  onAudioStart(); }});
aud.addEventListener('pause', function() {{ setPlaying(false); }});

function toggle() {{ aud.paused ? aud.play() : aud.pause(); }}
function seek(e)  {{
  var r = e.currentTarget.getBoundingClientRect();
  if (aud.duration) aud.currentTime = (e.clientX - r.left) / r.width * aud.duration;
}}

// ── TEACHER ANIMATIONS ──
var mouthAnim = null;
var talkPhase = 0;
var errorShown = [false, false, false];
var speechMessages = [
  "Let me walk you through the issues I found...",
  "Here\u2019s what needs to be fixed in your code:",
  "Great, the analysis is complete!"
];

function onAudioStart() {{
  // Show speech bubble
  speech.textContent = speechMessages[0];
  speech.classList.add('show');

  // Start mouth talking
  startMouth();

  // Point arm at board
  setTimeout(function() {{
    armEl.classList.remove('resting');
    armEl.classList.add('pointing');
    var pencil = document.getElementById('pencilPointer');
    var tip    = document.getElementById('pencilTip');
    if (pencil) {{ pencil.style.opacity = '1'; }}
    if (tip)    {{ tip.style.opacity = '1'; }}
  }}, 400);

  // Reveal errors with staggered timing
  var dur = aud.duration || 10;
  var third = dur / 3;

  setTimeout(function() {{ revealError(0); }}, 800);
  setTimeout(function() {{ revealError(1); speech.textContent = speechMessages[1]; }}, third * 1000);
  setTimeout(function() {{ revealError(2); }}, third * 2 * 1000);
}}

function revealError(idx) {{
  var el = document.getElementById('ei' + idx);
  if (el) {{ el.classList.add('revealed'); }}
}}

function retractAll() {{
  // Arm returns to rest
  armEl.classList.remove('pointing');
  armEl.classList.add('resting');
  var pencil = document.getElementById('pencilPointer');
  var tip    = document.getElementById('pencilTip');
  if (pencil) pencil.style.opacity = '0';
  if (tip)    tip.style.opacity = '0';

  // Update speech bubble
  speech.textContent = speechMessages[2];
  setTimeout(function() {{ speech.classList.remove('show'); }}, 3000);
}}

// ── MOUTH SYNC ──
var mouthInterval = null;
var mouthOpen = false;
function startMouth() {{
  if (mouthInterval) return;
  mouthInterval = setInterval(function() {{
    mouthOpen = !mouthOpen;
    var h = mouthOpen ? (3 + Math.random() * 4) : 0.3;
    mouthOp.setAttribute('ry', h);
    teethEl.style.opacity = mouthOpen ? '0.9' : '0';
    // Eyebrow expressiveness
    if (mouthOpen && Math.random() > 0.7) {{
      browL.setAttribute('d', 'M 38 56 Q 47 52 57 55');
      browR.setAttribute('d', 'M 65 55 Q 75 52 85 56');
    }} else {{
      browL.setAttribute('d', 'M 38 58 Q 47 54 57 57');
      browR.setAttribute('d', 'M 65 57 Q 75 54 85 58');
    }}
  }}, 160 + Math.random() * 80);
}}
function stopMouth() {{
  clearInterval(mouthInterval);
  mouthInterval = null;
  mouthOp.setAttribute('ry', 0);
  teethEl.style.opacity = '0';
  browL.setAttribute('d', 'M 38 58 Q 47 54 57 57');
  browR.setAttribute('d', 'M 65 57 Q 75 54 85 58');
}}

// ── BLINK ANIMATION ──
function doBlink() {{
  // Animate eyelid drop
  var steps = [0,4,9,14,9,4,0];
  var i = 0;
  var blinkTimer = setInterval(function() {{
    var h = steps[i];
    blinkL.setAttribute('height', h);
    blinkR.setAttribute('height', h);
    i++;
    if (i >= steps.length) {{
      clearInterval(blinkTimer);
      blinkL.setAttribute('height', 0);
      blinkR.setAttribute('height', 0);
    }}
  }}, 35);
}}
// Blink every 3-6 seconds
function scheduleBlink() {{
  setTimeout(function() {{ doBlink(); scheduleBlink(); }}, 3000 + Math.random()*3000);
}}
scheduleBlink();

// ── UPDATE teacher based on playback position ──
function updateTeacherAnim() {{
  if (aud.paused) return;
  var pct = aud.duration ? aud.currentTime / aud.duration : 0;
  // Slow arm sweep as audio progresses
  var baseAngle = -25;
  var sweep = pct * 15; // sweeps 15 degrees across duration
  if (armEl.classList.contains('pointing')) {{
    armEl.style.transform = 'rotate(' + (baseAngle + sweep) + 'deg)';
  }}
}}

// ── TAP TO PLAY (when autoplay disabled) ──
function startSession() {{
  var overlay = document.getElementById('tapOverlay');
  if (overlay) overlay.remove();
  aud.play();
}}

// ── AUTOPLAY ──
{"aud.play().catch(function(){});" if autoplay else ""}
</script>
</body>
</html>"""

    components.html(html, height=590)


# ============================================================
#  CHAT CONTAINER
# ============================================================
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# ── Replay history ──
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'''
        <div class="user-message-container">
            <div class="user-avatar">YOU</div>
            <div class="user-message-wrapper">
                <div class="user-label">you</div>
                <div class="user-message">{message["content"]}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="assistant-message-container">
            <div class="assistant-avatar">AI</div>
            <div class="assistant-message-wrapper">
                <div class="assistant-label">ai_debugger</div>
                <div class="assistant-message">{message["content"]}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        if message.get("audio_b64"):
            idx = st.session_state.messages.index(message)
            st.markdown('<div class="audio-wrapper">', unsafe_allow_html=True)
            render_teacher_player(
                message["audio_b64"],
                message.get("error_points", ["Bug detected", "Logic error", "Needs refactor"]),
                autoplay=False,
                player_id=f"hist_{idx}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
#  CHAT INPUT
# ============================================================
if user_input := st.chat_input("⬡ Paste buggy code or ask anything..."):

    st.session_state.char_count += len(user_input)

    # User message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'''
    <div class="user-message-container">
        <div class="user-avatar">YOU</div>
        <div class="user-message-wrapper">
            <div class="user-label">you</div>
            <div class="user-message">{user_input}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Thinking indicator
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown('''
    <div class="assistant-message-container">
        <div class="assistant-avatar">AI</div>
        <div class="assistant-message-wrapper">
            <div class="assistant-label">ai_debugger</div>
            <div class="assistant-message" style="color: var(--accent);">
                <span>⬡ Analyzing architecture</span>
                <span style="display:inline-block;width:30px;">
                    <span style="animation:dotBounce 1.2s ease-in-out infinite;">.</span>
                    <span style="animation:dotBounce 1.2s ease-in-out infinite 0.2s;">.</span>
                    <span style="animation:dotBounce 1.2s ease-in-out infinite 0.4s;">.</span>
                </span>
            </div>
        </div>
    </div>
    <style>
    @keyframes dotBounce {
        0%,80%,100%{opacity:0.3;transform:translateY(0)}
        40%{opacity:1;transform:translateY(-5px)}
    }
    </style>
    ''', unsafe_allow_html=True)

    with st.status("⬡ Processing...", expanded=True) as status:
        st.write("→ Running static analysis...")
        full_fix = get_ai_fix(user_input)

        st.write("→ Extracting error highlights...")
        error_points = get_error_highlights(full_fix)

        st.write("→ Generating audio insight...")
        audio_script = get_audio_summary(full_fix)

        st.write("→ Synthesising teacher voice...")
        audio_b64 = generate_audio_b64(audio_script)

        status.update(label="✓ Analysis complete", state="complete", expanded=False)

    thinking_placeholder.empty()

    # Assistant text response
    st.markdown(f'''
    <div class="assistant-message-container">
        <div class="assistant-avatar">AI</div>
        <div class="assistant-message-wrapper">
            <div class="assistant-label">ai_debugger</div>
            <div class="assistant-message">{full_fix}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # 🎓 AI TEACHER PLAYER
    st.markdown('<div class="audio-wrapper">', unsafe_allow_html=True)
    render_teacher_player(audio_b64, error_points, autoplay=True, player_id=f"new_{st.session_state.fix_count}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Persist
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_fix,
        "audio_script": audio_script,
        "audio_b64": audio_b64,
        "error_points": error_points,
    })
    st.session_state.fix_count += 1

    st.rerun()