import os
import threading
import datetime as dt

import streamlit as st

st.set_page_config(
    page_title="AskVicky",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Same key name as Vickys-Email-Writer, so one Groq key works for both apps.
DEFAULT_MODEL = "openai/gpt-oss-120b"
EMAIL_WRITER_URL = "https://vickys-email-writer.streamlit.app/"

MAX_MSGS_PER_SESSION = 40      # per browser session
GLOBAL_DAILY_CAP = 500         # across all visitors, resets daily (UTC)
HISTORY_WINDOW = 20            # last N messages sent to the model
MAX_RECENTS = 20

SYSTEM_PROMPT = """You are AskVicky — a highly capable, friendly AI assistant for everyone — professionals, students, and anyone who needs help. You can help with absolutely any topic:

- SQL & Databases: Oracle SQL, PL/SQL, Finacle CBS schema (tbaadm, crmuser, custom tables like gam, lam, lht, ldt, eab)
- Banking & Finance: RBI guidelines, NPA classification, NABARD rules, NACH/NPCI, co-operative banking, Finacle CBS operations
- Mathematics: EMI, simple/compound interest, NPA provisioning. Always use Indian format (Rs, lakhs, crores)
- Coding: Python, JavaScript, Shell scripting, any programming language
- Writing: Emails, reports, letters, translations (Marathi, Hindi, English)
- General Knowledge: Science, history, law, current affairs, technology, any topic
- Career: Resume writing, interview tips, productivity

Always give complete, accurate, practical answers. Show step-by-step for calculations. Format code properly. Be helpful, warm and professional."""

QUICK_CHIPS = [
    ("🗄️", "SQL Query", "Write a SQL query to find all NPA accounts with overdue amount from tbaadm schema"),
    ("🧮", "EMI Calc", "Calculate EMI for a loan of Rs 10,00,000 at 9% interest for 5 years. Show step by step."),
    ("🐍", "Python", "Write a Python script to read a CSV file and calculate column totals"),
    ("🏦", "NPA Rules", "Explain RBI guidelines for NPA classification with examples"),
    ("🌏", "Translate", "Translate to Marathi: Good morning, how are you? I need your help with this work."),
    ("⚙️", "Shell Script", "Write a shell script to take Oracle database backup and send email alert"),
    ("📊", "Interest", "Calculate compound interest on Rs 5,00,000 for 3 years at 8.5% per annum"),
]

CAPABILITIES = [
    ("🗄️", "SQL & Oracle", "Finacle schema, PL/SQL"),
    ("💻", "Coding", "Python, Shell, JS"),
    ("🧮", "Calculations", "EMI, interest, tax"),
    ("🏦", "Banking", "RBI rules, NPA, CBS"),
    ("✍️", "Writing", "Emails, reports"),
    ("🌍", "Any Topic", "Science, law, history"),
]

# ── SECRETS / CONFIG HELPERS ──────────────────────────────────────────────────
def get_secret(name, default=""):
    """st.secrets first (Streamlit Cloud), then environment (local dev)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


MODEL = get_secret("GROQ_TEXT_MODEL", DEFAULT_MODEL)


@st.cache_resource
def get_client(api_key):
    from groq import Groq
    return Groq(api_key=api_key)


@st.cache_resource
def global_counter():
    """One counter shared by every visitor of this running app."""
    return {"day": None, "count": 0, "lock": threading.Lock()}


def take_global_slot():
    """Reserve one request against the daily cap. Returns False when exhausted."""
    ctr = global_counter()
    today = dt.datetime.now(dt.timezone.utc).date()
    with ctr["lock"]:
        if ctr["day"] != today:
            ctr["day"], ctr["count"] = today, 0
        if ctr["count"] >= GLOBAL_DAILY_CAP:
            return False
        ctr["count"] += 1
        return True


def friendly_error(exc):
    """Turn an API exception into a message that is useful and never leaks the key."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if status == 404:
        return ("⚠️ The AI model is not available. Groq may have retired it. "
                "The app owner needs to set a new model name (`GROQ_TEXT_MODEL`).")
    if status == 401 or name == "AuthenticationError":
        return "⚠️ The API key was rejected. The app owner needs to check `GROQ_API_KEY`."
    if status == 429 or name == "RateLimitError":
        return "⚠️ The free AI quota is used up for now. Please try again in a few minutes."
    if name in ("APIConnectionError", "APITimeoutError"):
        return "⚠️ Could not reach the AI service. Please try again."
    return f"⚠️ Something went wrong ({name}). Please try again."


# ── SESSION STATE ─────────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("chat_id", f"c_{dt.datetime.now().timestamp()}")
ss.setdefault("recents", [])      # kept for this browser session only
ss.setdefault("sent", 0)
ss.setdefault("pending", None)


def upsert_recent():
    msgs = ss.messages
    if not msgs:
        return
    entry = {
        "id": ss.chat_id,
        "preview": msgs[0]["content"][:50],
        "time": dt.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "messages": list(msgs),
    }
    ss.recents = [r for r in ss.recents if r["id"] != ss.chat_id]
    ss.recents.insert(0, entry)
    ss.recents = ss.recents[:MAX_RECENTS]


def new_chat():
    upsert_recent()
    ss.messages = []
    ss.chat_id = f"c_{dt.datetime.now().timestamp()}"


def open_chat(chat_id):
    upsert_recent()
    for r in ss.recents:
        if r["id"] == chat_id:
            ss.messages = list(r["messages"])
            ss.chat_id = chat_id
            return


def delete_chat(chat_id):
    ss.recents = [r for r in ss.recents if r["id"] != chat_id]
    if chat_id == ss.chat_id:
        ss.messages = []
        ss.chat_id = f"c_{dt.datetime.now().timestamp()}"


def clear_all():
    ss.recents = []
    ss.messages = []
    ss.chat_id = f"c_{dt.datetime.now().timestamp()}"


def queue_prompt(text):
    ss.pending = text


def chat_markdown():
    lines = ["# AskVicky chat\n"]
    for m in ss.messages:
        who = "You" if m["role"] == "user" else "AskVicky"
        lines.append(f"**{who}:**\n\n{m['content']}\n")
    return "\n".join(lines)


# ── MODEL CALL ────────────────────────────────────────────────────────────────
def generate_reply(api_key):
    """Stream the reply into the current chat bubble and return the full text."""
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages[-HISTORY_WINDOW:]]
    try:
        stream = get_client(api_key).chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            # gpt-oss is a reasoning model: reasoning tokens count against this limit,
            # so it must be well above the length of the visible answer.
            max_tokens=4096,
            temperature=0.7,
            stream=True,
        )
    except Exception as exc:
        return friendly_error(exc)

    def chunks():
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                yield piece

    try:
        text = st.write_stream(chunks())
    except Exception as exc:
        return friendly_error(exc)
    if isinstance(text, str) and text.strip():
        return text
    return "Sorry, I could not get a response. Please try again."


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.5rem; max-width: 760px; }
      .av-title { font-size: 1.5rem; font-weight: 700; margin: 0; }
      .av-sub { color: #6b806b; font-size: 0.85rem; margin: 0 0 0.75rem 0; }
      .av-live { color:#22c55e; border:1px solid rgba(34,197,94,.3); background:rgba(34,197,94,.1);
                 border-radius:20px; padding:2px 10px; font-size:.7rem; font-weight:600; }
      .av-card { background:#111811; border:1px solid rgba(255,255,255,.07); border-radius:14px;
                 padding:12px 14px; margin-bottom:10px; }
      .av-card b { font-size:.85rem; } .av-card span { color:#6b806b; font-size:.72rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

head_l, head_r = st.columns([5, 1])
with head_l:
    st.markdown('<p class="av-title">🤖 AskVicky</p>'
                '<p class="av-sub">Ask anything, anytime — Get it done!</p>', unsafe_allow_html=True)
with head_r:
    st.markdown('<div style="text-align:right;padding-top:.6rem"><span class="av-live">● LIVE</span></div>',
                unsafe_allow_html=True)

api_key = get_secret("GROQ_API_KEY")
if not api_key:
    st.error("No Groq API key found. Add `GROQ_API_KEY` in the app's Secrets "
             "(or as an environment variable when running locally). "
             "Get a free key at https://console.groq.com")

# Welcome screen
if not ss.messages:
    st.markdown("#### Hello! I'm AskVicky")
    st.caption("Ask anything, anytime — Get it done! I'm here to help with any topic. 🚀")
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(CAPABILITIES):
        cols[i % 3].markdown(
            f'<div class="av-card"><div style="font-size:1.4rem">{icon}</div>'
            f"<b>{title}</b><br><span>{desc}</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("**Try one:**")
    chip_cols = st.columns(4)
    for i, (icon, label, prompt_text) in enumerate(QUICK_CHIPS):
        chip_cols[i % 4].button(f"{icon} {label}", key=f"chip_{i}", on_click=queue_prompt,
                                args=(prompt_text,), use_container_width=True)
    chip_cols[len(QUICK_CHIPS) % 4].link_button("✉️ Email Writer", EMAIL_WRITER_URL,
                                                use_container_width=True)

# History
for idx, m in enumerate(ss.messages):
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            with st.expander("📋 Copy response"):
                st.code(m["content"], language="markdown")

# New input
user_text = st.chat_input("Ask me anything...", max_chars=4000, disabled=not api_key)
if ss.pending:
    user_text, ss.pending = ss.pending, None

if user_text and api_key:
    ss.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_text)

    with st.chat_message("assistant", avatar="🤖"):
        if ss.sent >= MAX_MSGS_PER_SESSION:
            reply = (f"⚠️ You've reached the limit of {MAX_MSGS_PER_SESSION} messages for this "
                     "session. Please come back later.")
            st.markdown(reply)
        elif not take_global_slot():
            reply = "⚠️ AskVicky has hit its daily free limit. Please try again tomorrow."
            st.markdown(reply)
        else:
            ss.sent += 1
            with st.spinner("AskVicky is thinking..."):
                reply = generate_reply(api_key)
            if reply.startswith("⚠️") or reply.startswith("Sorry,"):
                st.markdown(reply)
    ss.messages.append({"role": "assistant", "content": reply})
    upsert_recent()
    st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### AskVicky")
    st.button("➕ New chat", on_click=new_chat, use_container_width=True)
    st.link_button("✉️ Vicky's Email Writer", EMAIL_WRITER_URL, use_container_width=True)
    if ss.messages:
        st.download_button("⬇️ Download this chat", chat_markdown(), file_name="askvicky_chat.md",
                           mime="text/markdown", use_container_width=True)
    st.markdown("#### Recent chats")
    if not ss.recents:
        st.caption("No recent chats yet. They last until you close or refresh this page.")
    for r in ss.recents:
        c1, c2 = st.columns([5, 1])
        c1.button(f"💬 {r['preview']}", key=f"open_{r['id']}", on_click=open_chat,
                  args=(r["id"],), use_container_width=True, help=f"{len(r['messages'])} messages · {r['time']}")
        c2.button("✕", key=f"del_{r['id']}", on_click=delete_chat, args=(r["id"],))
    if ss.recents or ss.messages:
        st.button("🗑️ Clear all", on_click=clear_all, use_container_width=True)
    st.caption(f"Built with 💚 by Bhushan\n\nPowered by Groq AI · {MODEL}")
