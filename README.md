# AskVicky Web 🤖

Browser version of the AskVicky Android app. Same system prompt, same quick prompts, built with Streamlit and Groq.

🌐 **Live app: https://askvicky-ai.streamlit.app/**

## Run locally

```
pip install -r requirements.txt
copy .streamlit\secrets.toml.example .streamlit\secrets.toml   (Windows)
# edit secrets.toml and paste your Groq key
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (e.g. `AskVicky-Web`).
2. share.streamlit.io -> New app -> pick the repo, main file `app.py`.
3. App settings -> Secrets -> add `GROQ_API_KEY = "gsk_..."`.

## Limits (the app is public, the Groq key is shared)

- 40 messages per browser session
- 500 messages per day across all visitors (edit `GLOBAL_DAILY_CAP` in `app.py`)
- Recent chats live only in the current browser session; use "Download this chat" to keep one

## Changing the model

Set `GROQ_TEXT_MODEL` in Secrets. No code change needed when Groq retires a model.
