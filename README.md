# AskVicky Web 🤖

Browser version of the AskVicky Android app. Same system prompt, built with Streamlit and Groq. Clean single-screen UI with a link to Vicky's Email Writer and file attachments.

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
- Recent chats (last 20, text only) are saved in this browser's local storage, so they survive closing the app. They stay on your device only: nothing is stored on a server, other browsers/devices do not see them, and clearing site data or "Clear all" removes them. Attached files and images are not saved. Use "Download this chat" to keep one permanently.
- Attachments: PDF, Word, Excel, CSV or text files, plus screenshots/photos (PNG, JPG, WEBP); up to 3 per message, 5 MB each, 10 files and 5 images per session. A message with files counts as 3 against the daily cap, one with images as 10. Scanned PDFs are not read (attach a screenshot of the page instead).
- Attached text is sent to Groq's AI service, so do not upload customer data or confidential bank documents.

## Changing the model

Set `GROQ_TEXT_MODEL` (text, default `openai/gpt-oss-120b`) and `GROQ_VISION_MODEL` (images, default `qwen/qwen3.8-27b`) in Secrets. No code change needed when Groq renames or retires a model.

Images are resized to 1600 px and re-encoded as JPEG before sending, which also removes hidden metadata such as GPS location.
