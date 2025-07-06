import streamlit as st
import openai
import faiss
import json
import numpy as np
import pathlib
import tempfile
import os
import streamlit.components.v1 as components
from moviepy.editor import VideoFileClip

openai.api_key = os.getenv("OPENAI_API_KEY")

INDEX_FILE = "faiss_index.index"
METADATA_FILE = "chunk_metadata.json"
VIDEO_FOLDER = "."

# Load search index + metadata
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Convert 00:00:12,000 → seconds
def parse_srt_timecode(ts):
    h, m, s = ts.replace(",", ".").split(":")
    return float(h)*3600 + float(m)*60 + float(s)

# Clip and export a video segment
def clip_video_segment(filename, start_sec, duration):
    try:
        clip = VideoFileClip(str(filename)).subclip(start_sec, start_sec + duration)
        tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        clip.write_videofile(tmpfile.name, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        return tmpfile.name
    except Exception as e:
        print(f"⚠️ Error clipping: {e}")
        return None

# Embed query using OpenAI
def embed_query(query):
    res = openai.Embedding.create(
        model="text-embedding-3-large",
        input=query
    )
    return np.array(res["data"][0]["embedding"], dtype=np.float32).reshape(1, -1)

# Run semantic search
def search(query, k=3):
    vec = embed_query(query)
    distances, indices = index.search(vec, k)
    return [metadata[i] for i in indices[0]]

# GPT summary
@st.cache_data(show_spinner=False)
def summarize_with_gpt(text):
    try:
        res = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Summarize this Holocaust survivor testimony excerpt in 1–2 sentences."},
                {"role": "user", "content": text}
            ]
        )
        return res["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"⚠️ Summary failed: {e}"

# Streamlit UI
st.set_page_config(page_title="Ask al Viejo Loco", layout="wide")
st.title("🎥 Ask al Viejo Loco")

query = st.text_input("Ask a question", placeholder="e.g. ¿Dónde menciona el gueto de Varsovia?")
clip_duration = st.slider("Clip length (seconds):", 10, 120, 60, step=10)

if query:
    st.info("🔎 Searching archive...")
    results = search(query)

    for i, result in enumerate(results, 1):
        st.markdown(f"---\n### Match #{i}")
        st.markdown(f"**⏱ {result['start']} → {result['end']}** — *{result['video']}*")

        summary = summarize_with_gpt(result["text"])
        st.markdown(f"**🧠 Summary:** {summary}")

        start_sec = parse_srt_timecode(result["start"])
        video_path = pathlib.Path(VIDEO_FOLDER) / result["video"]

        if video_path.exists():
            clip_path = clip_video_segment(video_path, start_sec, clip_duration)
            if clip_path:
                st.video(clip_path)
            else:
                st.warning("⚠️ Could not generate video clip.")
        else:
            st.error(f"⚠️ Video not found: {result['video']}")

        # Load full transcript
        srt_path = pathlib.Path("transcripts") / result["video"].replace(".mp4", ".srt")
        if srt_path.exists():
            with open(srt_path, "r", encoding="utf-8") as f:
                blocks = f.read().strip().split("\n\n")

            html = ""
            for block in blocks:
                lines = block.strip().splitlines()
                if len(lines) < 3: continue
                start, end = lines[1].split(" --> ")
                text = " ".join(lines[2:])
                is_hit = (start == result["start"] and end == result["end"])
                html += f"<div style='padding:5px; background:{'#ffe' if is_hit else '#f8f8f8'}; border-radius:4px; margin-bottom:4px;'>"
                html += f"<b>{start} → {end}</b><br>{text}</div>"

            st.markdown("#### 📜 Transcript")
            components.html(
                f"<div style='max-height:300px; overflow-y:auto'>{html}</div>",
                height=320, scrolling=True)

