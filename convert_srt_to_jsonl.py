import os
import json

SOURCE_DIR = "transcripts"
OUTPUT_FILE = "transcript_chunks.jsonl"

def parse_srt(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read().strip()
    blocks = content.split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        timestamp = lines[1]
        start, end = timestamp.split(" --> ")
        text = " ".join(lines[2:])
        yield {
            "start": start.strip(),
            "end": end.strip(),
            "text": text.strip()
        }

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for filename in sorted(os.listdir(SOURCE_DIR)):
        if filename.endswith(".srt"):
            video_name = filename.replace(".srt", ".mp4")
            for block in parse_srt(os.path.join(SOURCE_DIR, filename)):
                block["video"] = video_name
                out.write(json.dumps(block, ensure_ascii=False) + "\n")

