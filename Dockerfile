FROM python:3.10-bullseye

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --force-reinstall --no-deps openai==0.28.1

EXPOSE 8501
CMD ["streamlit", "run", "ask_the_archive.py", "--server.port=8501", "--server.address=0.0.0.0"]

