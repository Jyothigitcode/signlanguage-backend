FROM python:3.10-bullseye

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libx11-6 \
    libx11-xcb1 \
    libxau6 \
    libxdmcp6 \
    libxinerama1 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "image_backend:app", "--host", "0.0.0.0", "--port", "8000"]
