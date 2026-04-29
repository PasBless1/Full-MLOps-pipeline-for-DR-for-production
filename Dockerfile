FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install -e .
EXPOSE 8080
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
