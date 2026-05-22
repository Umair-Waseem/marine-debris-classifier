FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app.py /app/app.py
COPY src /app/src
COPY app /app/app
COPY assets /app/assets
COPY outputs/models /app/outputs/models
COPY outputs/reports /app/outputs/reports
COPY outputs/plots /app/outputs/plots
COPY .streamlit /app/.streamlit

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "7860", "--server.headless", "true", "--server.fileWatcherType", "none"]
