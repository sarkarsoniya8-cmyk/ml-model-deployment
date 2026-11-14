FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code and train the model inside the image, so the model
# and the scikit-learn version that loads it always match
COPY train.py .
COPY app ./app
RUN python train.py

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

# Cloud Run and similar platforms set PORT; default to 8080
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 60 app.main:app"]
