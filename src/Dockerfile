FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY models ./models
EXPOSE 5000
CMD ["bash", "-lc", "gunicorn -w 2 -b 0.0.0.0:5000 src.api:app"]
