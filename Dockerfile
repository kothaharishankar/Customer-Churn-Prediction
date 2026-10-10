FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by XGBoost on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
# Train the model at build time (fails the build if training fails, instead of silently using a dummy model)
RUN python -m app.services.train_model --data-path data/train.csv --output app/models/churn_pipeline.pkl

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]