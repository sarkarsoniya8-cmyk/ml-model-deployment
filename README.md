# Deploying a Machine Learning Model

A trained breast-cancer classifier served as a REST API (Flask + gunicorn), packaged in Docker and ready to deploy to Google Cloud Run or AWS.

## What is in this repo
| Deliverable | Where |
|---|---|
| Docker image definition | [`Dockerfile`](Dockerfile) |
| Published Docker image | `ghcr.io/sarkarsoniya8-cmyk/ml-model-deployment:latest` (Docker Hub or ghcr.io, see the guide, section 6) |
| Deployed API | `ADD YOUR SERVICE URL HERE` (see the guide, section 7) |
| Deployment and API documentation | [`docs/deployment_guide.md`](docs/deployment_guide.md) |
| API code | [`app/main.py`](app/main.py), model training in [`train.py`](train.py) |
| Tests | [`tests/test_api.py`](tests/test_api.py) (12 tests) |

## Quick start
```bash
docker build -t breast-cancer-api:1.0 .
docker run --rm -p 8080:8080 ghcr.io/sarkarsoniya8-cmyk/ml-model-deployment:latest
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d @examples/sample_malignant.json
```
Expected response: `{"label": 1, "prediction": "malignant", "probability_malignant": 1.0}`

## Endpoints
`GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch`. Full request and response details are in the [deployment guide](docs/deployment_guide.md).

## Model
StandardScaler + Logistic Regression on the Breast Cancer Wisconsin (Diagnostic) dataset. Test accuracy 0.965, recall 0.929, F1 0.951. Educational demo, not a medical device.
