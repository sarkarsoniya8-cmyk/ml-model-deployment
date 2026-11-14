# Deployment Guide: Breast Cancer Classification API

This project deploys a trained machine learning model as a web API, packaged in a Docker container so it runs the same way on a laptop and in the cloud.

> **Educational demo.** The model classifies tumours as malignant or benign from 30 cell-nucleus measurements. It is not a medical device and must not be used for real diagnosis.

## 1. Overview

| Part | Choice |
|---|---|
| Model | StandardScaler + Logistic Regression (C=1), the final model from the model-selection task |
| Dataset | Breast Cancer Wisconsin (Diagnostic), built into scikit-learn |
| Test performance | accuracy 0.965, precision 0.975, recall 0.929, F1 0.951, ROC AUC 0.996 (114 held-out patients) |
| API framework | Flask, served by gunicorn |
| Packaging | Docker (`python:3.12-slim`, non-root user) |
| Cloud target | Google Cloud Run (AWS option described in section 8) |

**How it fits together:** `train.py` trains the model and writes `model/model.joblib`. The Dockerfile runs `train.py` while building the image, so the model and the scikit-learn version that loads it always match. When the container starts, gunicorn runs `app/main.py`, which loads the model and answers HTTP requests on the port given by the `PORT` environment variable (default 8080).

## 2. Project structure
```
Dockerfile
requirements.txt
train.py                     trains and saves the model
app/main.py                  the API
tests/test_api.py            automated tests (12)
examples/                    sample request bodies (real dataset rows)
scripts/smoke_test.py        checks a running API, local or cloud
docs/deployment_guide.md     this document
.github/workflows/docker-publish.yml   builds and publishes the image on GitHub
```

## 3. API reference
Base URL: `http://localhost:8080` locally, or your cloud service URL. All bodies are JSON.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service description |
| GET | `/health` | Liveness check |
| GET | `/model-info` | Model description, the 30 feature names, test metrics |
| POST | `/predict` | One prediction |
| POST | `/predict/batch` | Up to 1000 predictions |

### POST /predict
Send either named features or a list of 30 numbers in the order returned by `/model-info`.
```json
{"features": {"mean radius": 17.99, "mean texture": 10.38, "...": "all 30 features"}}
```
or
```json
{"features": [17.99, 10.38, 122.8, "...", "30 numbers"]}
```
A complete example is in `examples/sample_malignant.json`.

Response (HTTP 200), from a real run:
```json
{"label": 1, "prediction": "malignant", "probability_malignant": 1.0}
```
`label` is 1 for malignant and 0 for benign. `probability_malignant` is the model's probability for the malignant class; a prediction is malignant when it is 0.5 or higher.

### POST /predict/batch
```json
{"instances": [ {"mean radius": 17.99, "...": "..."}, [13.54, 14.36, "..."] ]}
```
Each instance can be a named object or a list. Example response, from a real run:
```json
{"count": 2, "predictions": [
  {"label": 1, "prediction": "malignant", "probability_malignant": 1.0},
  {"label": 0, "prediction": "benign", "probability_malignant": 0.0022}]}
```

### Errors
Invalid input returns HTTP 400 with a JSON message; unknown paths return 404 and wrong methods 405.
```json
{"error": "features: expected 30 values, got 3"}
```
Rejected inputs: missing or unknown feature names, a list of the wrong length, non-numeric or non-finite values, a body that is not a JSON object, and an empty or oversized batch.

### Example calls
```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_malignant.json
```
```python
import json, requests
body = json.load(open("examples/sample_benign.json"))
print(requests.post("http://localhost:8080/predict", json=body).json())
```

## 4. Run locally without Docker
```bash
pip install -r requirements.txt
python train.py
python -m unittest discover -s tests -v      # 12 tests
python -m app.main                           # development server on port 8080
python scripts/smoke_test.py http://localhost:8080
```

## 5. Build and run the Docker image
Install Docker Desktop (Windows/Mac) or Docker Engine (Linux), then from the project folder:
```bash
docker build -t breast-cancer-api:1.0 .
docker run --rm -p 8080:8080 breast-cancer-api:1.0
```
In a second terminal:
```bash
python scripts/smoke_test.py http://localhost:8080
```
All five checks should print `[PASS]`.

## 6. Publish the Docker image (the image deliverable)
**Option A: Docker Hub**
```bash
docker login
docker tag breast-cancer-api:1.0 YOUR_DOCKERHUB_USERNAME/breast-cancer-api:1.0
docker push YOUR_DOCKERHUB_USERNAME/breast-cancer-api:1.0
```
The image is then public at `https://hub.docker.com/r/YOUR_DOCKERHUB_USERNAME/breast-cancer-api` and can be run anywhere with `docker run -p 8080:8080 YOUR_DOCKERHUB_USERNAME/breast-cancer-api:1.0`.

**Option B: GitHub Container Registry, without installing Docker**
The workflow in `.github/workflows/docker-publish.yml` builds the image on GitHub and publishes it to `ghcr.io/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME`.
1. Push the repo to GitHub on the `main` branch. The workflow starts automatically (see the **Actions** tab); you can also start it with **Run workflow**.
2. When it succeeds, the image appears under your profile's **Packages**.
3. New packages are private by default. Open the package, then **Package settings**, and change the visibility to **Public** so reviewers can pull it.
4. Pull and run it with `docker run -p 8080:8080 ghcr.io/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME:latest`.

Put the image link in your README.

## 7. Deploy to Google Cloud Run
Cloud Run runs a container and gives it a public HTTPS URL. It sets the `PORT` variable, which the Dockerfile already respects.

**Prerequisites:** a Google Cloud account, a project with **billing enabled**, and the [gcloud CLI](https://cloud.google.com/sdk/docs/install). Check Google's current pricing and free-credit terms before you start.

```bash
gcloud auth login
gcloud config set project PROJECT_ID

# turn on the required services
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# let Cloud Build build from source (PROJECT_NUMBER is on the project dashboard)
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com \
  --role=roles/run.builder
```
The role can take a couple of minutes to take effect. Then, from the project folder:
```bash
gcloud run deploy breast-cancer-api --source . --region REGION --allow-unauthenticated
```
- `--source .` makes Cloud Build build the image from the Dockerfile and deploy it.
- Choose a `REGION` near you from the list in the Cloud Run documentation.
- `--allow-unauthenticated` makes the URL public so reviewers can try it. Leave it out to keep the service private.
- If asked to create an Artifact Registry repository, answer `y`.

When it finishes, gcloud prints the **Service URL**. Test it:
```bash
python scripts/smoke_test.py https://YOUR-SERVICE-URL
```

**Clean up** so you are not charged:
```bash
gcloud run services delete breast-cancer-api --region REGION
```
The built image can still incur small storage charges in Artifact Registry; delete the repository or the whole project when you are finished.

## 8. AWS option
AWS App Runner is closed to new customers (since 30 April 2026). For new projects AWS recommends Amazon ECS Express Mode, which runs a container on Fargate behind a load balancer. The outline:

1. Install and configure the AWS CLI (`aws configure`).
2. Create a container repository and push the image:
   ```bash
   aws ecr create-repository --repository-name breast-cancer-api --region REGION
   aws ecr get-login-password --region REGION | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com
   docker tag breast-cancer-api:1.0 ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/breast-cancer-api:1.0
   docker push ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/breast-cancer-api:1.0
   ```
3. In the Amazon ECS console, create a service with **Express Mode**, choose the ECR image, and set the container port to 8080. Follow the current AWS guide, because the console steps change.
4. Test the public URL with `scripts/smoke_test.py`, then delete the service when finished.

## 9. Evidence to include with your submission
- The link to the published Docker image (Docker Hub or ghcr.io)
- The deployed service URL (if you deploy; Cloud Run or AWS)
- A screenshot or terminal output of `scripts/smoke_test.py` against the deployed URL
- A screenshot of the Cloud Run (or ECS) service page showing it running

## 10. Security and limitations
- The public endpoint has no authentication, rate limiting or request logging. Fine for a demo; a real service needs authenticated access and monitoring.
- The container runs as a non-root user, and inputs are validated (type, finiteness, count, batch size) before reaching the model.
- The model was trained on one small dataset and has not been clinically validated.
- The model is trained during the image build, so rebuilding the image retrains it with the same random seed and gives the same model.
- A new container version is not tested automatically on deploy; run the smoke test after every deployment.

## 11. Troubleshooting
| Problem | Likely cause and fix |
|---|---|
| `docker: command not found` | Docker is not installed or not running; start Docker Desktop. |
| `port is already allocated` | Another program uses 8080; use `-p 9090:8080` and call port 9090. |
| Cloud Run says the container failed to start | Check the logs in the Cloud Run console; the app must listen on `$PORT`, which the Dockerfile already does. |
| Build fails with a permissions error on Cloud Run | The `roles/run.builder` role is missing or has not propagated yet; wait a few minutes and retry. |
| `400` response from `/predict` | Read the `error` field; usually a missing feature name or the wrong number of values. |
| ghcr.io image cannot be pulled | The package is still private; make it public in package settings. |
