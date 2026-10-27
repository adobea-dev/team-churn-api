# team-churn-api

Ishango.ai Week 1 team project: take a Telecom Churn model from a notebook-style script to a tested, containerized, CI-checked API.

```
Dataset (UCI 563) → Pipeline → FastAPI → Docker → PR → CI → local release (cloud optional)
```

This README lets anyone on the team (or a new reader) recreate the project from an empty folder, in the same order as the course site:

- Course: <https://group-work-ishango.netlify.app/week-01-production-apps/index.html>
- Team setup: <https://group-work-ishango.netlify.app/team-setup>
- Anchor project, Labs 1-3 and the Challenge are linked from the Week 1 page.

## What you build

| Piece | Where it lives | Course step |
|---|---|---|
| Trained pipeline and metadata | `ml_platform/training/churn/` | Anchor project |
| FastAPI service (`api.main:app`) | `api/main.py` | Lab 1 |
| Tests | `production_app/tests/test_api.py` | Lab 1, Challenge |
| Docker image | `Dockerfile`, `entrypoint.sh`, `.dockerignore` | Lab 2 |
| CI and release | `.github/workflows/checks.yml`, git tag `week1-v1` | Lab 3 |
| Hardened API | `api/main.py` (final version below) | Challenge |

**Runtime contract:** always run from the repository root and import `api.main:app`. Do not rename `api/`.

**Dataset:** UCI Iranian Churn, ID 563, 3,150 records, 13 features, binary `Churn` target.

## Repository layout

```
team-churn-api/
├── .github/workflows/checks.yml
├── api/
│   ├── __init__.py
│   └── main.py
├── ml_platform/
│   └── training/churn/
│       ├── train.py
│       └── models/
│           ├── churn_pipeline.joblib
│           ├── feature_names.json
│           └── metrics.json
├── production_app/
│   └── tests/
│       ├── __init__.py
│       └── test_api.py
├── .dockerignore
├── .gitattributes
├── .gitignore
├── Dockerfile
├── entrypoint.sh
├── pyproject.toml
├── uv.lock
└── README.md
```

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Git and a GitHub account
- Docker (Docker Desktop, or a teammate's machine or Codespace)
- Windows: the course examples are bash. Use WSL, or the PowerShell equivalents in the table below.

| bash | PowerShell |
|---|---|
| `mkdir -p a/b` | `New-Item -ItemType Directory -Force -Path a\b` |
| `touch file` | `New-Item -ItemType File -Force -Path file` |
| `cat file` | `Get-Content file` |
| `curl ...` | `curl.exe ...` (or `Invoke-RestMethod`) |
| `export X=1` | `$env:X = "1"` |
| `\` line continuation | put the command on one line |

---

## 1. Team setup

1. Form a team of 3-4. Assign initial owners: **model/data**, **API**, **testing**, **packaging/docs**. Rotate roles. Everyone authors at least one PR and reviews another.
2. One member creates an **empty** GitHub repo named `team-churn-api` (no auto-generated files) and invites the others.
3. Branch rules (enable branch protection if your plan allows, otherwise follow manually):
   - `main` is the integration branch.
   - Every change goes through a PR reviewed by someone else.
   - Never merge with failing checks. Never approve your own PR.

### Create the project (one member, once)

```bash
mkdir team-churn-api && cd team-churn-api
git init -b main

# Minimal pyproject.toml, then dependencies
uv init --bare
uv add fastapi "uvicorn[standard]" pandas scikit-learn joblib ucimlrepo
uv add --dev pytest httpx ruff

# Folders and package markers
mkdir -p api production_app/tests ml_platform/training/churn/models
touch api/__init__.py production_app/__init__.py production_app/tests/__init__.py
```

Append tool settings to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "B"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["production_app/tests"]
```

Set `requires-python = ">=3.12"` in the `[project]` section if `uv init` did not.

`.gitignore`:

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
.env
```

`.gitattributes` (keeps `entrypoint.sh` on LF line endings, required for Docker on Windows):

```
*.sh text eol=lf
entrypoint.sh text eol=lf
```

First commit and push:

```bash
uv sync
git add .
git commit -m "chore: initialise team Week 1 project"
git remote add origin https://github.com/YOUR-TEAM-OWNER/team-churn-api.git
git push -u origin main
```

### Everyone else

```bash
git clone https://github.com/YOUR-TEAM-OWNER/team-churn-api.git
cd team-churn-api
uv sync --locked
```

### Daily PR workflow

```bash
git switch main
git pull --ff-only
git switch -c week1/YOUR-NAME-TASK

# ...work, then:
uv run --locked ruff format .
uv run --locked ruff check .
uv run --locked pytest

git add PATHS-YOU-CHANGED
git commit -m "feat: describe your change"
git push -u origin HEAD
```

Open a PR into `main`, request a review from a teammate, and merge only when checks and review pass.

---

## 2. Anchor project: train the model

Suggested owner: model/data. A second member checks feature names and saved files.

Before modelling, write down what the target means, inspect its distribution, and name three features that could plausibly influence churn.

Create `ml_platform/training/churn/train.py`:

```python
"""Train and save the churn pipeline (UCI Iranian Churn, id=563).

Run from the repo root:
    uv run --locked python ml_platform/training/churn/train.py
"""
import json
from pathlib import Path

import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

ARTIFACT_DIR = Path(__file__).resolve().parent / "models"


def main() -> None:
    dataset = fetch_ucirepo(id=563)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0]

    # Collapse stray double spaces so names match the API mapping exactly
    X.columns = [" ".join(c.split()) for c in X.columns]

    print("Shape:", X.shape)
    print("Features:", X.columns.tolist())
    print("Target distribution:\n", y.value_counts(normalize=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, predictions, output_dict=True)
    auc = roc_auc_score(y_test, probabilities)

    print(classification_report(y_test, predictions))
    print("ROC-AUC:", round(auc, 4))
    print("Sample probabilities:", probabilities[:5].round(3))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACT_DIR / "churn_pipeline.joblib")
    (ARTIFACT_DIR / "feature_names.json").write_text(
        json.dumps(X.columns.tolist(), indent=2)
    )
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "roc_auc": round(auc, 4),
                "churn_precision": round(report["1"]["precision"], 4),
                "churn_recall": round(report["1"]["recall"], 4),
                "churn_f1": round(report["1"]["f1-score"], 4),
                "n_train": len(X_train),
                "n_test": len(X_test),
            },
            indent=2,
        )
    )
    print(f"Saved artifacts to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
```

Run it:

```bash
uv run --locked python ml_platform/training/churn/train.py
```

Checks before moving on:

- [ ] `models/` contains `churn_pipeline.joblib`, `feature_names.json`, `metrics.json`
- [ ] `feature_names.json` lists the 13 features with single spaces (`Call Failure`, `Complains`, `Subscription Length`, `Charge Amount`, `Seconds of Use`, `Frequency of use`, `Frequency of SMS`, `Distinct Called Numbers`, `Age Group`, `Tariff Plan`, `Status`, `Age`, `Customer Value`)
- [ ] Precision, recall and F1 for the churn class reviewed, not just accuracy
- [ ] The team can explain what a false negative costs the telecom (a churner you missed)

Commit the small trusted artifacts so CI and fresh clones can run the API. Never load `.joblib` files from untrusted sources.

**Why save the whole pipeline?** Preprocessing and model travel together, which avoids training-serving mismatch.

---

## 3. Lab 1: the FastAPI service

Suggested owners: API member implements, model member reviews the feature mapping, testing member writes tests.

Create `api/main.py`:

```python
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = APP_DIR / "ml_platform" / "training" / "churn" / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
FEATURES_PATH = MODELS_DIR / "feature_names.json"

app = FastAPI(
    title="Ishango Telecom Churn API",
    version="1.0.0",
    description="Week 1 production API for telecom churn prediction.",
)


class PredictionRequest(BaseModel):
    call_failure: int = Field(ge=0)
    complains: int = Field(ge=0, le=1)
    subscription_length: int = Field(ge=0)
    charge_amount: int = Field(ge=0)
    seconds_of_use: int = Field(ge=0)
    frequency_of_use: int = Field(ge=0)
    frequency_of_sms: int = Field(ge=0)
    distinct_called_numbers: int = Field(ge=0)
    age_group: int = Field(ge=1)
    tariff_plan: int = Field(ge=1)
    status: int = Field(ge=1)
    age: int = Field(ge=0)
    customer_value: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    churn_probability: float
    model_version: str


if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model artifact not found at {MODEL_PATH}. "
        "Run the Week 1 anchor training step first."
    )
if not FEATURES_PATH.exists():
    raise RuntimeError(f"Feature metadata not found at {FEATURES_PATH}.")

model = joblib.load(MODEL_PATH)
feature_names = json.loads(FEATURES_PATH.read_text())


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "telecom-churn-api"}


@app.post("/predictions", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    api_values = {
        "Call Failure": payload.call_failure,
        "Complains": payload.complains,
        "Subscription Length": payload.subscription_length,
        "Charge Amount": payload.charge_amount,
        "Seconds of Use": payload.seconds_of_use,
        "Frequency of use": payload.frequency_of_use,
        "Frequency of SMS": payload.frequency_of_sms,
        "Distinct Called Numbers": payload.distinct_called_numbers,
        "Age Group": payload.age_group,
        "Tariff Plan": payload.tariff_plan,
        "Status": payload.status,
        "Age": payload.age,
        "Customer Value": payload.customer_value,
    }

    try:
        row = pd.DataFrame(
            [[api_values[name] for name in feature_names]],
            columns=feature_names,
        )
        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1])
        return PredictionResponse(
            prediction=prediction,
            churn_probability=round(probability, 4),
            model_version="week1-v1",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc
```

Design notes:

- `Path(__file__).resolve().parents[1]` is the repo root, so no username or machine path is hard-coded.
- The app refuses to start if the model or feature file is missing, instead of pretending to be healthy.
- The API uses snake_case fields and maps them to the training column names deliberately.

Run it:

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8080
```

Test, health first:

```bash
curl http://localhost:8080/health
```

Then open <http://localhost:8080/docs> and send a valid prediction:

```bash
curl -X POST http://localhost:8080/predictions \
  -H "Content-Type: application/json" \
  -d '{"call_failure": 8, "complains": 1, "subscription_length": 20, "charge_amount": 3, "seconds_of_use": 2500, "frequency_of_use": 55, "frequency_of_sms": 12, "distinct_called_numbers": 25, "age_group": 3, "tariff_plan": 1, "status": 1, "age": 42, "customer_value": 1200.5}'
```

PowerShell version:

```powershell
$body = '{"call_failure": 8, "complains": 1, "subscription_length": 20, "charge_amount": 3, "seconds_of_use": 2500, "frequency_of_use": 55, "frequency_of_sms": 12, "distinct_called_numbers": 25, "age_group": 3, "tariff_plan": 1, "status": 1, "age": 42, "customer_value": 1200.5}'
Invoke-RestMethod -Uri http://localhost:8080/predictions -Method Post -ContentType "application/json" -Body $body
```

Expected shape (values depend on your trained artifact):

```json
{ "prediction": 0, "churn_probability": 0.2741, "model_version": "week1-v1" }
```

Then deliberately send an invalid request (`{"call_failure": -9, "complains": 7, "age": -20}`). It must return HTTP 422 and never reach the model.

### Tests

`production_app/tests/test_api.py`:

```python
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invalid_prediction_is_rejected() -> None:
    response = client.post("/predictions", json={"call_failure": -1})

    assert response.status_code == 422
```

Quality checks (run from the repo root):

```bash
uv run --locked ruff format .
uv run --locked ruff check .
uv run --locked pytest
```

Note: the API loads the model at import time, so tests need the committed artifact.

**Lab 1 done when:**

- [ ] `api/main.py` exposes `app`
- [ ] `GET /health` returns 200
- [ ] Valid `POST /predictions` returns prediction, probability and model version
- [ ] Invalid input is rejected with 422 before inference
- [ ] `/docs` shows both endpoints
- [ ] Tests, Ruff and pytest pass

---

## 4. Lab 2: Docker

Suggested owner: packaging member. Rotate the driver so everyone practises. Use a machine or Codespace with a working Docker engine (`docker --version` and `docker info` must both succeed).

`Dockerfile` (repo root):

```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
COPY api ./api
COPY ml_platform/training/churn/models ./ml_platform/training/churn/models
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x /app/entrypoint.sh
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
```

`entrypoint.sh` (must use LF line endings):

```sh
#!/bin/sh
set -eu
cd /app
exec /app/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8080}"
```

`.dockerignore`:

```
.git
.github
.venv
__pycache__
*.pyc
.pytest_cache
.ruff_cache
.env
```

Do **not** ignore `ml_platform/training/churn/models`, because the model travels inside the image.

Build and run:

```bash
docker build -t team-churn:v1 .
docker run --rm --name ishango-week1-api -p 8080:8080 team-churn:v1
```

From a second terminal, repeat the Lab 1 `/health` and prediction requests.

Learning exercises (explain each result):

1. **Missing image:** `docker run --rm --pull=never team-churn:does-not-exist` fails before the API starts. It is not a model problem.
2. **Inspect:** `docker ps`, `docker logs ishango-week1-api`, `docker inspect ishango-week1-api`.
3. **Artifact is inside:** `docker exec ishango-week1-api ls -lh /app/ml_platform/training/churn/models/`
4. **Wrong entry point:** `docker stop ishango-week1-api`, then `docker run --rm --entrypoint /app/.venv/bin/uvicorn team-churn:v1 missing.main:app`. This is an import error, different from a missing artifact.
5. **Port mapping:** `docker run --rm --name ishango-week1-port-test -p 9999:8080 team-churn:v1`. `localhost:8080` fails, `localhost:9999` works. The mapping is host:container.
6. **Immutability:** add `"version": "v2"` to `/health`, confirm the running `v1` image is unchanged, then `docker build -t team-churn:v2 .` and run `v2`.

Compare images with `docker images team-churn`.

**Lab 2 done when:**

- [ ] Image builds from the repo root
- [ ] Container serves `/health` and predictions
- [ ] Model file proven to exist inside the container
- [ ] Wrong-entrypoint and port-mapping exercises explained
- [ ] `v1` and `v2` compared
- [ ] Ruff and pytest still pass

---

## 5. Lab 3: PR, CI and local release

Each member authors at least one meaningful PR and reviews another.

### CI workflow

`.github/workflows/checks.yml` (commit through a reviewed setup PR). It runs checks only and does not deploy:

```yaml
name: Team quality checks
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
permissions:
  contents: read
jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          version: "0.11.2"
          python-version: "3.12"
      - run: uv sync --locked
      - run: uv run --locked ruff check .
      - run: uv run --locked pytest
```

### Ship it

```bash
git branch --show-current
git diff --name-only origin/main...HEAD

git add api ml_platform production_app pyproject.toml uv.lock Dockerfile entrypoint.sh .dockerignore .gitignore .gitattributes
git commit -m "feat: package team churn service"
git push -u origin HEAD
```

Open a PR into `main` describing the change, how you tested it, and any model or contract impact. The reviewer reads the diff, checks the API contract, tests locally where possible, and asks about the artifact.

If CI fails, find the first real error and classify it: lockfile mismatch, missing artifact, lint, test, or runner/network. Reproduce locally and push the fix to the same PR. Never disable checks to get green.

**Merging only updates `main`. It does not deploy anything.**

### Fresh-clone check (a different teammate)

```bash
git clone https://github.com/YOUR-TEAM-OWNER/team-churn-api.git
cd team-churn-api
uv sync --locked
uv run --locked pytest
```

### Release image and verification

```bash
git rev-parse HEAD
docker build -t team-churn:week1-v1 .
docker run --rm --name team-churn-release -p 8080:8080 team-churn:week1-v1

curl http://localhost:8080/health
curl http://localhost:8080/openapi.json
docker logs team-churn-release
```

Repeat the valid request (expect prediction, probability and version) and the invalid request (expect 422). Record the commit SHA with the image tag.

### Tag

```bash
git switch main
git pull --ff-only
git tag week1-v1
git push origin week1-v1
```

### Optional: cloud host

Google Cloud Run, another container platform, or a VM all work. Check the provider's current docs, pricing and account requirements first. Bind to `0.0.0.0`, pass `PORT`, and use `/health`. Never commit credentials. A successful local release fully satisfies the required route.

### Diagnosing delivery failures

| Symptom | Investigate |
|---|---|
| Green CI, failing image build | Dockerfile, build context, model packaging |
| Container exits | Entrypoint, import path, missing artifact |
| Running container, unreachable port | Host/container mapping, bind address |
| Health works, prediction fails | Schema mapping, artifact compatibility, inference |
| Optional cloud push denied | Registry authentication and permissions |

---

## 6. Challenge: harden the API

Split these into issues, one PR each, each reviewed by another member:

1. Stronger request validation with field descriptions
2. Explicit response model including `request_id`
3. Request ID per prediction (`uuid4`)
4. Application logging (no `print`, no full customer records)
5. Health endpoint with `model_version` and `api_version`
6. Controlled 500 on inference failure, with a tight `try` boundary
7. Tests for success and failure paths
8. One stretch improvement of your own (for example `/version`, request-ID header, or inference timing)

Reference implementation: replace `api/main.py` with the following (includes the missing-artifact startup check as the stretch item).

```python
import json
import logging
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"
MODEL_VERSION = "week1-v1"

APP_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = APP_DIR / "ml_platform" / "training" / "churn" / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
FEATURES_PATH = MODELS_DIR / "feature_names.json"

FIELD_TO_FEATURE = {
    "call_failure": "Call Failure",
    "complains": "Complains",
    "subscription_length": "Subscription Length",
    "charge_amount": "Charge Amount",
    "seconds_of_use": "Seconds of Use",
    "frequency_of_use": "Frequency of use",
    "frequency_of_sms": "Frequency of SMS",
    "distinct_called_numbers": "Distinct Called Numbers",
    "age_group": "Age Group",
    "tariff_plan": "Tariff Plan",
    "status": "Status",
    "age": "Age",
    "customer_value": "Customer Value",
}

for path in (MODEL_PATH, FEATURES_PATH):
    if not path.exists():
        raise RuntimeError(
            f"Required artifact missing: {path}. "
            "Run ml_platform/training/churn/train.py first."
        )

model = joblib.load(MODEL_PATH)
feature_names = json.loads(FEATURES_PATH.read_text())
logger.info("model_loaded path=%s model_version=%s", MODEL_PATH, MODEL_VERSION)

app = FastAPI(
    title="Ishango Telecom Churn API",
    version=API_VERSION,
    description="Week 1 production API for telecom churn prediction.",
)


class ChurnRequest(BaseModel):
    call_failure: int = Field(ge=0, description="Number of failed calls")
    complains: int = Field(ge=0, le=1, description="Customer complaint flag (0/1)")
    subscription_length: int = Field(ge=0, description="Months subscribed")
    charge_amount: float = Field(ge=0, description="Charge amount band")
    seconds_of_use: float = Field(ge=0, description="Total seconds of calls")
    frequency_of_use: float = Field(ge=0, description="Total number of calls")
    frequency_of_sms: float = Field(ge=0, description="Total number of SMS")
    distinct_called_numbers: float = Field(ge=0, description="Distinct numbers called")
    age_group: int = Field(ge=0, description="Age group code")
    tariff_plan: int = Field(ge=0, description="Tariff plan code")
    status: int = Field(ge=0, description="Account status code")
    age: int = Field(ge=0, description="Customer age")
    customer_value: float = Field(ge=0, description="Calculated customer value")


class PredictionResponse(BaseModel):
    prediction: int
    churn_probability: float = Field(ge=0, le=1)
    model_version: str
    request_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "telecom-churn-api",
        "model_version": MODEL_VERSION,
        "api_version": API_VERSION,
    }


@app.post("/predictions", response_model=PredictionResponse)
def predict(payload: ChurnRequest) -> PredictionResponse:
    request_id = str(uuid4())

    values = {
        FIELD_TO_FEATURE[field]: value
        for field, value in payload.model_dump().items()
    }
    row = pd.DataFrame(
        [[values[name] for name in feature_names]],
        columns=feature_names,
    )

    try:
        prediction = int(model.predict(row)[0])
        churn_probability = float(model.predict_proba(row)[0][1])
    except Exception as exc:
        logger.exception("prediction_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction service failed to process the request.",
        ) from exc

    logger.info(
        "prediction_completed request_id=%s prediction=%s model_version=%s",
        request_id,
        prediction,
        MODEL_VERSION,
    )
    return PredictionResponse(
        prediction=prediction,
        churn_probability=round(churn_probability, 4),
        model_version=MODEL_VERSION,
        request_id=request_id,
    )
```

Updated `production_app/tests/test_api.py`:

```python
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "call_failure": 8,
    "complains": 1,
    "subscription_length": 20,
    "charge_amount": 3,
    "seconds_of_use": 2500,
    "frequency_of_use": 55,
    "frequency_of_sms": 12,
    "distinct_called_numbers": 25,
    "age_group": 3,
    "tariff_plan": 1,
    "status": 1,
    "age": 42,
    "customer_value": 1200.5,
}


def test_health_has_version_metadata() -> None:
    response = client.get("/health")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert "model_version" in body
    assert "api_version" in body


def test_valid_prediction_contract() -> None:
    response = client.post("/predictions", json=VALID_PAYLOAD)
    body = response.json()

    assert response.status_code == 200
    assert body["prediction"] in (0, 1)
    assert body["model_version"]
    assert body["request_id"]


def test_probability_is_between_zero_and_one() -> None:
    body = client.post("/predictions", json=VALID_PAYLOAD).json()

    assert 0 <= body["churn_probability"] <= 1


def test_negative_input_is_rejected() -> None:
    payload = {**VALID_PAYLOAD, "call_failure": -1}
    response = client.post("/predictions", json=payload)

    assert response.status_code == 422


def test_missing_required_field_is_rejected() -> None:
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    response = client.post("/predictions", json=payload)

    assert response.status_code == 422
```

Then re-test inside Docker and release through the same PR, CI and tag path:

```bash
docker build -t week1-churn-api:challenge .
docker run --rm -p 8080:8080 week1-churn-api:challenge
```

Your PR description should cover: reliability improvements, tests added, how you tested locally and in Docker, and one invalid request with its resulting behaviour.

Rubric: **Contract**, **Reliability**, **Observability**, **Delivery**. The target is "Production-ready Week 1"; the stretch item earns "Excellent".

---

## Hand-in checklist

Submit the repository URL and release commit or tag, plus:

- [ ] Model evaluation (precision, recall, F1, ROC-AUC from `metrics.json`)
- [ ] Valid and invalid API request evidence
- [ ] Passing tests and CI run
- [ ] Docker build and run evidence, with container logs
- [ ] Fresh-clone check performed by a different member
- [ ] Contribution table (below)
- [ ] Facilitator added as collaborator if the repo is private
- [ ] Week 1 quiz completed

| Member | Authored PR | Reviewed PR | Demonstrated responsibility |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

Team demo: each member explains a part of the workflow and answers one question outside their initial role. Show the API contract, artifact location, a valid prediction, a rejected request, test evidence and container logs.

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv` not found | Install uv, then restart the terminal |
| `ModuleNotFoundError: api` | Run from the repo root, where `api/` is importable |
| Model artifact missing | Run `train.py` and commit the three files in `models/` |
| DataFrame column error | Compare `feature_names.json` with the mapping in `api/main.py` |
| 422 on a valid-looking request | Read the response body; it names the failing field |
| 500 on prediction | Check Uvicorn logs; do not expose tracebacks to clients |
| `entrypoint.sh` fails in Docker on Windows | Ensure LF endings via `.gitattributes`, then re-checkout the file |
| CI lockfile error | Run `uv lock` locally and commit `uv.lock` in a reviewed PR |
| Ruff failure | `uv run ruff format .` then `uv run ruff check .` and fix the reported lines |
| Docker unavailable | Report it and use a teammate's machine or Codespace for the demo |

## Reference

- [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- [FastAPI and Docker](https://fastapi.tiangolo.com/deployment/docker/)
- [GitHub pull requests](https://docs.github.com/en/pull-requests/get-started/pull-request-quickstart)
- Dataset: UCI Iranian Churn, DOI 10.24432/C5JW3Z