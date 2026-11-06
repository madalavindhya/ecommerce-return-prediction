# E-Commerce Product Return Prediction

An end-to-end classification project built from the supplied
`Section7_Q4_Ecommerce_Product_Return_CICD.csv`. It trains and compares a
class-weighted Logistic Regression and Random Forest, logs runs locally with
MLflow, serves the selected pipeline through FastAPI, and includes tests,
Docker packaging, and a GitHub Actions workflow for GHCR.

## Project layout

```text
data/
  Section7_Q4_Ecommerce_Product_Return_CICD.csv
  processed/processed_dataset.csv
  processed/preprocessing_info.json
src/
  preprocessing.py
  train.py
  predict.py
app/
  main.py
models/best_model.joblib
reports/evaluation_results.json
tests/
.github/workflows/cicd.yml
Dockerfile
requirements.txt
```

## Setup and train

Use Python 3.11. From the project root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
```

The trainer reports the real dataset shape, missingness, cleaning summary,
engineered values, held-out model comparison, confusion matrix, and MLflow
registration. It writes the processed data and preprocessing information under
`data/processed/`, the best estimator to `models/best_model.joblib`, and metrics
and run identifiers to `reports/evaluation_results.json`. The split is
stratified (80/20, random seed 42); `Order_ID` is excluded. F1 for the
`Returned` class determines the winner, with recall and accuracy as tie-breakers.

## MLflow tracking

Training uses a local SQLite tracking/registry store (`mlflow.db`) by default;
no hosted service is required. Start the dashboard from the project root:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db `
  --registry-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000> and select
`Ecommerce_Product_Return_Prediction`. For a remote tracking server, set
`MLFLOW_TRACKING_URI` before training.

## API

Start the API after training:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Swagger UI: <http://127.0.0.1:8000/docs>

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Example prediction:

```powershell
$body = @{
  Product_Category = "Fashion"
  Product_Price_INR = 2500
  Discount_Percent = 20
  Expected_Delivery_Days = 5
  Actual_Delivery_Days = 7
  Customer_Rating_PreOrder = 3.5
  Previous_Returns = 2
  Previous_Orders = 10
  Payment_Method = "Card"
  Damaged_Package_Flag = "No"
  Package_Size = "Medium"
  Promotion_Active = "Yes"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" -Body $body
```

The response contains the predicted status, probability of `Returned`, and
selected model name. Invalid percentage/rating/range values and unknown fields
receive HTTP 422 responses.

## Tests

```powershell
python -m pytest -q
```

Tests cover loading and preprocessing the supplied CSV, feature engineering
(including zero previous orders), model prediction, response structure, health,
and invalid API requests.

## Docker

Docker must be installed and running:

```powershell
docker build -t ecommerce-return-prediction .
docker run --rm -p 8000:8000 ecommerce-return-prediction
```

The image includes the trained model and API code and runs as an unprivileged
user. Check `/health` and `/docs` on port 8000.
The image installs only the runtime dependencies from `requirements-api.txt`;
MLflow and pytest remain available in the full development `requirements.txt`.
After publishing, the GitHub Actions workflow pulls the commit-tagged image,
starts it, and smoke-tests `/health` and `/predict` on the hosted runner.

## GitHub repository and verified CI/CD

The project is pushed to
<https://github.com/madalavindhya/ecommerce-return-prediction>. The latest
verified project-code commit is `213897f38b18af498148035e34f19b72738ecc68`.
The successful GitHub Actions run is
<https://github.com/madalavindhya/ecommerce-return-prediction/actions/runs/42198106312>.
Both validation/build and publish jobs passed. The publish job pushed
`ghcr.io/madalavindhya/ecommerce-return-prediction` with `latest` and
`sha-213897f38b18af498148035e34f19b72738ecc68` tags, pulled that commit-tagged
image on the GitHub-hosted Docker runner, ran it, and verified both endpoints.

The recorded container responses were:

```json
{"status":"healthy","model_loaded":true}
{"predicted_return_status":"Returned","prediction_probability":0.5022433105390813,"model_name":"LogisticRegression"}
```

GitHub Actions runs tests and builds Docker for pushes and pull requests to
`main`; publishing is restricted to successful pushes to `main` and uses the
built-in `GITHUB_TOKEN`. The publish job also smoke-tests the published
container. For a local pull/run on a machine with Docker installed:

```powershell
docker pull ghcr.io/madalavindhya/ecommerce-return-prediction:latest
docker run --rm -p 8000:8000 ghcr.io/madalavindhya/ecommerce-return-prediction:latest
Invoke-RestMethod http://127.0.0.1:8000/health
```

Docker is not installed on the development PC used for this project; the
published image was instead pulled and run successfully by the GitHub-hosted
workflow runner. See [the final completion report](./reports/final_completion_report.md)
for the nine-task evidence checklist and evaluation results.
