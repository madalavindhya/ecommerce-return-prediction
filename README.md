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

## GitHub repository and CI/CD

Initialize and commit the project locally:

```powershell
git init
git branch -M main
git add .
git commit -m "Build return prediction MLOps project"
```

Create a private GitHub repository using the website, or use GitHub CLI if
installed. The CLI command adds the remote automatically:

```powershell
gh repo create <OWNER>/<REPOSITORY> --private --source . --remote origin
```

If you created the repository in the website, add its remote, then push:

```powershell
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
```

The Actions workflow runs source validation, pytest, and Docker image build for
pushes and pull requests targeting `main`. Only a successful push to `main`
publishes `ghcr.io/<lowercase-owner>/<lowercase-repository>` with `latest` and
`sha-<commit>` tags. The publish job uses the built-in `GITHUB_TOKEN` and
`packages: write`; configure package/repository settings to allow the workflow
to publish if your organization requires it. Pull-request runs never publish.

To demonstrate an automatic run, make a meaningful app/model change, e.g.
improve the API description in `app/main.py`, then:

```powershell
git add app/main.py
git commit -m "Clarify prediction API metadata"
git push origin main
```

Open the repository's **Actions** tab, select the `CI/CD` workflow run, and
inspect each job's logs. When the publish job succeeds, verify the package on
the repository's **Packages** page, then:

```powershell
docker pull ghcr.io/<lowercase-owner>/<lowercase-repository>:latest
docker run --rm -p 8000:8000 ghcr.io/<lowercase-owner>/<lowercase-repository>:latest
Invoke-RestMethod http://127.0.0.1:8000/health
```

Use the prediction example above to exercise `/predict`. This workspace does
not have Docker or GitHub CLI installed and is not authenticated to GitHub, so
image build/publish, remote workflow execution, and pulling the published
image must be completed after installing Docker and authenticating/pushing to
your own GitHub repository. No remote execution is claimed here.

## Assignment checklist

1. **Load/preprocess:** actual CSV loaded; duplicates removed; missing and invalid values handled in fitted imputers; categories normalized and encoded.
2. **Feature engineering:** validated discount retained; delivery delay and zero-safe previous-return frequency generated; identifier excluded.
3. **Train/evaluate:** Logistic Regression and Random Forest evaluated with stratified split, class weights, accuracy, precision, recall, F1, confusion matrices, and reports.
4. **Serve:** FastAPI `/health` and `/predict` with Pydantic validation and Swagger.
5. **Repository-ready project:** source, model, data, dependencies, tests, and configuration included; repository creation requires your GitHub account.
6. **CI validation/testing:** GitHub Actions installs dependencies, compiles source, and runs pytest.
7. **Container/registry:** workflow builds the image and publishes to GHCR only from validated pushes to `main`.
8. **Automatic trigger:** push-to-main trigger and demonstration commands are configured; actual trigger requires a GitHub push.
9. **Verify deployment:** API and tests can be exercised locally; remote run logs and GHCR pull require GitHub/Docker access.
