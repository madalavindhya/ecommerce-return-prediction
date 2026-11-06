# Final Project Completion and Verification Report

**Project:** E-Commerce Product Return Prediction with MLflow and CI/CD  
**Repository:** <https://github.com/madalavindhya/ecommerce-return-prediction>  
**Verification date:** 2026-11-06  
**Verified project-code commit:** `213897f38b18af498148035e34f19b72738ecc68`

## Final verification status

- **Latest verified GitHub Actions run:** [Run #3](https://github.com/madalavindhya/ecommerce-return-prediction/actions/runs/42198106312) — completed successfully.
- **CI jobs:** `validate-test-build` passed (dependency install, Python compilation, pytest, Docker build); `publish` passed (GHCR login, image publish, image pull, and container endpoint smoke tests).
- **GitHub source:** the local `main` HEAD and remote `origin/main` matched the verified commit SHA when checked.
- **GHCR:** the commit-tagged image and `latest` were published as `ghcr.io/madalavindhya/ecommerce-return-prediction`.
- **Docker pull/run:** performed successfully on the GitHub-hosted runner by the `Pull and smoke-test published image` step. Docker CLI is not installed on the development PC, so no local-machine pull is claimed.

### Published-container evidence

The workflow log confirms it downloaded:

```text
ghcr.io/madalavindhya/ecommerce-return-prediction:sha-213897f38b18af498148035e34f19b72738ecc68
```

It started the image and tested `/health` and `/predict`. Actual responses:

```json
{"status":"healthy","model_loaded":true}
{"predicted_return_status":"Returned","prediction_probability":0.5022433105390813,"model_name":"LogisticRegression"}
```

### Dataset and model results

- Supplied CSV shape: **1,200 rows × 14 columns**.
- Target counts: **929 Not Returned**, **271 Returned**.
- Missing values in the source: 36 each for `Product_Price_INR`, `Discount_Percent`, `Actual_Delivery_Days`, `Customer_Rating_PreOrder`, and `Payment_Method`.
- No duplicate records were found. `Order_ID` is excluded from model features.
- Processed export: **1,200 rows × 26 columns**, with no missing values.
- Split: stratified 80/20; **960 training rows**, **240 test rows**; seed 42.
- Best model: Logistic Regression, selected by Returned-class F1, then recall and accuracy.

| Model | Accuracy | Precision (Returned) | Recall (Returned) | F1 (Returned) | Confusion matrix* |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 0.6083 | 0.2727 | 0.4444 | 0.3380 | `[[122, 64], [30, 24]]` |
| Random Forest | 0.7625 | 0.2857 | 0.0370 | 0.0656 | `[[181, 5], [52, 2]]` |

*Matrix order: actual rows and predicted columns, labels `Not Returned`, `Returned`.

The MLflow experiment `Ecommerce_Product_Return_Prediction` contains both
training runs. The best model is registered as
`EcommerceProductReturnBestModel`, version 3.

## Evidence for all nine assignment requirements

| # | Assignment requirement | Implementation and actual evidence |
|---:|---|---|
| 1 | Load and preprocess examiner dataset | Loaded the supplied 1,200 × 14 CSV. Cleaned category/target labels, removed duplicates, identified missing and invalid numeric entries, and applied train-fitted imputers. Evidence: `data/processed/preprocessing_info.json`; full train/test workflow passed. |
| 2 | Engineer return-related features | Created `delivery_delay` and zero-safe `customer_return_frequency`; retained/range-validated `Discount_Percent`; excluded `Order_ID`. Evidence: `src/preprocessing.py`, processed dataset, and preprocessing report. |
| 3 | Train and evaluate at least two classifiers | Trained class-weighted Logistic Regression and Random Forest with a stratified split. Saved accuracy, precision, recall, F1, confusion matrices, and classification reports. Evidence and actual metrics are in `reports/evaluation_results.json`. |
| 4 | Serve predictions with FastAPI | Implemented validated `/health` and `/predict`, Swagger, predicted class and probability. Live endpoint responses are recorded above; hosted-container smoke tests passed. |
| 5 | Create repository with code, model, dependencies, configuration | Repository exists at the URL above; dataset, source, serialized model, reports, tests, Docker files, and workflow are on `main`. Local/remote SHA was verified. |
| 6 | Automate dependency install, code validation, and tests | Latest Actions `validate-test-build` job passed dependency install, `compileall`, and pytest. Local pytest result: **6 passed**. |
| 7 | Build and publish Docker image to GHCR | Latest `validate-test-build` Docker build passed. Latest `publish` job logged in with `GITHUB_TOKEN`, published `latest` and the commit SHA tag, and completed successfully. |
| 8 | Demonstrate workflow automatic trigger | Pushes to `main` triggered workflow run #3 automatically. Run details and commit SHA are linked above. |
| 9 | Verify workflow, image, and updated app | Both latest workflow jobs passed. The publish job pulled and ran the commit-tagged GHCR image and tested both `/health` and `/predict`; actual outputs are recorded above. |

## Deliverables

- Source and setup instructions: [README](../README.md)
- Evaluation metrics and MLflow run IDs: [evaluation_results.json](./evaluation_results.json)
- Fully processed, encoded dataset: [processed_dataset.csv](../data/processed/processed_dataset.csv)
- Fitted model: [best_model.joblib](../models/best_model.joblib)
- GitHub Actions workflow: [cicd.yml](../.github/workflows/cicd.yml)
- GHCR package: <https://github.com/madalavindhya/ecommerce-return-prediction/pkgs/container/ecommerce-return-prediction>

**Completion:** all nine assignment tasks have implementation and evidence.
The CI-hosted Docker pull/run was verified. A separate pull on the development
PC was not possible because Docker is not installed there.
