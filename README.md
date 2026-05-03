# Disease Prediction Project - PIMA Diabetes

This project predicts diabetes risk using the PIMA Indians Diabetes dataset and an XGBoost classifier.

## Files
- `disease_prediction_notebook.ipynb`: EDA, preprocessing, experiments, tuning, and evaluation notebook.
- `experiment_results.csv`: Comparison of model experiments.
- `xgb_diabetes_model.joblib`: Final tuned XGBoost model.
- `scaler.joblib`: StandardScaler used before inference.
- `model_metadata.json`: Feature order, medians, tuning parameters, and metrics.
- `api.py`: FastAPI serving endpoint.
- `requirements.txt`: Required Python packages.
- `final_report.md`: Final documentation and ethical analysis.

## Run API
```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

Then open:
```text
http://127.0.0.1:8000/docs
```

## Example request
```json
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 120,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

## Final test metrics
{
  "accuracy": 0.7403,
  "precision": 0.6,
  "recall": 0.7778,
  "f1_score": 0.6774,
  "auc_roc": 0.8211,
  "tn": 72,
  "fp": 28,
  "fn": 12,
  "tp": 42
}
