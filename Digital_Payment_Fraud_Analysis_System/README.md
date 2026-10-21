# Digital Payment Fraud Pattern Analysis — ML + Rules

This is the updated **five-page** Streamlit application and simplified classroom-friendly Colab notebook.

## Pages
1. Overview Dashboard — dynamically counts the active uploaded dataset, risk categories, alerts.
2. Dataset Analysis — CSV upload, preview, EDA and data quality.
3. Fraud Detection — **ML-Based**, **Rule-Based**, **Comparison** tabs (still one page).
4. Fraud Review Center — notes and decisions stored in SQLite.
5. History & Reports — saved analysis runs, review history, CSV exports.

## Train model in Colab
1. Upload `Digital_Payment_Fraud_Pattern_Analysis_Spark.ipynb` to Google Colab.
2. Run the cells in order and upload a **labelled** training CSV when asked. Required training fields: `trans_id`, `sender_id`, `receiver_id`, `payment_type`, `amount`, `oldbalance_sender`, `newbalance_sender`, `trans_hour`, `is_fraud` (0/1).
3. Export `fraud_model_artifacts.zip` in the last cell.
4. Extract its `models/fraud_model.joblib` to the project's `models/` folder. Copy `data/feature_columns.json`, `data/model_metrics.json`, and `data/risk_rules.json` into the project's `data/` folder.
5. **Do not use `model_metrics.json` training counts as live dashboard counts.** The app calculates dashboard totals from the selected CSV.

## Run Streamlit from VS Code (Windows)
```powershell
py -3.14 -m pip install -r requirements.txt
py -3.14 -m streamlit run app.py
```
If dependencies fail on Python 3.14, use Python 3.12 or 3.13 instead.

Upload any compatible CSV on **Dataset Analysis**. Required prediction fields: `payment_type`, `amount`, `oldbalance_sender`, `newbalance_sender`, `trans_hour`. Optional for rule detection: `sender_id`, `receiver_id`, `failed_attempts`, `location`, `is_fraud`. Unlabelled datasets work; the app doesn't display verified fraud counts without valid ground truth.

## Important concepts for presentation
- PySpark: CSV load, DataFrame filters, grouping, basic Spark SQL and RDD operations.
- scikit-learn: trains the fraud classifier in Colab, exports `fraud_model.joblib`.
- Rules: separate heuristic points for suspicious patterns, editable in Fraud Detection.
- Comparison: ML-only, rules-only, both or neither.
- SQLite: persists alerts, analysis runs and human review notes.
- **A rule flag or model prediction is not verified or confirmed fraud.**

## Test status
Code syntax and isolated synthetic tests may pass, but a complete Colab run and an interactive Streamlit browser test require a real dataset and environment. No trained model is included in this ZIP; generate it in Colab.
