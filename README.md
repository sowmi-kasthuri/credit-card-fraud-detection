# Credit Card Fraud Detection – Mini MLOps Project

This project is a compact, production-style demonstration of an end-to-end ML lifecycle using industry-standard tools like MLflow and FastAPI.  
The goal was to build a small but realistic ML system and reinforce the fundamentals before continuing to the main **MLOps Mountain** project.

---

## 🎯 Project Scope

This is a **contained mini-project**, focused only on:

- Clean ML project structure  
- Dataset ingestion  
- EDA basics (class imbalance understanding)  
- Baseline training (Logistic Regression)  
- Improved model (class_weight="balanced")  
- MLflow experiment tracking  
- MLflow Model Registry (v1 → v2)  
- Production model selection (v2)  
- FastAPI inference service loading Production model  
- Full end-to-end model prediction from API

No Prefect, no Docker, no CI/CD — these belong to the main **MLOps Mountain** project.

---

## 📁 Project Structure

credit-card-fraud-detection/
│
├── src/
│ ├── data/ # dataset download
│ ├── training/ # baseline + improved models
│ ├── inference/ # FastAPI serving
│ └── utils/ # common helpers
│
├── data/
│ ├── raw/
│ └── processed/
│
├── mlruns/ # MLflow experiments (ignored in git)
│
├── configs/
├── notebooks/
├── ADR.md
└── README.md


---

## 🧠 ML Models

### **Baseline (v1)**
- Logistic Regression  
- Solid precision and F1  
- Lower recall  
- Registered as Version 1  
- Archived after evaluation

### **Improved (v2)**
- Logistic Regression + `class_weight="balanced"`  
- Higher recall  
- Lower precision  
- Promoted to **Production**  
- Used by FastAPI inference

---

## 🚀 FastAPI Inference

The API loads the current production model directly from MLflow:

```python
model_uri="models:/fraud-model/Production"

Endpoints:

GET / → health check

POST /predict → returns:

{
  "fraud_probability": 0.93,
  "fraud_label": 1
}

Run locally:
uvicorn src.inference.api:app --reload

Swagger UI:
http://127.0.0.1:8000/docs

✔️ Completed Deliverables

Baseline model ✔

Improved model ✔

MLflow tracking ✔

Model Registry (v1 + v2) ✔

Production model: v2 ✔

FastAPI inference using Model Registry ✔

Repository cleaned and finalized ✔

This project is complete.
All future MLOps engineering (Prefect, Docker, CI/CD, Prometheus, Grafana) will happen exclusively in the MLOps Mountain project.

🧹 Status

Archived & Completed — ready to showcase or revisit later.