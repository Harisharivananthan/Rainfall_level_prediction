# 🌧️ Rainfall Level Prediction & Flood Risk Forecasting

A time-series ML system that predicts rainfall levels and flags flood risk using historical weather data from Chennai (1984–2024).

---

## 📌 Overview

Accurate rainfall forecasting is critical for disaster preparedness and urban planning. This project builds an end-to-end ML pipeline — from raw data ingestion to a deployed REST API — that forecasts rainfall and estimates flood risk for a given date range.

**Key results:**
- ~85% prediction accuracy on the test set
- ~18% reduction in RMSE compared to the baseline Random Forest model
- Dockerized FastAPI backend for real-time inference

---

## ✨ Features

- 📊 Time-series feature engineering with leakage-safe lag/rolling features
- 🤖 LSTM model for sequential rainfall forecasting
- 🌲 Random Forest baseline for benchmarking
- ⚡ FastAPI backend for real-time inference
- 🐳 Docker support for consistent deployment
- 📈 Monitoring setup for model performance tracking

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python |
| ML Models | LSTM (TensorFlow/Keras), Random Forest (Scikit-learn) |
| API | FastAPI |
| Containerization | Docker |
| Data | Chennai rainfall & weather data, 1984–2024 |

---

## 🗂️ Project Structure

```
Rainfall_level_prediction/
├── src/                  # Model training, preprocessing, inference
├── data/                 # Raw and processed datasets
├── docker/               # Dockerfile and compose files
├── k8s/                  # Kubernetes config (optional scaling)
├── monitoring/           # Performance monitoring setup
├── .github/workflows/    # CI pipeline
├── requirements.txt
└── Makefile
```

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized run)

### Installation

```bash
git clone https://github.com/Harisharivananthan/Rainfall_level_prediction
cd Rainfall_level_prediction

pip install -r requirements.txt
```

### Run locally

```bash
make all
```

### Run with Docker

```bash
docker build -f docker/Dockerfile -t rainfall-api .
docker run -p 8000:8000 rainfall-api
```

---

## 📡 API Usage

Once running, the API is available at `http://localhost:8000`.

**Example request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"date": "2024-06-15", "features": [...]}'
```

---

## 📊 Model Performance

| Model | Accuracy | RMSE Improvement |
|-------|----------|-----------------|
| Random Forest (baseline) | ~72% | — |
| LSTM | ~85% | ~18% reduction |

---

## 🚀 Future Improvements

- Add real-time weather API integration (e.g. OpenWeatherMap)
- Build a Streamlit dashboard for visual forecasting
- Deploy to cloud (AWS/GCP) with auto-retraining pipeline
- Extend dataset to cover more Indian cities

---

## 👨‍💻 Author

**Harish Arivananthan**
- GitHub: [@Harisharivananthan](https://github.com/Harisharivananthan)
- LinkedIn: [harish-a-38ab36255](https://linkedin.com/in/harish-a-38ab36255)
