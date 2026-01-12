# 🌾 Smart Rice Farming System

An AI-powered, farmer-centric decision support platform designed to improve rice cultivation in Sri Lanka through **disease detection**, **yield prediction**, **market price forecasting**, and **pest outbreak prevention**. The system bridges advanced machine learning with real-world agricultural practices, optimized for rural and low-connectivity environments.

---

## 📌 Table of Contents

1. Overview
2. System Modules

   * Rice Disease Detection, Management & Early Warning
   * Rice Yield Prediction
   * Market Price Forecasting & Decision Support
   * Pest Prediction & Prevention
3. Technology Stack
4. Deployment Environment
5. Version Control
6. Novelty & Impact

---

## 🌱 1. Overview

Rice cultivation is a critical livelihood for small-scale farmers in Sri Lanka. However, productivity is often threatened by diseases, pests, unpredictable weather, and volatile market prices. Traditional methods are manual, reactive, and dependent on expert availability.

The **Smart Rice Farming System** addresses these challenges by providing:

* Offline-capable AI solutions
* Explainable predictions
* Farmer-friendly recommendations
* Early warnings for proactive decision-making

---
![WhatsApp Image 2026-01-11 at 10 01 04 PM](https://github.com/user-attachments/assets/946ccc12-64c8-49bd-bace-02ab73a6308f)




## 🧩 2. System Modules

---

## 🦠 2.1 Smart Rice Disease Detection, Management & Early Warning System

### 📖 Description

An offline, image-based rice disease diagnosis system that allows farmers to detect diseases early using a mobile phone camera. The system identifies common rice diseases and provides explainable results with actionable treatment guidance.

### 🎯 Key Features

* Image-based disease detection
* Offline on-device inference
* Explainable AI with heatmaps (Grad-CAM)
* Disease severity estimation
* Early warning alerts
* Farmer-friendly management recommendations

### 🦠 Supported Disease Classes

* Brown Spot
* False Smut
* Tungro
* Healthy

### 🛠 Technology Stack

**Programming Languages**

* Python
* JavaScript / TypeScript

**Frameworks & Tools**

* TensorFlow / Keras
* Flask / FastAPI
* React Native
* Expo Go
* Visual Studio Code

**Python Libraries**

* NumPy
* Pandas
* OpenCV (cv2)
* Matplotlib / Seaborn
* Scikit-learn
* Joblib / TensorFlow SavedModel

**Machine Learning Model**

* DenseNet121 (Pre-trained, Frozen)
* Custom Fully Connected Classifier

**Explainable AI**

* Grad-CAM heatmaps

**Deployment**

* Android devices (Android 8+)
* Offline on-device inference
* Optimized for low-end smartphones

---

## 🌾 2.2 Smart Rice Yield Prediction System

### 📖 Description

A data-driven yield prediction module that helps farmers and agricultural officers estimate rice yield based on environmental, seasonal, and regional factors.

### 🎯 Key Features

* Region-aware yield prediction
* Seasonal trend analysis
* Environmental feature integration
* Interpretable baseline and ensemble models

### 📊 Model Output

* Predicted yield (tons/hectare)
* Seasonal trends
* Regional comparisons

**Example:**

```
Region: Anuradhapura
Season: Yala
Predicted Yield: 4.8 tons/hectare
```

### 🛠 Technology Stack

**Programming Languages**

* Python
* JavaScript / TypeScript

**Frameworks & Tools**

* Scikit-learn
* Flask / FastAPI
* React / React Native
* Visual Studio Code

**Machine Learning Models**

* Random Forest Regressor
* Linear Regression

**Feature Engineering**

* Soil & weather data integration
* Seasonal encoding (Yala / Maha)
* Region-based feature mapping

**Evaluation Metrics**

* RMSE
* MAE
* R² Score

---

## 💰 2.3 Market Price Forecasting & Decision Support System

### 📖 Description

A time-series forecasting and recommendation system that predicts paddy market prices and converts predictions into actionable selling decisions.

### 🎯 Key Features

* Hybrid price forecasting (ARIMA + LSTM)
* Confidence-based recommendations
* Explainable trend insights
* SMS support for feature phone users

### 📈 Recommendation Outputs

* Sell Now
* Hold (X days/weeks)
* Confidence score with explanation

### 🛠 Technology Stack

**Programming Languages**

* Python
* JavaScript / TypeScript

**Frameworks & Tools**

* TensorFlow / Keras
* Statsmodels (ARIMA)
* Flask / FastAPI
* React / Next.js / React Native

**Machine Learning Models**

* ARIMA (short-term trends)
* LSTM (long-term & nonlinear trends)
* Hybrid forecasting strategy

**User Communication**

* Web & mobile dashboards
* SMS gateway integration

---

## 🐛 2.4 Smart Pest Prediction & Prevention System

### 📖 Description

A preventive AI-based pest management system that predicts pest outbreak risks using field conditions, weather data, and farming practices—before damage occurs.

### 🎯 Key Features

* Pest outbreak probability prediction
* Risk-level classification (Low / Medium / High)
* Integrated Pest Management (IPM) recommendations
* Explainable risk insights
* Push notifications & alerts

### 🚦 Risk Levels

* **Low (0–30%)** – Routine monitoring
* **Medium (31–60%)** – Preventive measures
* **High (61–100%)** – Immediate intervention

### 🛠 Technology Stack

**Machine Learning Models**

* Random Forest Classifier
* Logistic Regression
* Weighted Ensemble Strategy

**Python Libraries**

* NumPy
* Pandas
* Scikit-learn
* Matplotlib / Seaborn
* Imbalanced-learn (SMOTE)
* Joblib

**External Integrations**

* Weather APIs
* Rice Research & Development Institute data
* GIS-based regional pest pressure mapping

**Recommendations**

* Cultural controls
* Biological controls
* Mechanical controls
* Chemical controls (last resort)

---

## 🧰 3. Technology Stack Summary

| Layer         | Technologies                        |
| ------------- | ----------------------------------- |
| Frontend      | React, React Native, Expo           |
| Backend       | Flask, FastAPI                      |
| ML / DL       | TensorFlow, Keras, Scikit-learn     |
| Visualization | Matplotlib, Seaborn                 |
| Storage       | SQLite, PostgreSQL (Optional Cloud) |
| Deployment    | Android, Server-based APIs          |

---

## 🚀 4. Deployment Environment

* Android devices (offline-first support)
* Server-based inference for yield, pest & price models
* Optimized for low-bandwidth rural environments
* Docker-based production deployment (future-ready)

---

## 🔁 5. Version Control & Collaboration

* Git & GitHub
* Branch-based development
* Pull requests & issue tracking

---

## 🌍 6. Novelty & Impact

* Tailored specifically for Sri Lankan rice farming
* Offline AI for rural accessibility
* Explainable predictions to build farmer trust
* Integrates disease, pest, yield, and market intelligence
* Promotes sustainable agriculture and reduced chemical usage

---

## 🤝 Contribution & Future Enhancements

* Multilingual support (Sinhala / Tamil / English)
* Voice-based interaction
* Cloud-based analytics & model retraining
* Community-level early warning systems

---

**Empowering farmers with AI-driven, practical, and sustainable agricultural intelligence. 🌱**
