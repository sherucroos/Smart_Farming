#  Smart Rice Farming System

Rice cultivation is a critical agricultural activity for small-scale farmers in Sri Lanka, yet crop productivity is frequently threatened by diseases such as Brown Spot, False Smut, and Tungro. Early identification of these diseases is challenging, particularly for new farmers who lack experience in visual diagnosis and have limited access to agricultural experts. Traditional disease identification methods are largely manual, time-consuming, and reactive, often resulting in late intervention, increased chemical usage, and significant yield loss.


The Smart Rice Disease Detection, Management & Early Warning System is designed to address these challenges by enabling real-time, image-based disease diagnosis using a mobile device. Farmers can capture or upload rice leaf images through a user-friendly mobile application. The system preprocesses the image and applies a trained deep learning model to identify the disease condition accurately. The solution operates entirely offline, making it suitable for rural farming regions with limited or no internet connectivity.


To enhance transparency and trust, the system incorporates Explainable AI techniques that visually highlight the affected regions of the leaf using heatmaps. This allows farmers to clearly understand where the disease is present rather than relying solely on textual predictions. In addition to disease identification, the system evaluates disease severity levels and provides early warnings before infections spread across the field.


Based on the detected disease and severity level, the system delivers simple, farmer-friendly management and treatment guidance. These recommendations help farmers take timely preventive or corrective actions, reduce unnecessary chemical application, and improve overall crop health. By combining disease detection, explainability, and actionable guidance into a single platform, this component transforms technical image analysis into practical agricultural decision support.


Overall, this system supports sustainable rice cultivation by reducing crop losses, empowering new farmers with accessible knowledge, and enabling proactive disease management. It plays a key role in bridging the gap between advanced technological solutions and real-world agricultural practices.


---

##  Table of Contents

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

##  1. Overview

Rice cultivation is a critical livelihood for small-scale farmers in Sri Lanka. However, productivity is often threatened by diseases, pests, unpredictable weather, and volatile market prices. Traditional methods are manual, reactive, and dependent on expert availability.

The **Smart Rice Farming System** addresses these challenges by providing:

* Offline-capable AI solutions
* Explainable predictions
* Farmer-friendly recommendations
* Early warnings for proactive decision-making

---
![WhatsApp Image 2026-01-11 at 10 01 04 PM](https://github.com/user-attachments/assets/946ccc12-64c8-49bd-bace-02ab73a6308f)




##  2. System Modules

---

##  2.1 Smart Rice Disease Detection, Management & Early Warning System

###  Description

An offline, image-based rice disease diagnosis system that allows farmers to detect diseases early using a mobile phone camera. The system identifies common rice diseases and provides explainable results with actionable treatment guidance.

###  Key Features

* Image-based disease detection
* Offline on-device inference
* Explainable AI with heatmaps (Grad-CAM)
* Disease severity estimation
* Early warning alerts
* Farmer-friendly management recommendations

###  Supported Disease Classes

* Brown Spot
* False Smut
* Tungro
* Healthy

###  Technology Stack

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

##  2.2 Smart Rice Yield Prediction System

###  Description

A data-driven yield prediction module that helps farmers and agricultural officers estimate rice yield based on environmental, seasonal, and regional factors.

###  Key Features

* Region-aware yield prediction
* Seasonal trend analysis
* Environmental feature integration
* Interpretable baseline and ensemble models

###  Model Output

* Predicted yield (tons/hectare)
* Seasonal trends
* Regional comparisons

**Example:**

```
Region: Anuradhapura
Season: Yala
Predicted Yield: 4.8 tons/hectare
```

###  Technology Stack

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

##  2.3 Market Price Forecasting & Decision Support System

###  Description

A time-series forecasting and recommendation system that predicts paddy market prices and converts predictions into actionable selling decisions.

###  Key Features

* Hybrid price forecasting (ARIMA + LSTM)
* Confidence-based recommendations
* Explainable trend insights
* SMS support for feature phone users

###  Recommendation Outputs

* Sell Now
* Hold (X days/weeks)
* Confidence score with explanation

###  Technology Stack

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

##  2.4 Smart Pest Prediction & Prevention System

###  Description

A preventive AI-based pest management system that predicts pest outbreak risks using field conditions, weather data, and farming practices—before damage occurs.

###  Key Features

* Pest outbreak probability prediction
* Risk-level classification (Low / Medium / High)
* Integrated Pest Management (IPM) recommendations
* Explainable risk insights
* Push notifications & alerts

###  Risk Levels

* **Low (0–30%)** – Routine monitoring
* **Medium (31–60%)** – Preventive measures
* **High (61–100%)** – Immediate intervention

###  Technology Stack

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

##  3. Technology Stack Summary

| Layer         | Technologies                        |
| ------------- | ----------------------------------- |
| Frontend      | React, React Native, Expo           |
| Backend       | Flask, FastAPI                      |
| ML / DL       | TensorFlow, Keras, Scikit-learn     |
| Visualization | Matplotlib, Seaborn                 |
| Storage       | SQLite, PostgreSQL (Optional Cloud) |
| Deployment    | Android, Server-based APIs          |

---

##  4. Deployment Environment

* Android devices (offline-first support)
* Server-based inference for yield, pest & price models
* Optimized for low-bandwidth rural environments
* Docker-based production deployment (future-ready)

---

##  5. Version Control & Collaboration

* Git & GitHub
* Branch-based development
* Pull requests & issue tracking

---

##  6. Novelty & Impact

* Tailored specifically for Sri Lankan rice farming
* Offline AI for rural accessibility
* Explainable predictions to build farmer trust
* Integrates disease, pest, yield, and market intelligence
* Promotes sustainable agriculture and reduced chemical usage

---

##  Contribution & Future Enhancements

* Multilingual support (Sinhala / Tamil / English)
* Voice-based interaction
* Cloud-based analytics & model retraining
* Community-level early warning systems

---

**Empowering farmers with AI-driven, practical, and sustainable agricultural intelligence. **
