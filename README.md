Smart Farming Assistant for Rice Farmers in Sri Lanka

📱 AI-Powered Agricultural Decision Support System

📌 Project Overview

The Smart Farming Assistant is an AI-powered, mobile-based agricultural decision support system designed to help small-scale rice farmers in Sri Lanka make informed, data-driven decisions.

The system integrates machine learning, deep learning, and time-series forecasting to address key challenges in rice cultivation, including disease detection, pest outbreak prevention, yield estimation, and market price forecasting.

The platform supports both smartphone users (mobile/web application) and feature phone users (SMS alerts), ensuring accessibility in rural and low-connectivity environments.

🧩 System Components
🌱 Component 1: Rice Disease Detection, Management & Early Warning System
🐛 Component 2: Smart Pest Prediction and Prevention System
🌾 Component 3: Rice Yield Prediction System
📈 Component 4: Market Price Forecasting & Decision Support System

![WhatsApp Image 2026-01-11 at 10 01 04 PM](https://github.com/user-attachments/assets/239b1f38-2324-492e-b92d-4075b18aa6c9)







🌱 Component 1: Rice Disease Detection, Management & Early Warning System

Rice cultivation is a critical agricultural activity for small-scale farmers in Sri Lanka. However, crop productivity is frequently threatened by diseases such as Brown Spot, False Smut, and Tungro. Traditional disease identification methods are largely manual, time-consuming, and reactive, often resulting in delayed intervention, excessive chemical usage, and yield loss.

This component enables real-time, image-based disease diagnosis using a mobile device. Farmers can capture or upload rice leaf images through a user-friendly mobile application. The system preprocesses images and applies a trained deep learning model to accurately identify disease conditions. The solution supports offline inference, making it suitable for rural regions with limited internet connectivity.

To enhance transparency and trust, the system integrates Explainable AI (Grad-CAM) techniques to visually highlight infected leaf regions. It also assesses disease severity levels and generates early warnings to prevent disease spread. Based on predictions, farmers receive simple management and treatment recommendations, reducing unnecessary chemical usage and improving crop health.

🔍 Key Features

Image-based rice disease detection

Offline on-device inference

Explainable AI heatmaps (Grad-CAM)

Disease severity estimation

Farmer-friendly treatment guidance

⚙️ Module-Specific Dependencies
Programming Languages

Python – Backend development, image preprocessing, model training, inference

JavaScript / TypeScript – Cross-platform mobile application development

Frameworks & Tools

TensorFlow / Keras – Deep learning model implementation

Flask / FastAPI – RESTful API exposure

React Native – Cross-platform mobile app

Expo Go – App testing and debugging

Visual Studio Code – Development environment

Python Libraries

NumPy, Pandas, OpenCV

Matplotlib / Seaborn

Scikit-learn

Joblib / TensorFlow SavedModel

Machine Learning Models

DenseNet121 (Pre-trained, Frozen)

Custom classifier layers

Supported Classes

Brown Spot

False Smut

Tungro

Healthy

Explainable AI

Grad-CAM – Visual heatmaps for transparency

Deployment

Android devices (Android 8+)

Offline on-device inference

Optimized for low-end smartphones

🐛 Component 2: Smart Pest Prediction and Prevention System

This AI-powered component predicts potential pest outbreaks based on field conditions, crop stage, and farming practices, enabling preventive action before damage occurs. Unlike image-based pest detection, this system focuses on outbreak probability prediction and Integrated Pest Management (IPM).

🔍 Key Features

Crop stage–aware pest risk prediction

Preventive pest management

Integrated Pest Management (IPM) recommendations

Explainable risk insights

⚙️ Technologies

ML Models: Random Forest, Logistic Regression

Framework: Scikit-learn

Backend: Flask / FastAPI

Data Sources: Pest history, weather, field data

📦 Output

Pest type

Outbreak probability

Risk level (Low / Medium / High)

Stage-specific IPM recommendations

🌾 Component 3: Rice Yield Prediction System

This component predicts expected rice yield using historical yield records, soil characteristics, and weather data. It helps farmers and agricultural officers plan cultivation, storage, and supply management.

🔍 Key Features

Region- and season-based yield prediction

Supports Sri Lanka’s micro-climate zones

Seasonal trend analysis and comparison

⚙️ Technologies

ML Models: Random Forest Regressor, Linear Regression

Framework: Scikit-learn

Backend: Flask / FastAPI

Data Sources: Yield, soil, weather data

📦 Model Output

Estimated yield (tons/hectare)

Seasonal trends

Regional comparison

Example:
Region: Anuradhapura
Season: Yala
Predicted Yield: 4.8 tons/hectare

📈 Component 4: Market Price Forecasting & Decision Support System

This component forecasts future rice market prices using historical market data and delivers actionable selling recommendations to farmers via SMS and mobile/web dashboards.

🔍 Key Features

Short- and long-term price forecasting

Sell-now or hold recommendations

Confidence-based decision support

SMS alerts for feature phone users

⚙️ Technologies

Models: ARIMA, LSTM

Frameworks: TensorFlow, Statsmodels

Backend: Flask / FastAPI

Frontend: React / Next.js / React Native

SMS Gateway: API-based service

📦 Output

Forecasted prices

Price trend graphs

SMS recommendations

Example:
Current Price: Rs. 115/kg
Forecasted Price (2 weeks): Rs. 128/kg
Recommendation: Wait before selling

🗄️ Data Storage

Local databases (SQLite / PostgreSQL)

Optional cloud storage for analytics and model retraining

🚀 Deployment Environment

Server-based inference

Optimized for low-bandwidth rural environments

Offline support (selected components)

🔄 Version Control

Git & GitHub – Source code management and collaboration

👥 Team Members
Name	Component
Aleem MJA	Rice Disease Detection
Nawarathna M A S W	Pest Outbreak Prediction
Farjees MTMT	Rice Yield Prediction
Croos IS	Market Price Forecasting & SMS Alerts
🌟 Future Enhancements

Voice-based interaction

IoT sensor integration

Expansion to other crops

Centralized agricultural analytics

📄 License

Developed for academic and research purposes only.
