# Smart_Farming
Mobile-based AI-powered smart farming assistant for Sri Lankan rice farmers, featuring disease detection, pest identification, yield prediction, and market price forecasting.
# 🌾 Mobile-Based Smart Farming Assistant for Rice Farmers in Sri Lanka

## 📌 Project Overview
This project is an AI-powered smart farming assistant designed specifically for **small-scale rice farmers in Sri Lanka**. The system helps farmers make data-driven decisions by integrating **machine learning, computer vision, and time-series forecasting** into a user-friendly mobile/web platform.

The solution addresses major agricultural challenges such as delayed disease detection, pest infestations, inaccurate yield estimation, and unpredictable market price fluctuations.

---

## 🎯 Key Features
- 🌱 **Rice Disease Detection**
  - Image-based detection using CNN models (MobileNet / VGG16)
  - Supports common rice diseases such as Rice Blast, Bacterial Leaf Blight, Sheath Blight, Brown Spot, and False Smut
  - Explainable AI using **Grad-CAM** to highlight affected leaf areas

- 🐛 **Rice Pest Identification**
  - Image-based pest classification for Sri Lankan rice pests
  - Provides organic and chemical control recommendations
  - Farmer feedback loop to improve model accuracy

- 🌾 **Rice Yield Prediction**
  - Predicts yield using historical yield, soil, and weather data
  - ML models such as Random Forest and Linear Regression
  - Supports Sri Lanka’s micro-climate zones

- 📈 **Market Price Forecasting**
  - Time-series forecasting using ARIMA / LSTM
  - Personalized SMS alerts based on crop, region, and language
  - Helps farmers decide the best time to sell

---

## 🧠 Technologies Used
### AI & Machine Learning
- TensorFlow / Keras
- Scikit-learn
- CNN (MobileNet, VGG16)
- LSTM / ARIMA
- Grad-CAM (Explainable AI)

### Backend
- Python (Flask / Django)
- REST APIs
- MongoDB / SQLite

### Frontend
- Flutter / React Native
- Multilingual support (Sinhala, Tamil, English)

### Data Sources
- Rice disease & pest image datasets
- Historical yield and soil data
- Weather APIs (e.g., OpenWeatherMap)
- Government market price bulletins

---

## 🏗️ System Architecture
1. Farmer uploads image / inputs data via mobile app  
2. Backend processes data using ML/DL models  
3. Predictions and recommendations are generated  
4. Results delivered via app dashboard or SMS alerts  

---

## 👥 Team Members
- **Farjees MTMT** – Rice Yield Prediction  
- **Aleem MJA** – Rice Disease Detection (CNN + XAI)  
- **Nawarathna M A S W** – Rice Pest Identification  
- **Croos IS** – Market Price Forecasting & SMS Alerts  

---

## 📍 Target Users
- Small-scale rice farmers in Sri Lanka  
- Agricultural officers and researchers  

---

## 🚀 Future Enhancements
- Offline image inference for rural areas
- Voice-based interaction for low-literacy users
- Expansion to other crops
- IoT sensor integration for real-time field data

---

## 📄 License
This project is developed for academic and research purposes.

---

## 🙏 Acknowledgements
- Sri Lankan Department of Agriculture  
- Open-source ML community  
- University supervisors and evaluators
