🌾 Smart Farming Assistant for Rice Farmers in Sri Lanka
📱 Mobile-Based AI-Powered Agricultural Decision Support System
📌 Project Overview

The Smart Farming Assistant is an AI-powered, mobile-based system developed to support small-scale rice farmers in Sri Lanka.
It integrates machine learning, deep learning, and time-series forecasting to help farmers make data-driven decisions related to crop health, pest management, yield planning, and market timing.

The system is composed of four intelligent components, each addressing a major challenge in rice cultivation.

🧩 System Components
🌱 Component 1: Rice Disease Detection System
🌾 Purpose

This component enables farmers to identify rice leaf diseases at an early stage by analyzing images captured using a mobile phone. Early detection helps reduce crop loss and ensures timely disease management.

🔍 Key Features

Image-based rice disease detection.

Supports diseases such as:

Rice Blast

Bacterial Leaf Blight

Brown Spot

Sheath Blight

False Smut

Uses deep learning CNN models for classification.

Provides Explainable AI (Grad-CAM) visual feedback.

Farmer-friendly disease insights and basic control suggestions.

⚙️ Technologies Used
Area	Tech
Model Architecture	CNN (MobileNet, VGG16)
Framework	TensorFlow / Keras
Image Processing	OpenCV
Explainable AI	Grad-CAM
Backend	Flask (Python)
Dataset	Rice leaf disease image datasets
🧪 Disease Detection Flow

Farmer captures or uploads a rice leaf image.

Image is preprocessed and normalized.

CNN model predicts disease class.

Grad-CAM highlights infected regions.

Results and recommendations are displayed.

📦 Model Output

Disease name

Confidence percentage

Heatmap of infected area

Basic treatment guidance

🛡️ Novelty

Improves farmer trust through Explainable AI.

Optimized for mobile image capture.

Focused on Sri Lankan rice diseases.

🐛 Component 2: Rice Pest Identification & Outbreak Prediction System
🌾 Purpose

This component predicts potential rice pest outbreaks based on field conditions and farming practices, enabling farmers to take preventive action before severe damage occurs.

🔍 Key Features

Predicts pest outbreak probability instead of image-based detection.

Focuses on major rice pests such as Brown Planthopper (BPH), Rice Stem Borer, Stem leaf folder.

Crop stage–aware pest risk analysis.

Integrated Pest Management (IPM) recommendations.

Promotes eco-friendly pest control and reduces chemical misuse.

⚙️ Technologies Used
Area	Tech
ML Models	Random Forest / Logistic Regression
Framework	Scikit-learn
Data Processing	Pandas, NumPy
Backend	Flask (Python)
Input Method	Farmer field-data form
Recommendation Engine	Rule-based IPM logic
Dataset	Historical pest & field-condition data
🧪 Pest Outbreak Prediction Flow

Farmer enters field information:

Location

Crop growth stage

Recent pest observations

Pesticide usage history

ML model analyzes pest risk.

Outbreak probability is calculated.

Risk level (Low / Medium / High) is assigned.

Preventive recommendations are generated.

📦 Model Output

Predicted pest type

Outbreak probability score

Risk level

Stage-specific pest management advice

🛡️ Novelty

Preventive pest management approach.

Supports Integrated Pest Management (IPM).

Reduces unnecessary pesticide use.

Designed for Sri Lankan cultivation patterns.

✅ Output Example

Pest: Brown Planthopper

Crop Stage: Vegetative

Risk Level: High (82%)

Recommendation:

Improve water management

Use biological control

Increase monitoring

Apply chemical control only as a last option

🌾 Component 3: Rice Yield Prediction System
🌾 Purpose

This component predicts expected rice yield using historical, soil, and weather data, helping farmers and agricultural officers plan cultivation, storage, and supply management.

🔍 Key Features

Predicts yield per season and region.

Uses environmental and historical datasets.

Supports Sri Lanka’s micro-climate zones.

Provides yield trends and seasonal comparison.

⚙️ Technologies Used
Area	Tech
ML Models	Random Forest, Linear Regression
Framework	Scikit-learn
Data Processing	Pandas, NumPy
Backend	Flask (Python)
Data Sources	Yield, soil, weather data
🧪 Yield Prediction Flow

Farmer selects region and cultivation season.

System retrieves relevant environmental data.

ML model predicts expected yield.

Results are displayed with trend insights.

📦 Model Output

Estimated yield value

Seasonal yield trend

Comparative analysis

🛡️ Novelty

Region-aware yield prediction.

Combines multiple environmental factors.

Supports data-driven agricultural planning.

✅ Output Example

Region: Anuradhapura

Season: Yala

Predicted Yield: 4.8 tons/hectare

📈 Component 4: Market Price Forecasting & SMS Alert System
🌾 Purpose

This component forecasts future rice market prices and delivers SMS alerts to farmers, enabling them to choose the best time to sell their harvest and maximize profit.

🔍 Key Features

Short-term and long-term price forecasting.

Uses historical market price data.

Personalized SMS alerts.

Multilingual support (Sinhala, Tamil, English).

Sell-now or wait recommendations.

⚙️ Technologies Used
Area	Tech
Forecast Models	ARIMA, LSTM
Framework	TensorFlow, Statsmodels
Data Processing	Pandas
Backend	Flask (Python)
SMS Service	API-based SMS gateway
Data Source	Government price bulletins
🧪 Price Forecasting Flow

Historical price data is collected.

Time-series models are trained.

Future prices are forecasted.

SMS alerts are sent to registered farmers.

📦 Model Output

Forecasted price values

Price trend graphs

SMS recommendation messages

🛡️ Novelty

Region-specific price forecasting.

Real-time farmer notification system.

Improves income decision-making.

✅ Output Example

Current Price: Rs. 115/kg

Forecasted Price (2 weeks): Rs. 128/kg

SMS Alert: “Wait before selling for better price”

👥 Team Members
Name	Component
Aleem MJA	Rice Disease Detection
Nawarathna M A S W	Pest Outbreak Prediction
Farjees MTMT	Rice Yield Prediction
Croos IS	Market Price Forecasting & SMS Alerts
🚀 Future Enhancements

Offline inference for rural areas

Voice-based interaction

IoT sensor integration

Expansion to other crops

📄 License

Developed for academic and research purposes only.
