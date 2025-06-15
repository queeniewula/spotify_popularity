# 🎧 Spotify Popularity Predictor

This Streamlit app predicts whether a song is likely to be popular on Spotify based on its audio features.  
It’s powered by a Random Forest model trained on 100K+ Spotify tracks and deployed using a hosted model from Hugging Face.

---

## 📌 Project Background

This app is a personal extension of a group machine learning project originally developed for UCLA's CS M148 course.  
The group project explored what makes a song "popular" using audio feature data from Spotify and built several classification models.

This version builds on that work by:
- Retraining the best-performing model with balanced data using SMOTE
- Hosting the model on Hugging Face for deployment in a lightweight app
- Creating a fully interactive web app with real song presets and prediction feedback
- Exploring SHAP to interpret model predictions and understand feature importance (excluded from app for speed)

The original group repo can be found here (for transparency):  
🔗 [Original GitHub Repo](https://github.com/nathandhummi/popularity-wrapped)

---

## 📂 Project Structure

- `01_group_baseline_popularity_analysis.ipynb`:  
  The original group notebook from our UCLA CS M148 project. It includes data cleaning, exploratory analysis, model comparison (Logistic Regression, KNN, Random Forest, etc.), and initial evaluation metrics. This was built collaboratively.

- `02_personal_extensions_popularity_analysis.ipynb`:  
  My solo follow-up notebook where I extended the project by experimenting with SMOTE (to handle class imbalance), SHAP (to interpret feature importance), and explored model predictions on personal Spotify data. This formed the foundation for the interactive Streamlit app.

- `spotify_app.py`:  
  The final deployed app that allows users to adjust song features and get real-time predictions. The model is hosted externally via Hugging Face to keep the app lightweight and fast.

---

## 💡 Features

- 🎛 Adjustable sliders for song features (tempo, energy, valence, etc.)
- 🎵 Preset examples from real popular songs
- 🔍 Prediction feedback: “Popular” or “Unpopular”
- ☁️ Model loaded from Hugging Face for lightweight deployment

---

## 🧠 Model Details

- **Algorithm:** Random Forest Classifier  
- **Training Set:** Balanced using SMOTE  
- **Performance:**  
  - Accuracy: ~97%  
  - True Positive Rate: ~62%  
- **Hosted Model:**  
  [Hugging Face Model Repo](https://huggingface.co/queeniewula/spotify-popularity-model)

---

## 📦 Setup & Run Locally

1. Clone this repo:
   ```bash
   git clone https://github.com/queeniewula/spotify-popularity-app.git
   cd spotify-popularity-app
