import streamlit as st
import numpy as np
import requests
import io
import joblib

# load model
@st.cache_resource
def load_model():
    url = "https://huggingface.co/queeniewula/spotify-popularity-model/resolve/main/random_forest_model_compressed.pkl"
    response = requests.get(url)
    response.raise_for_status()
    return joblib.load(io.BytesIO(response.content))

model = load_model()


st.title("🎧 Spotify Popularity Predictor")
st.markdown("""
            This app uses a random forest machine learning model trained on Spotify song data to predict whether a track is likely to be popular.

            You can:
            - Try preset songs from known hits
            - Adjust audio feature sliders to experiment
            - See how features like tempo, valence, and energy affect popularity
            """)

# example songs that are popular 
example_songs = {
    "Him & I (with Halsey)": {
        "duration_ms": 268866,
        "danceability": 0.589,
        "energy": 0.731,
        "key": 2,
        "loudness": -6.343,
        "mode": 1,
        "speechiness": 0.0868,
        "acousticness": 0.0534,
        "instrumentalness": 0.0,
        "liveness": 0.308,
        "valence": 0.191,
        "tempo": 87.908,
        "time_signature": 4
    },
    "SKELETONS - Travis Scott": {
        "duration_ms": 145588,
        "danceability": 0.46,
        "energy": 0.686,
        "key": 0,
        "loudness": -5.948,
        "mode": 0,
        "speechiness": 0.0367,
        "acousticness": 0.00146,
        "instrumentalness": 0.0,
        "liveness": 0.375,
        "valence": 0.252,
        "tempo": 148.054,
        "time_signature": 4
    }, 
        "Counting Stars - OneRepublic": {
        "duration_ms": 257266,
        "danceability": 0.664,
        "energy": 0.705,
        "key": 1,
        "loudness": -4.972,
        "mode": 0,
        "speechiness": 0.0382,
        "acousticness": 0.0654,
        "instrumentalness": 0.0,
        "liveness": 0.118,
        "valence": 0.477,
        "tempo": 122.016, 
        "time_signature": 4
    }
}

# create a sidebar to present the preset popular examples 
st.sidebar.title("Preset Examples")
selected_song = st.sidebar.selectbox("Choose a song to autofill sliders:", ["Custom Input"] + list(example_songs.keys()))
# get defaults
default = example_songs.get(selected_song, {})

# user inputs + defaults 
duration_ms = st.slider("Duration (ms)", 0, 600000, int(default.get("duration_ms", 180000)))
danceability = st.slider("Danceability", 0.0, 1.0, default.get("danceability", 0.5))
energy = st.slider("Energy", 0.0, 1.0, default.get("energy", 0.5))
key = st.slider("Key", 0, 11, int(default.get("key", 5)))
loudness = st.slider("Loudness (dB)", -60.0, 0.0, default.get("loudness", -10.0))
mode = st.slider("Mode (0 = minor, 1 = major)", 0, 1, int(default.get("mode", 1)))
speechiness = st.slider("Speechiness", 0.0, 1.0, default.get("speechiness", 0.1))
acousticness = st.slider("Acousticness", 0.0, 1.0, default.get("acousticness", 0.5))
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, default.get("instrumentalness", 0.0))
liveness = st.slider("Liveness", 0.0, 1.0, default.get("liveness", 0.2))
valence = st.slider("Valence", 0.0, 1.0, default.get("valence", 0.5))
tempo = st.slider("Tempo (BPM)", 50, 200, int(default.get("tempo", 120)))
time_signature = st.slider("Time Signature", 1, 7, int(default.get("time_signature", 4)))

# make prediction based on user input
input_data = np.array([[
    duration_ms, danceability, energy, key, loudness, mode,
    speechiness, acousticness, instrumentalness, liveness,
    valence, tempo, time_signature
]])
prediction = model.predict(input_data)

if prediction[0] == 1:
    st.success("🎵 This song is likely **popular** on Spotify!")
else:
    st.warning("🎵 This song is likely **not popular** on Spotify.")

# shap attempts 
st.markdown(""" 
            Note: SHAP-based feature explanations were tested but excluded from this demo for performance reasons. Available in the extended notebook version.
            """)