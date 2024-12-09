import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load("trained_model.pkl")

# Streamlit App
st.title("Premier League Player Market Value Predictor")

# Sidebar for User Input
st.sidebar.header("Input Player Features")

def user_input():
    club_average_age = st.sidebar.number_input("Club Average Age", 18.0, 40.0, 25.0)
    player_age = st.sidebar.number_input("Player Age", 16.0, 40.0, 24.0)
    games_played = st.sidebar.number_input("Games Played", 0, 1000, 50)
    goals_contributed = st.sidebar.number_input("Goals Contributed", 0, 500, 10)
    height = st.sidebar.number_input("Height (cm)", 150, 220, 180)
    
    data = {
        "club_average_age": club_average_age,
        "player_age": player_age,
        "games_played": games_played,
        "goals_contributed": goals_contributed,
        "height_in_cm": height
    }
    return pd.DataFrame([data])

# Input and Prediction
input_data = user_input()

st.subheader("Player Input Features")
st.write(input_data)

if st.button("Predict Market Value"):
    prediction = model.predict(input_data)
    st.subheader("Predicted Player Market Value:")
    st.write(f"€ {prediction[0]:,.2f}")
