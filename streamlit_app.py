import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Title
st.title("Number of Goals Predictor")

# Sidebar Inputs
st.sidebar.header("User Input Features")
matches_played = st.sidebar.slider("Matches Played", 10, 38, 19)
goals_scored = st.sidebar.slider("Goals Scored", 0, 30, 10)

# Sample Data (You can replace this with real EPL data)

if "df" not in st.session_state:
    data = {
        "Matches Played": np.random.randint(5, 38, 50),
        "Goals Scored": np.random.randint(0, 25, 50),
        "Finishing Ability": np.random.randint(5, 20, 50),
    }
    st.session_state.df = pd.DataFrame(data)

df = st.session_state.df  # Use stored dataset


# Define Features (X) and Target (y)
X = df[["Matches Played", "Goals Scored"]]
y = df["Finishing Ability"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display Data
st.subheader("Sample Dataset")
st.write(df.head())

# Show Model Metrics
st.subheader("Model Performance")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Predict User Input
user_input = np.array([[matches_played, goals_scored]])
predicted_finishing = model.predict(user_input)[0]

# Show Prediction
st.subheader("Predicted Finishing Ability")
st.write(f"For a player with **{matches_played}** matches and **{goals_scored}** goals, the predicted finishing ability is **{predicted_finishing:.2f}/20**")

# Run in Terminal: streamlit run app.py
