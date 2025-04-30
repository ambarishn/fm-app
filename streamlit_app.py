import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("⚽ Shots vs. Goals Predictor")
st.write("Use simple linear regression to predict average goals per game based on shots per game.")

# Sample data: Shots per game vs Goals per game
X = np.array([[2], [4], [6], [8], [10]])  # Shots per game
y = np.array([0.5, 1.0, 1.5, 2.0, 2.5])  # Goals per game

# Train the model
model = LinearRegression()
model.fit(X, y)

# User input
shots = st.slider("Shots per Game", min_value=0, max_value=15, value=5)
predicted_goals = model.predict([[shots]])[0]

# Show prediction
st.subheader("Prediction")
st.write(f"🧐 Expected Goals per Game: **{predicted_goals:.2f}**")

# Show regression details
st.subheader("Model Equation")
slope = model.coef_[0]
intercept = model.intercept_
st.latex(r"y = {:.2f} \cdot x + {:.2f}".format(slope, intercept))

# Plotting
fig, ax = plt.subplots()
ax.scatter(X, y, color='green', label='Actual data')
ax.plot(X, model.predict(X), color='orange', label='Regression line')
ax.scatter(shots, predicted_goals, color='red', label='Your prediction', zorder=5)
ax.set_xlabel("Shots per Game")
ax.set_ylabel("Goals per Game")
ax.set_title("Linear Regression: Shots vs Goals")
ax.legend()
st.pyplot(fig)
