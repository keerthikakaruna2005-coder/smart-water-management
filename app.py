import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("AI-Based Smart Water Management Dashboard")

# Historical Data
days = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
usage = np.array([100,110,120,130,145,155,170,180,195,210])

# Historical Usage Table
st.subheader("Historical Water Usage Data")

df = pd.DataFrame({
    "Day": [1,2,3,4,5,6,7,8,9,10],
    "Water Usage (L)": usage
})

st.dataframe(df)

# Train AI Model
model = LinearRegression()
model.fit(days, usage)

# Predict Future Usage
future_days = np.array([11,12,13,14,15]).reshape(-1,1)
predicted = model.predict(future_days)

# Prediction Graph
st.subheader("Water Demand Prediction")

fig, ax = plt.subplots()

ax.plot(days, usage, marker='o', label='Historical Usage')
ax.plot(future_days, predicted, marker='*', label='Predicted Usage')

ax.set_xlabel("Days")
ax.set_ylabel("Water Usage (Liters)")
ax.set_title("AI-Based Water Consumption Forecast")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# Latest Prediction
st.metric("Predicted Water Usage", f"{predicted[-1]:.0f} L")

# AI Alert
st.subheader("AI Alert System")

if predicted[-1] > 250:
    st.warning("⚠ High Water Consumption Predicted")
    st.write("Recommendation: Implement water-saving measures.")
else:
    st.success("✅ Water Usage Within Normal Range")
