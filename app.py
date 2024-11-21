import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

# Load pre-trained SLR model
with open('slr_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Generate internal sample data
sample_data = pd.DataFrame({
    'Engine Size(L)': [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
    'CO2 Emissions(g/km)': [145, 185, 210, 250, 275, 320, 350, 400, 450]
})

# App title and description
st.title("🚗 CO2 Emissions Predictor")
st.markdown("""
Predict **CO2 emissions** of vehicles using their engine size with a Simple Linear Regression (SLR) model.  
Explore sample data trends and make predictions dynamically!
""")

# Sidebar: User input for engine size
st.sidebar.header("Input Parameters")
engine_size = st.sidebar.slider("Select Engine Size (L)", min_value=1.0, max_value=6.0, step=0.1)

# Sidebar: Sample data toggle
show_sample_data = st.sidebar.checkbox("Show Sample Data")

# Predict CO2 Emissions
input_data = np.array([[engine_size]])
prediction = model.predict(input_data)[0]

# Display the prediction
st.subheader("Predicted CO2 Emissions")
st.metric(label="CO2 Emissions (g/km)", value=f"{prediction:.2f}")

# Display sample data
if show_sample_data:
    st.subheader("Sample Data")
    st.write(sample_data)

# Visualization: Engine Size vs. CO2 Emissions
st.subheader("Engine Size vs. CO2 Emissions")
plt.figure(figsize=(8, 6))

# Scatterplot for sample data
plt.scatter(sample_data['Engine Size(L)'], sample_data['CO2 Emissions(g/km)'], color='blue', label="Sample Data")

# Regression line
plt.plot(sample_data['Engine Size(L)'], model.predict(sample_data[['Engine Size(L)']]), color='red', label="Regression Line")

# Customize plot
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Relationship Between Engine Size and CO2 Emissions")
plt.legend()

# Display the plot
st.pyplot(plt)

# Fun Fact Section
st.subheader("🚀 Fun Fact")
st.info("Did you know? A smaller engine size generally produces fewer CO2 emissions, making vehicles more eco-friendly!")

# Footer
st.markdown("""
---
**Developed by [Ashish Patel](https://github.com/ashishpatel8736)**  
Powered by **Streamlit** and **Scikit-learn**.
""")
