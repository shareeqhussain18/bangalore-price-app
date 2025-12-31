import streamlit as st
import pickle
import json
import numpy as np

st.image("profile.jpg", width=120)
st.markdown("### Developed by MOHD.SHAREEQ")

# Load model
with open("bangalore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)



# Load columns
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

locations = data_columns[3:]  # skip sqft, bath, bhk

st.title("Bangalore House Price Prediction")

sqft = st.number_input("Total Square Feet", 300, 10000, step=50)
bath = st.number_input("Bathrooms", 1, 10)
bhk = st.number_input("BHK", 1, 10)

location = st.selectbox("Location", locations)

if st.button("Predict Price"):
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    price = model.predict([x])[0]
    st.success(f"Estimated Price: ₹ {round(price, 2)} Lakhs")
