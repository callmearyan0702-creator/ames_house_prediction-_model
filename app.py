
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ames Housing Price Predictor", page_icon="🏠", layout="wide")

# --------------------
# Load assets
# --------------------
model = pickle.load(open("ridge_model.pkl", "rb"))
processor = pickle.load(open("processor.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

df = pd.read_csv("AmesHousing.csv")
X = df.drop(["SalePrice", "Order", "PID"], axis=1)

# Replace with your notebook metrics
R2_SCORE = 0.90
MAE = 15542

# --------------------
# Header
# --------------------
st.title("🏠 Ames Housing Price Prediction")
st.markdown(
    "Predict residential property prices using a **tuned Ridge Regression model** trained on the Ames Housing dataset."
)

# --------------------
# Sidebar
# --------------------
with st.sidebar:
    st.header("Property Features")

    overall_qual = st.slider("Overall Quality", 1, 10, 6)
    gr_liv_area = st.number_input("Living Area (sq ft)", 300, 10000, 1500)
    total_bsmt_sf = st.number_input("Total Basement SF", 0, 6000, 900)
    first_flr_sf = st.number_input("1st Floor SF", 300, 6000, 1200)
    garage_cars = st.slider("Garage Cars", 0, 6, 2)
    full_bath = st.slider("Full Baths", 0, 6, 2)
    year_built = st.number_input("Year Built", 1870, 2026, 2000)

    neighborhood = st.selectbox(
        "Neighborhood",
        sorted(df["Neighborhood"].dropna().unique())
    )

    house_style = st.selectbox(
        "House Style",
        sorted(df["House Style"].dropna().unique())
    )

# --------------------
# Metrics row
# --------------------
c1, c2 = st.columns(2)
c1.metric("Model R²", f"{R2_SCORE:.3f}")
c2.metric("MAE", f"${MAE:,.0f}")

with st.expander("ℹ️ How it works"):
    st.write("""
    **Model:** Ridge Regression

    **Why Ridge?**
    - Handles multicollinearity
    - Reduces overfitting
    - Produces more stable predictions

    **Pipeline**
    1. Raw inputs collected
    2. Same preprocessing used during training
    3. Features scaled
    4. Ridge model predicts house value

    **Interpretation**
    Average prediction error is approximately $15.5K.
    """)

# --------------------
# Prediction
# --------------------
if st.button("Predict House Price", use_container_width=True):

    row = X.iloc[[0]].copy()

    updates = {
        "Overall Qual": overall_qual,
        "Gr Liv Area": gr_liv_area,
        "Total Bsmt SF": total_bsmt_sf,
        "1st Flr SF": first_flr_sf,
        "Garage Cars": garage_cars,
        "Full Bath": full_bath,
        "Year Built": year_built,
        "Neighborhood": neighborhood,
        "House Style": house_style,
    }

    for k, v in updates.items():
        if k in row.columns:
            row[k] = v

    encoded = processor.transform(row)
    scaled = scaler.transform(encoded)

    prediction = float(model.predict(scaled)[0])

    lower = prediction * 0.85
    upper = prediction * 1.15

    st.success(
        f"Estimated Price: ${prediction:,.0f}  |  Expected Range: ${lower:,.0f} – ${upper:,.0f}"
    )

    # --------------------
    # Dashboard
    # --------------------
    left, right = st.columns([1,1])

    with left:
        st.subheader("📊 Your Home vs Dataset Average")

        compare = pd.DataFrame({
            "Feature": ["Living Area","Basement","Garage Cars","Bathrooms"],
            "Your Home": [
                gr_liv_area,
                total_bsmt_sf,
                garage_cars,
                full_bath
            ],
            "Dataset Average": [
                df["Gr Liv Area"].mean(),
                df["Total Bsmt SF"].mean(),
                df["Garage Cars"].mean(),
                df["Full Bath"].mean()
            ]
        }).set_index("Feature")

        st.bar_chart(compare)

    with right:
        st.subheader("🚀 Top Price Drivers")

        drivers = []

        if overall_qual >= 8:
            drivers.append(("Overall Quality", "Premium build quality"))

        top_neighborhoods = (
            df.groupby("Neighborhood")["SalePrice"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .index
        )

        if neighborhood in top_neighborhoods:
            drivers.append(("Neighborhood", "High-value location"))

        if gr_liv_area > df["Gr Liv Area"].mean():
            drivers.append(("Living Area", "Above-average size"))

        if total_bsmt_sf > df["Total Bsmt SF"].mean():
            drivers.append(("Basement", "Large basement area"))

        if garage_cars >= 3:
            drivers.append(("Garage", "Large garage capacity"))

        if not drivers:
            drivers.append(("Balanced Profile", "Mostly average characteristics"))

        for idx, item in enumerate(drivers[:3], start=1):
            st.markdown(f"**{idx}. {item[0]}**")
            st.write(item[1])


    st.info(
        f"""
        The model estimates this property is worth approximately **${prediction:,.0f}**.
        Given the model's average error of **${MAE:,.0f}**, the realistic market value
        is expected to fall within the displayed range. This prediction is driven primarily
        by property quality, neighborhood, living area, and structural characteristics.
        """
    )
