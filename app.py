# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "landing"

# --------------------------------------------------
# CSS (STABLE + SAFE)
# --------------------------------------------------
st.markdown(
    f"""
    <style>
    /* cache buster {np.random.rand()} */

    @import url('https://fonts.cdnfonts.com/css/satoshi');

    * {{
        font-family: 'Satoshi', sans-serif !important;
    }}

    .stApp {{
        background-color: #1e3a8a;
    }}

    .block-container {{
        padding: 2rem;
    }}

    section[data-testid="stVerticalBlock"] > div {{
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        color: #0f172a;
    }}

    h1, h2, h3 {{
        color: white !important;
        font-weight: 700;
    }}

    .block-container * {{
        color: #0f172a;
    }}

    [data-testid="stSidebar"] {{
        background-color: #0f172a;
    }}

    [data-testid="stSidebar"] * {{
        color: white !important;
        fill: white !important;
    }}

    button {{
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 600;
    }}

    footer {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LANDING PAGE
# --------------------------------------------------
def landing_page():
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1e3a8a, #2563eb);
            padding: 80px 40px;
            border-radius: 24px;
            text-align: center;
            color: white;
        ">
            <h1 style="font-size:48px;">Sales Forecasting Dashboard</h1>
            <p style="font-size:20px; max-width:700px; margin:auto;">
                Forecast sales trends, evaluate performance,
                and make data-driven business decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Enter Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    train = pd.read_csv(os.path.join(base_dir, "data", "train.csv"))
    features = pd.read_csv(os.path.join(base_dir, "data", "features.csv"))
    stores = pd.read_csv(os.path.join(base_dir, "data", "stores.csv"))

    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    df["Date"] = pd.to_datetime(df["Date"])
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    return df

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
def dashboard():
    df = load_data()

    # SIDEBAR
    st.sidebar.header("User Inputs")

    store_id = st.sidebar.selectbox("Select Store", sorted(df["Store"].unique()))
    dept_id = st.sidebar.selectbox("Select Department", sorted(df["Dept"].unique()))

    date_range = st.sidebar.date_input(
        "Select Date Range",
        [df["Date"].min(), df["Date"].max()]
    )

    model_type = st.sidebar.radio(
        "Select Model",
        ["Linear Regression", "Support Vector Regression (SVR)"]
    )

    holiday_only = st.sidebar.checkbox("Show Holiday Sales Only")
    weeks_ahead = st.sidebar.slider("Forecast Weeks Ahead", 1, 12, 4)

    # FILTER DATA
    filtered_df = df[
        (df["Store"] == store_id) &
        (df["Dept"] == dept_id) &
        (df["Date"] >= pd.to_datetime(date_range[0])) &
        (df["Date"] <= pd.to_datetime(date_range[1]))
    ]

    if holiday_only:
        filtered_df = filtered_df[filtered_df["IsHoliday"]]

    if len(filtered_df) < 30:
        st.warning("Not enough data for selected inputs.")
        st.stop()

    # METRICS
    st.subheader("Key Business Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales", f"{filtered_df['Weekly_Sales'].sum():,.0f}")
    c2.metric("Average Weekly Sales", f"{filtered_df['Weekly_Sales'].mean():,.0f}")
    c3.metric("Maximum Weekly Sales", f"{filtered_df['Weekly_Sales'].max():,.0f}")

    # MODEL
    X = filtered_df[
        ["Year", "Month", "Week", "IsHoliday",
         "Temperature", "Fuel_Price", "CPI",
         "Unemployment", "Size"]
    ]
    y = filtered_df["Weekly_Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    if model_type == "Linear Regression":
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        model = SVR(C=100, epsilon=0.1).fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # PERFORMANCE
    st.subheader("Model Performance")
    m1, m2 = st.columns(2)
    m1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.2f}")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

    # PLOT
    st.subheader("Actual vs Predicted Sales")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test.values, label="Actual")
    ax.plot(y_pred, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    # DATA PREVIEW
    st.subheader("Filtered Data Preview")
    st.dataframe(filtered_df.head(20))

# --------------------------------------------------
# PAGE SWITCH
# --------------------------------------------------
if st.session_state.page == "landing":
    landing_page()
else:
    dashboard()



