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
# FINAL UI THEME
# --------------------------------------------------
st.markdown(
    f"""
    <style>
    /* cache-buster: {np.random.rand()} */

    @import url('https://fonts.cdnfonts.com/css/satoshi');

    /* GLOBAL FONT */
    html, body, * {{
        font-family: 'Satoshi', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    /* MAIN BACKGROUND */
    .stApp {{
        background-color: #1e3a8a;
    }}

    /* PAGE CONTENT */
    .block-container {{
        padding: 2rem;
    }}

    /* WHITE CARDS */
    section[data-testid="stVerticalBlock"] > div {{
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }}

    /* SECTION HEADINGS (WHITE) */
    h1, h2, h3 {{
        color: #ffffff !important;
        font-weight: 700 !important;
    }}

    /* CARD TEXT */
    p, span, div {{
        color: #0f172a;
    }}

    /* SIDEBAR */
    [data-testid="stSidebar"] {{
        background-color: #0f172a;
    }}

    /* SIDEBAR LABELS */
    [data-testid="stSidebar"] label {{
        color: #ffffff !important;
        font-weight: 500;
    }}

    /* SIDEBAR INPUT TEXT */
    [data-testid="stSidebar"] .stSelectbox div,
    [data-testid="stSidebar"] .stDateInput input,
    [data-testid="stSidebar"] .stRadio div,
    [data-testid="stSidebar"] .stCheckbox div,
    [data-testid="stSidebar"] .stSlider {{
        color: #ffffff !important;
    }}

    /* METRICS */
    [data-testid="stMetric"] {{
        background-color: #ffffff;
        border-radius: 14px;
        padding: 1.2rem;
        border: 1px solid #e5e7eb;
    }}

    [data-testid="stMetricLabel"] {{
        color: #475569 !important;
        font-size: 13px;
    }}

    [data-testid="stMetricValue"] {{
        color: #0f172a !important;
        font-size: 28px;
        font-weight: 700;
    }}

    /* BUTTONS */
    button {{
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        border: none !important;
    }}

    button:hover {{
        background-color: #1d4ed8 !important;
    }}

    /* DATAFRAME */
    [data-testid="stDataFrame"] {{
        background-color: #ffffff;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
    }}

    /* HIDE FOOTER */
    footer {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

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

df = load_data()

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
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

# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------
filtered_df = df[
    (df["Store"] == store_id) &
    (df["Dept"] == dept_id) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

if holiday_only:
    filtered_df = filtered_df[filtered_df["IsHoliday"]]

if filtered_df.shape[0] < 30:
    st.warning("Not enough data for selected inputs.")
    st.stop()

# --------------------------------------------------
# KEY BUSINESS METRICS
# --------------------------------------------------
st.subheader("Key Business Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Total Sales", f"{filtered_df['Weekly_Sales'].sum():,.0f}")
c2.metric("Average Weekly Sales", f"{filtered_df['Weekly_Sales'].mean():,.0f}")
c3.metric("Maximum Weekly Sales", f"{filtered_df['Weekly_Sales'].max():,.0f}")

# --------------------------------------------------
# MODEL
# --------------------------------------------------
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
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
else:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVR(C=100, epsilon=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
st.subheader("Model Performance")

m1, m2 = st.columns(2)
m1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):,.2f}")
m2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------
st.subheader("Actual vs Predicted Sales")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test.values, label="Actual")
ax.plot(y_pred, label="Predicted")
ax.legend()
st.pyplot(fig)

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------
st.subheader("Filtered Data Preview")
st.dataframe(filtered_df.head(20))
