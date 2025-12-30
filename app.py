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
st.title("📊 Sales Forecasting Dashboard")
st.write("Interactive sales analysis and forecasting using machine learning")

# --------------------------------------------------
# LOAD DATA (STREAMLIT SAFE)
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

    # Feature engineering
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    return df

df = load_data()

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("🔧 User Inputs")

store_id = st.sidebar.selectbox(
    "Select Store",
    options=sorted(df["Store"].unique()),
    format_func=lambda x: f"Store {x}"
)

dept_id = st.sidebar.selectbox(
    "Select Department",
    options=sorted(df["Dept"].unique()),
    format_func=lambda x: f"Department {x}"
)

min_date = df["Date"].min()
max_date = df["Date"].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
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
    filtered_df = filtered_df[filtered_df["IsHoliday"] == True]

if filtered_df.shape[0] < 30:
    st.warning("Not enough data for selected inputs.")
    st.stop()

# --------------------------------------------------
# KEY BUSINESS METRICS
# --------------------------------------------------
st.subheader("📌 Key Business Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"{filtered_df['Weekly_Sales'].sum():,.0f}")
col2.metric("Average Weekly Sales", f"{filtered_df['Weekly_Sales'].mean():,.0f}")
col3.metric("Maximum Weekly Sales", f"{filtered_df['Weekly_Sales'].max():,.0f}")

# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------
feature_cols = [
    "Year",
    "Month",
    "Week",
    "IsHoliday",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Size"
]

X = filtered_df[feature_cols]
y = filtered_df["Weekly_Sales"]

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
if model_type == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    X_future = X.tail(1).copy()

else:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel="rbf", C=100, epsilon=0.1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    X_future = scaler.transform(X.tail(1))

# --------------------------------------------------
# MODEL METRICS
# --------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📈 Model Performance")
col4, col5 = st.columns(2)
col4.metric("MAE", f"{mae:,.2f}")
col5.metric("RMSE", f"{rmse:,.2f}")

# --------------------------------------------------
# ACTUAL VS PREDICTED PLOT
# --------------------------------------------------
st.subheader("📉 Actual vs Predicted Sales")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test.values, label="Actual")
ax.plot(y_pred, label="Predicted")
ax.set_xlabel("Time")
ax.set_ylabel("Weekly Sales")
ax.legend()
st.pyplot(fig)

# --------------------------------------------------
# FUTURE SALES FORECAST
# --------------------------------------------------
st.subheader("🔮 Future Sales Forecast")

future_sales = []
last_date = filtered_df["Date"].max()

for i in range(weeks_ahead):
    if model_type == "Linear Regression":
        pred = model.predict(X.tail(1))[0]
    else:
        pred = model.predict(X_future)[0]

    future_sales.append(pred)

future_dates = [last_date + timedelta(weeks=i+1) for i in range(weeks_ahead)]

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": future_sales
})

st.line_chart(future_df.set_index("Date"))

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------
st.subheader("📄 Filtered Data Preview")
st.dataframe(filtered_df.head(20))
