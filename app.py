#force deploy
import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

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
st.write("Interactive Sales Forecasting using Machine Learning")

# --------------------------------------------------
# LOAD DATA (SAFE FOR STREAMLIT CLOUD)
# --------------------------------------------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    train = pd.read_csv(os.path.join(BASE_DIR, "data", "train.csv"))
    features = pd.read_csv(os.path.join(BASE_DIR, "data", "features.csv"))
    stores = pd.read_csv(os.path.join(BASE_DIR, "data", "stores.csv"))

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
    sorted(df["Store"].unique())
)

dept_id = st.sidebar.selectbox(
    "Select Department",
    sorted(df["Dept"].unique())
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

test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

# --------------------------------------------------
# FILTER DATA BASED ON USER INPUT
# --------------------------------------------------
filtered_df = df[
    (df["Store"] == store_id) &
    (df["Dept"] == dept_id) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

if filtered_df.shape[0] < 20:
    st.warning("Not enough data for the selected filters.")
    st.stop()

# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------
features_cols = [
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

X = filtered_df[features_cols]
y = filtered_df["Weekly_Sales"]

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=False
)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
if model_type == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

else:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel="rbf", C=100, epsilon=0.1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

# --------------------------------------------------
# METRICS
# --------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:,.2f}")
col2.metric("RMSE", f"{rmse:,.2f}")

# --------------------------------------------------
# PLOT ACTUAL VS PREDICTED
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
# DATA PREVIEW
# --------------------------------------------------
st.subheader("📄 Filtered Data Preview")
st.dataframe(filtered_df.head(20))
