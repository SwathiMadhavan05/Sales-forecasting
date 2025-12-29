import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    layout="wide"
)

st.title("📊 Sales Forecasting Dashboard")
st.write("Linear Regression vs Support Vector Regression (SVR)")

# -------------------------------------------------
# LOAD AND PREPROCESS DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    train = pd.read_csv("data/train.csv")
    features = pd.read_csv("data/features.csv")
    stores = pd.read_csv("data/stores.csv")

    # Merge datasets
    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    # Convert date
    df["Date"] = pd.to_datetime(df["Date"])

    # Handle missing values
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # Reduce size for speed (resume-friendly)
    df = df[(df["Store"] == 1) & (df["Dept"] == 1)]

    # Feature engineering
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

    return df

df = load_data()
st.success("Data loaded successfully")

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "Support Vector Regression (SVR)"]
)

test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

# -------------------------------------------------
# FEATURE SELECTION
# -------------------------------------------------
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

X = df[feature_cols]
y = df["Weekly_Sales"]

# -------------------------------------------------
# TRAIN-TEST SPLIT
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, shuffle=False
)

# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------
if model_choice == "Linear Regression":
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

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae:,.2f}")
col2.metric("RMSE", f"{rmse:,.2f}")

# -------------------------------------------------
# PLOT RESULTS
# -------------------------------------------------
st.subheader("📉 Actual vs Predicted Sales")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test.values, label="Actual")
ax.plot(y_pred, label="Predicted")
ax.set_xlabel("Time")
ax.set_ylabel("Weekly Sales")
ax.legend()

st.pyplot(fig)

# -------------------------------------------------
# DATA PREVIEW
# -------------------------------------------------
st.subheader("📄 Data Preview")
st.dataframe(df.head(20))
