import os
import json
import pandas as pd
import numpy as np
import joblib

from django.shortcuts import render
from django.contrib import messages
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from .forms import UserRegistrationForm
from .models import UserRegistrationModel


# ---------------------------
# USER REGISTRATION
# ---------------------------
def UserRegisterActions(request):

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)

        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered')
            return render(request, 'UserRegistrations.html', {'form': UserRegistrationForm()})
        else:
            messages.error(request, 'Email or Mobile already existed.')
    else:
        form = UserRegistrationForm()

    return render(request, 'UserRegistrations.html', {'form': form})


# ---------------------------
# USER LOGIN
# ---------------------------
def UserLoginCheck(request):

    if request.method == "POST":

        loginid = request.POST.get('loginname')
        pswd    = request.POST.get('pswd')

        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)

            if check.status == "activated":
                request.session['id']        = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid']   = loginid
                request.session['email']     = check.email
                return render(request, 'users/UserHome.html')
            else:
                messages.warning(request, 'Your account is not activated.')

        except UserRegistrationModel.DoesNotExist:
            messages.error(request, 'Invalid Login ID or Password.')

    return render(request, 'UserLogin.html')


# ---------------------------
# USER HOME
# ---------------------------
def UserHome(request):
    return render(request, 'users/UserHome.html')


# ---------------------------
# TRAIN MODEL
# ---------------------------
def train_crypto_models(request):

    if request.method == "POST":

        if not os.path.exists("models"):
            os.makedirs("models")

        data = pd.read_csv("media/crypto_market_data.csv")

        # Encode Symbol
        encoder = LabelEncoder()
        data['Symbol'] = encoder.fit_transform(data['Symbol'])
        joblib.dump(encoder, "models/symbol_encoder.pkl")

        X = data[['Symbol', 'Open', 'High', 'Low', 'Volume', 'Market Cap']]
        y = data['Close']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled  = scaler_X.transform(X_test)

        y_train_scaled = scaler_y.fit_transform(
            y_train.values.reshape(-1, 1)
        ).ravel()

        joblib.dump(scaler_X, "models/scaler_X.pkl")
        joblib.dump(scaler_y, "models/scaler_y.pkl")

        metrics = []

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train_scaled)
        lr_pred = scaler_y.inverse_transform(
            lr.predict(X_test_scaled).reshape(-1, 1)
        ).ravel()
        joblib.dump(lr, "models/lr_model.pkl")

        mse  = mean_squared_error(y_test, lr_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test, lr_pred)
        mape = mean_absolute_percentage_error(y_test, lr_pred)
        metrics.append(["Linear Regression", mse, rmse, mae, mape])

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train_scaled, y_train_scaled)
        rf_pred = scaler_y.inverse_transform(
            rf.predict(X_test_scaled).reshape(-1, 1)
        ).ravel()
        joblib.dump(rf, "models/rf_model.pkl")

        mse  = mean_squared_error(y_test, rf_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test, rf_pred)
        mape = mean_absolute_percentage_error(y_test, rf_pred)
        metrics.append(["Random Forest", mse, rmse, mae, mape])

        # SVM
        svm = SVR()
        svm.fit(X_train_scaled, y_train_scaled)
        svm_pred = scaler_y.inverse_transform(
            svm.predict(X_test_scaled).reshape(-1, 1)
        ).ravel()
        joblib.dump(svm, "models/svm_model.pkl")

        mse  = mean_squared_error(y_test, svm_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test, svm_pred)
        mape = mean_absolute_percentage_error(y_test, svm_pred)
        metrics.append(["SVM", mse, rmse, mae, mape])

        # MLP Neural Network (replaces LSTM — no TensorFlow needed)
        mlp = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42
        )
        mlp.fit(X_train_scaled, y_train_scaled)
        mlp_pred = scaler_y.inverse_transform(
            mlp.predict(X_test_scaled).reshape(-1, 1)
        ).ravel()
        joblib.dump(mlp, "models/mlp_model.pkl")

        mse  = mean_squared_error(y_test, mlp_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test, mlp_pred)
        mape = mean_absolute_percentage_error(y_test, mlp_pred)
        metrics.append(["Neural Network (MLP)", mse, rmse, mae, mape])

        metrics_df = pd.DataFrame(
            metrics, columns=["Model", "MSE", "RMSE", "MAE", "MAPE"]
        )
        metrics_df.to_csv("models/model_metrics.csv", index=False)

    try:
        metrics_table = pd.read_csv("models/model_metrics.csv").to_html()
    except Exception:
        metrics_table = "<p>No models trained yet.</p>"

    return render(request, "users/train.html", {"metrics": metrics_table})


# ---------------------------
# PREDICT
# ---------------------------
def predict_market_cap(request):

    prediction      = {}
    actual_list     = []
    predicted_list  = []
    future_list     = []
    dates           = []
    future_dates    = []

    if request.method == 'POST':

        symbol = request.POST.get('symbol')

        # LOAD MODELS
        encoder  = joblib.load("models/symbol_encoder.pkl")
        scaler_X = joblib.load("models/scaler_X.pkl")
        scaler_y = joblib.load("models/scaler_y.pkl")
        lr_model = joblib.load("models/lr_model.pkl")

        # ENCODE SYMBOL
        symbol_encoded = encoder.transform([symbol])[0] if symbol in encoder.classes_ else 0

        # USER INPUT → single prediction
        user_input = {
            "Symbol":     symbol_encoded,
            "Open":       float(request.POST.get("open")),
            "High":       float(request.POST.get("high")),
            "Low":        float(request.POST.get("low")),
            "Volume":     float(request.POST.get("volume")),
            "Market Cap": float(request.POST.get("marketcap"))
        }

        input_df    = pd.DataFrame([user_input])
        input_df    = input_df[['Symbol', 'Open', 'High', 'Low', 'Volume', 'Market Cap']]
        scaled_input = scaler_X.transform(input_df)

        # ── Single prediction (shown in the result box) ──
        pred = scaler_y.inverse_transform(
            lr_model.predict(scaled_input).reshape(-1, 1)
        )[0][0]

        prediction["Linear Regression"] = float(pred)

        # ── Load last 30 historical rows for the selected symbol ──
        data = pd.read_csv("media/crypto_market_data.csv")

        filtered_data = data[data['Symbol'] == symbol].copy()
        if filtered_data.empty:
            filtered_data = data.copy()

        filtered_data['Symbol'] = encoder.transform(filtered_data['Symbol'])
        filtered_data = filtered_data.tail(30)

        X = filtered_data[['Symbol', 'Open', 'High', 'Low', 'Volume', 'Market Cap']]
        y = filtered_data['Close']

        # Predict on those 30 historical rows
        X_scaled      = scaler_X.transform(X)
        y_pred_scaled = lr_model.predict(X_scaled)
        y_pred        = scaler_y.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).ravel()

        actual_list    = [float(x) for x in y.tolist()]
        predicted_list = [float(x) for x in y_pred.tolist()]

        # ── Build date labels for the 30 historical days ──
        today = datetime.today()
        dates = [
            (today - timedelta(days=29 - i)).strftime('%d %b')
            for i in range(30)
        ]

        # ── Append the user's prediction as the 31st (Today) point ──
        # The predicted line ends EXACTLY at the value in the result box.
        actual_list.append(None)
        predicted_list.append(float(pred))
        dates.append("Today ★")

        # ── Generate next 5 days forecast ──
        # Start from the user's input row and iterate forward
        future_row = input_df.copy()
        current_pred = float(pred)

        for i in range(1, 6):
            future_scaled = scaler_X.transform(future_row)
            next_pred = scaler_y.inverse_transform(
                lr_model.predict(future_scaled).reshape(-1, 1)
            )[0][0]
            next_pred = float(next_pred)
            future_list.append(next_pred)

            # Feed prediction back as next input
            future_row.at[future_row.index[0], 'Open']  = next_pred
            future_row.at[future_row.index[0], 'High']  = next_pred * 1.01
            future_row.at[future_row.index[0], 'Low']   = next_pred * 0.99
            current_pred = next_pred

        future_dates = [
            (today + timedelta(days=i)).strftime('%d %b')
            for i in range(1, 6)
        ]

    return render(request, "users/predict.html", {
        "prediction":   prediction,
        "actual":       json.dumps(actual_list),
        "predicted":    json.dumps(predicted_list),
        "future":       json.dumps(future_list),
        "dates":        json.dumps(dates),
        "future_dates": json.dumps(future_dates),
    })