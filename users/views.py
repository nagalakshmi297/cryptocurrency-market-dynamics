import os
import json
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from django.shortcuts import render
from django.contrib import messages
from django.http import JsonResponse
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from .data_manager import refresh_crypto_data


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

        if request.POST.get("refresh_data") == "on":
            refresh_crypto_data()

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

        # --- LSTM MODEL ---
        try:
            # Reshape for LSTM: [samples, time_steps, features]
            # For simplicity, we'll use a window of 1 (single day prediction but with LSTM architecture)
            # or we could implement a proper sequence window. 
            # Given the constraints, a window of 1 is easiest to integrate without changing the dataset structure too much.
            X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
            X_test_lstm  = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

            lstm_mod = Sequential([
                LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[1])),
                Dense(1)
            ])
            lstm_mod.compile(optimizer='adam', loss='mse')
            lstm_mod.fit(X_train_lstm, y_train_scaled, epochs=20, batch_size=32, verbose=0)
            
            lstm_pred_scaled = lstm_mod.predict(X_test_lstm)
            lstm_pred = scaler_y.inverse_transform(lstm_pred_scaled).ravel()
            lstm_mod.save("models/lstm_model.h5")

            mse  = mean_squared_error(y_test, lstm_pred)
            rmse = np.sqrt(mse)
            mae  = mean_absolute_error(y_test, lstm_pred)
            mape = mean_absolute_percentage_error(y_test, lstm_pred)
            metrics.append(["LSTM", mse, rmse, mae, mape])
        except Exception as e:
            print(f"LSTM training failed: {e}")

        # --- SELECT BEST MODEL ---
        metrics_df = pd.DataFrame(
            metrics, columns=["Model", "MSE", "RMSE", "MAE", "MAPE"]
        )
        metrics_df.to_csv("models/model_metrics.csv", index=False)
        
        # Best model is the one with lowest RMSE
        best_model_row = metrics_df.loc[metrics_df['RMSE'].idxmin()]
        best_model_name = best_model_row['Model']
        
        best_info = {
            "name": best_model_name,
            "rmse": float(best_model_row['RMSE']),
            "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open("models/best_model_info.json", "w") as f:
            json.dump(best_info, f)

    try:
        metrics_table = pd.read_csv("models/model_metrics.csv").to_html()
    except Exception:
        metrics_table = "<p>No models trained yet.</p>"

    return render(request, "users/train.html", {"metrics": metrics_table})


# ---------------------------
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

    # LOAD BEST MODEL INFO (Move out of POST to avoid UnboundLocalError)
    best_model_file = "models/best_model_info.json"
    best_model_name = "Linear Regression" # default
    if os.path.exists(best_model_file):
        with open(best_model_file, "r") as f:
            best_model_name = json.load(f).get("name", "Linear Regression")

    if request.method == 'POST':

        symbol = request.POST.get('symbol')

        # LOAD MODELS
        encoder  = joblib.load("models/symbol_encoder.pkl")
        scaler_X = joblib.load("models/scaler_X.pkl")
        scaler_y = joblib.load("models/scaler_y.pkl")
        
        # LOAD THE CHAMPION MODEL
        model_map = {
            "Linear Regression": "models/lr_model.pkl",
            "Random Forest": "models/rf_model.pkl",
            "SVM": "models/svm_model.pkl",
            "LSTM": "models/lstm_model.h5"
        }
        
        current_model_path = model_map.get(best_model_name, "models/lr_model.pkl")
        
        if current_model_path.endswith(".h5"):
            # Load Keras model
            champion_model = tf.keras.models.load_model(current_model_path)
            is_keras = True
        else:
            champion_model = joblib.load(current_model_path)
            is_keras = False

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

        # Handle prediction
        if is_keras:
            # Reshape for LSTM/Keras
            scaled_input_keras = scaled_input.reshape((scaled_input.shape[0], 1, scaled_input.shape[1]))
            pred_raw = champion_model.predict(scaled_input_keras)
        else:
            pred_raw = champion_model.predict(scaled_input)

        pred = scaler_y.inverse_transform(pred_raw.reshape(-1, 1))[0][0]

        prediction[best_model_name] = float(pred)

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
        
        if is_keras:
            X_scaled_keras = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            y_pred_scaled = champion_model.predict(X_scaled_keras)
        else:
            y_pred_scaled = champion_model.predict(X_scaled)
            
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
            
            if is_keras:
                fs_keras = future_scaled.reshape((future_scaled.shape[0], 1, future_scaled.shape[1]))
                next_pred_raw = champion_model.predict(fs_keras)
            else:
                next_pred_raw = champion_model.predict(future_scaled)
                
            next_pred = scaler_y.inverse_transform(
                next_pred_raw.reshape(-1, 1)
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
        "best_model":   best_model_name,
        "actual":       json.dumps(actual_list),
        "predicted":    json.dumps(predicted_list),
        "future":       json.dumps(future_list),
        "dates":        json.dumps(dates),
        "future_dates": json.dumps(future_dates),
    })


def live_prediction_api(request):
    """
    API endpoint for live prediction.
    Params: symbol (e.g. BTC), period (e.g. 1d), fiat (e.g. USD)
    """
    symbol = request.GET.get('symbol', 'BTC')
    period = request.GET.get('period', '1d')
    fiat   = request.GET.get('fiat', 'USD')
    
    ticker_symbol = f"{symbol}-{fiat}"
    
    # Map periods to intervals for better granularity
    interval_map = {
        '1d': '5m',
        '5d': '30m',
        '1mo': '1d',
        '1y': '1wk',
        '5y': '1mo',
        'max': '1mo'
    }
    interval = interval_map.get(period, '1d')
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return JsonResponse({'error': f'No data found for {ticker_symbol}'}, status=404)
        
        # Prepare data for graph
        prices = df['Close'].tolist()
        timestamps = [t.strftime('%Y-%m-%d %H:%M:%S') for t in df.index]
        current_price = prices[-1]
        
        # Calculate price change
        price_start = prices[0]
        price_change = current_price - price_start
        price_change_pct = (price_change / price_start) * 100 if price_start != 0 else 0
        
        # --- AUTO TRAIN & PREDICT ---
        # specifically fetch Daily data for the 5-day forecast
        daily_ticker = yf.Ticker(ticker_symbol)
        df_daily = daily_ticker.history(period='3mo', interval='1d')
        
        predictions = []
        n_lags = 5
        
        if len(df_daily) >= n_lags:
            daily_data = df_daily['Close'].values
            X_d = []
            y_d = []
            for i in range(len(daily_data) - n_lags):
                X_d.append(daily_data[i:i + n_lags])
                y_d.append(daily_data[i + n_lags])
            
            X_d = np.array(X_d)
            y_d = np.array(y_d)
            
            model_daily = RandomForestRegressor(n_estimators=100, random_state=42)
            model_daily.fit(X_d, y_d)
            
            # Predict next 5 DAYS sequentially
            current_window = daily_data[-n_lags:].tolist()
            for _ in range(5):
                last_window_arr = np.array(current_window[-n_lags:]).reshape(1, -1)
                next_pred = float(model_daily.predict(last_window_arr)[0])
                predictions.append(next_pred)
                current_window.append(next_pred)
        else:
            # Fallback
            predictions = [current_price * (1 + 0.02 * (i+1)) for i in range(5)]
            
        # --- SIGNAL GENERATION ---
        next_pred = predictions[0]
        diff_pct = ((next_pred - current_price) / current_price) * 100
        
        if diff_pct > 1.5:
            recommendation = "Strong Buy"
            signal_type = "strong-buy"
        elif diff_pct > 0.5:
            recommendation = "Buy"
            signal_type = "buy"
        elif diff_pct < -1.5:
            recommendation = "Strong Sell"
            signal_type = "strong-sell"
        elif diff_pct < -0.5:
            recommendation = "Sell"
            signal_type = "sell"
        else:
            recommendation = "Neutral / Hold"
            signal_type = "neutral"

        return JsonResponse({
            'symbol': symbol,
            'fiat': fiat,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'prices': prices,
            'timestamps': timestamps,
            'predictions': predictions,
            'recommendation': recommendation,
            'signal_type': signal_type,
            'last_updated': datetime.now().strftime('%d %b %Y, %H:%M:%S')
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)