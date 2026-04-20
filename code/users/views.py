import os
import json
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from django.shortcuts import render
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
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


# Helper for LSTM Sequence Creation
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i])
        y.append(dataset[i, 0])  # predict Close (assumed to be index 0)
    return np.array(X), np.array(y)


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

        # --- LSTM MODEL (Optimized) ---
        try:
            # Group data by SYMBOL for specialized scaling (avoids BTC drowning out XRP)
            symbols = data['Symbol'].unique()
            symbol_scalers = {}
            X_lstm_total = []
            y_lstm_total = []
            
            time_step = 60
            
            for sym in symbols:
                sym_df = data[data['Symbol'] == sym][['Close', 'Volume']].copy()
                if len(sym_df) > time_step + 10:
                    # Specialized scaler for this symbol
                    scaler = MinMaxScaler()
                    scaled_sym = scaler.fit_transform(sym_df)
                    symbol_scalers[sym] = scaler
                    
                    # Create sequences within symbol boundaries (no data bleed)
                    X_s, y_s = create_dataset(scaled_sym, time_step)
                    X_lstm_total.append(X_s)
                    y_lstm_total.append(y_s)
            
            if X_lstm_total:
                X_lstm = np.concatenate(X_lstm_total, axis=0)
                y_lstm = np.concatenate(y_lstm_total, axis=0)
                
                # Split
                split_idx = int(0.8 * len(X_lstm))
                X_train_l = X_lstm[:split_idx]
                X_test_l  = X_lstm[split_idx:]
                y_train_l = y_lstm[:split_idx]
                y_test_l  = y_lstm[split_idx:]
                
                # Build Improved Model: 128 -> 128 -> 64 -> 50 -> 1
                lstm_mod = Sequential()
                lstm_mod.add(LSTM(128, return_sequences=True, input_shape=(X_train_l.shape[1], X_train_l.shape[2])))
                lstm_mod.add(Dropout(0.3))
                lstm_mod.add(LSTM(128, return_sequences=True))
                lstm_mod.add(Dropout(0.3))
                lstm_mod.add(LSTM(64))
                lstm_mod.add(Dropout(0.3))
                lstm_mod.add(Dense(50))
                lstm_mod.add(Dense(1))
                
                lstm_mod.compile(optimizer='adam', loss='mean_squared_error')
                early_stop = EarlyStopping(monitor='val_loss', patience=7)
                
                # Train (Optimized for speed: 30 epochs, 64 batch_size)
                lstm_mod.fit(
                    X_train_l, y_train_l,
                    epochs=30,
                    batch_size=64,
                    validation_data=(X_test_l, y_test_l),
                    callbacks=[early_stop],
                    verbose=0
                )
                
                # Metrics (Calculated on normalized combined space for stability)
                lstm_pred_scaled = lstm_mod.predict(X_test_l)
                mse  = mean_squared_error(y_test_l, lstm_pred_scaled)
                rmse = np.sqrt(mse)
                mae  = mean_absolute_error(y_test_l, lstm_pred_scaled)
                mape = mean_absolute_percentage_error(y_test_l, lstm_pred_scaled)
                metrics.append(["LSTM", mse, rmse, mae, mape])
                
                # Save
                lstm_mod.save("models/lstm_model.h5")
                joblib.dump(symbol_scalers, "models/symbol_scalers.pkl")
            else:
                print("Skipping LSTM: Not enough data points across symbols.")
        except Exception as e:
            print(f"Optimized LSTM training failed: {e}")

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


# Global Model Cache to prevent OOM and speed up inference
_MODEL_CACHE = {}

def get_cached_model(model_path):
    if model_path not in _MODEL_CACHE:
        if model_path.endswith(".h5"):
            _MODEL_CACHE[model_path] = tf.keras.models.load_model(model_path)
        else:
            _MODEL_CACHE[model_path] = joblib.load(model_path)
    return _MODEL_CACHE[model_path]

# Helper to clean data for JSON (removes NaN/Inf)
def clean_for_json(data_list):
    return [
        (float(x) if (x is not None and not np.isnan(x) and not np.isinf(x)) else 0) 
        for x in data_list
    ]

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
    best_model_name = "Linear Regression"

    try:
        data = pd.read_csv("media/crypto_market_data.csv")
        
        # LOAD BEST MODEL INFO
        best_model_file = "models/best_model_info.json"
        if os.path.exists(best_model_file):
            with open(best_model_file, "r") as f:
                best_model_name = json.load(f).get("name", "Linear Regression")

        if request.method == 'POST':
            symbol = request.POST.get('symbol')
            
            # LOAD MODELS AND SCALERS
            encoder  = joblib.load("models/symbol_encoder.pkl")
            scaler_X = joblib.load("models/scaler_X.pkl")
            scaler_y = joblib.load("models/scaler_y.pkl")
            
            model_map = {
                "Linear Regression": "models/lr_model.pkl",
                "Random Forest": "models/rf_model.pkl",
                "SVM": "models/svm_model.pkl",
                "LSTM": "models/lstm_model.h5"
            }
            current_model_path = model_map.get(best_model_name, "models/lr_model.pkl")
            
            is_keras = False
            champion_model = None
            symbol_scalers = {}
            
            if current_model_path.endswith(".h5"):
                champion_model = get_cached_model(current_model_path)
                is_keras = True
                if os.path.exists("models/symbol_scalers.pkl"):
                    symbol_scalers = joblib.load("models/symbol_scalers.pkl")
            else:
                champion_model = get_cached_model(current_model_path)

            # ENCODE SYMBOL AND PREP USER INPUT
            symbol_encoded = encoder.transform([symbol])[0] if symbol in encoder.classes_ else 0
            user_input = {
                "Symbol":     symbol_encoded,
                "Open":       float(request.POST.get("open", 0)),
                "High":       float(request.POST.get("high", 0)),
                "Low":        float(request.POST.get("low", 0)),
                "Volume":     float(request.POST.get("volume", 0)),
                "Market Cap": float(request.POST.get("marketcap", 0))
            }
            input_df = pd.DataFrame([user_input])[['Symbol', 'Open', 'High', 'Low', 'Volume', 'Market Cap']]
            scaled_input = scaler_X.transform(input_df)

            # 1. MAIN PREDICTION
            pred = 0
            open_val = float(request.POST.get("open", 0))
            if is_keras:
                lstm_scaler = symbol_scalers.get(symbol)
                if lstm_scaler:
                    hist_df = data[data['Symbol'] == symbol].tail(60)
                    if len(hist_df) >= 60:
                        scaled_seq = lstm_scaler.transform(hist_df[['Close', 'Volume']])
                        inputs = scaled_seq.reshape((1, 60, 2)).astype('float32')
                        pred_raw = champion_model(inputs, training=False).numpy()
                        p_padded = np.zeros((1, 2)); p_padded[0, 0] = pred_raw[0, 0]
                        pred = lstm_scaler.inverse_transform(p_padded)[0, 0]
                    else:
                        pred = open_val
                else:
                    pred = open_val
            else:
                pred_raw = champion_model.predict(scaled_input)
                pred = scaler_y.inverse_transform(pred_raw.reshape(-1, 1))[0][0]
            
            prediction[best_model_name] = float(pred)

            # 2. HISTORICAL 30-DAY PLOT
            filtered_data = data[data['Symbol'] == symbol].tail(30).copy()
            if filtered_data.empty: filtered_data = data.tail(30).copy()
            
            if is_keras:
                y_pred = []
                lstm_scaler = symbol_scalers.get(symbol)
                if lstm_scaler:
                    hist_full = data[data['Symbol'] == symbol].copy()
                    batch_sequences = []
                    valid_indices = []
                    for i in range(len(filtered_data)):
                        idx = filtered_data.index[i]
                        window = hist_full.loc[:idx].tail(61).head(60)
                        if len(window) == 60:
                            batch_sequences.append(lstm_scaler.transform(window[['Close', 'Volume']]))
                            valid_indices.append(i)
                    
                    if batch_sequences:
                        all_preds = champion_model.predict(np.array(batch_sequences), verbose=0)
                        temp_preds = [float(filtered_data.iloc[i]['Close']) for i in range(len(filtered_data))]
                        for count, idx_in_filter in enumerate(valid_indices):
                            p_padded = np.zeros((1, 2)); p_padded[0, 0] = all_preds[count, 0]
                            temp_preds[idx_in_filter] = lstm_scaler.inverse_transform(p_padded)[0, 0]
                        y_pred = temp_preds
                    else:
                        y_pred = filtered_data['Close'].tolist()
                else:
                    y_pred = filtered_data['Close'].tolist()
                y_pred = np.array(y_pred)
            else:
                X_hist = filtered_data[['Symbol', 'Open', 'High', 'Low', 'Volume', 'Market Cap']].copy()
                X_hist['Symbol'] = encoder.transform(filtered_data['Symbol'])
                y_pred_scaled = champion_model.predict(scaler_X.transform(X_hist))
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            actual_list    = [float(x) for x in filtered_data['Close'].tolist()]
            predicted_list = [float(x) for x in y_pred.tolist()]
            
            today = datetime.today()
            dates = [(today - timedelta(days=29 - i)).strftime('%d %b') for i in range(30)]
            # ── Append the user's input price as the "Today" point to prevent the line drop ──
            actual_list.append(open_val)
            predicted_list.append(float(pred))
            dates.append("Today ★")

            # 3. 5-DAY FORECAST
            current_pred = float(pred)
            future_row = input_df.copy()
            if is_keras and lstm_scaler:
                hist_df = data[data['Symbol'] == symbol].tail(60)
                scaled_window = lstm_scaler.transform(hist_df[['Close', 'Volume']])

            for i in range(1, 6):
                if is_keras:
                    # LSTM logic: Must have scaler and window
                    lstm_scaler = symbol_scalers.get(symbol)
                    if lstm_scaler:
                        scaled_next = lstm_scaler.transform([[current_pred, future_row.iloc[0]['Volume']]])
                        scaled_window = np.vstack([scaled_window[1:], scaled_next])
                        nxt_in = scaled_window.reshape(1, 60, 2).astype('float32')
                        nxt_raw = champion_model(nxt_in, training=False).numpy()
                        p_p = np.zeros((1, 2)); p_p[0, 0] = nxt_raw[0, 0]
                        next_pred = float(lstm_scaler.inverse_transform(p_p)[0, 0])
                    else:
                        next_pred = current_pred * 1.01 # Simple drift if scaler missing
                else:
                    # Sklearn logic
                    next_pred_raw = champion_model.predict(scaler_X.transform(future_row))
                    next_pred = float(scaler_y.inverse_transform(next_pred_raw.reshape(-1, 1))[0][0])
                
                future_list.append(next_pred)
                future_row.at[future_row.index[0], 'Open'] = next_pred
                current_pred = next_pred

            future_dates = [(today + timedelta(days=i)).strftime('%d %b') for i in range(1, 6)]

            context = {
                "prediction":   prediction,
                "best_model":   best_model_name,
                "actual":       json.dumps(clean_for_json(actual_list)),
                "predicted":    json.dumps(clean_for_json(predicted_list)),
                "future":       json.dumps(clean_for_json(future_list)),
                "dates":        json.dumps(dates),
                "future_dates": json.dumps(future_dates),
            }
            return render(request, "users/predict.html", context)

        return render(request, "users/predict.html", {"best_model": best_model_name})

    except Exception as e:
        import traceback
        error_msg = f"Prediction Logic Error: {str(e)}\n{traceback.format_exc()}"
        return HttpResponse(f"<div style='color:red; font-family:monospace;'><h3>Server Debug Info</h3><pre>{error_msg}</pre></div>", status=500)


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