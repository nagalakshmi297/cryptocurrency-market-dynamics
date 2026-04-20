import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def refresh_crypto_data(symbols=['BTC', 'ETH', 'XRP']):
    """
    Fetches 1 year of historical data for the given symbols from yfinance 
     and updates media/crypto_market_data.csv.
    """
    all_data = []
    
    for symbol in symbols:
        ticker_symbol = f"{symbol}-USD"
        print(f"Fetching data for {ticker_symbol}...")
        ticker = yf.Ticker(ticker_symbol)
        
        # Get 1 year of history
        df = ticker.history(period="1y", interval="1d")
        
        if df.empty:
            print(f"No data found for {ticker_symbol}")
            continue
            
        # Get current metrics for Market Cap approximation
        info = ticker.info
        current_market_cap = info.get('marketCap')
        current_price = info.get('regularMarketPrice') or info.get('previousClose')
        
        # If current_price is missing, use the last close from history
        if not current_price and not df.empty:
            current_price = df['Close'].iloc[-1]
            
        supply_ratio = 0
        if current_market_cap and current_price:
            supply_ratio = current_market_cap / current_price
            
        # Prepare rows for the CSV
        for index, row in df.iterrows():
            date_str = index.strftime('%d-%m-%Y')
            close_price = row['Close']
            
            # Approximate historical market cap
            market_cap = close_price * supply_ratio if supply_ratio > 0 else 0
            
            all_data.append({
                'Date': date_str,
                'Symbol': symbol,
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Close': close_price,
                'Volume': row['Volume'],
                'Market Cap': market_cap
            })
            
    if not all_data:
        return False
        
    # Convert to DataFrame and save
    new_df = pd.DataFrame(all_data)
    
    # Ensure media directory exists
    if not os.path.exists("media"):
        os.makedirs("media")
        
    csv_path = "media/crypto_market_data.csv"
    new_df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with {len(new_df)} rows.")
    return True

if __name__ == "__main__":
    refresh_crypto_data()
