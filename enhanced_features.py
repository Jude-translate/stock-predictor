import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from catboost import CatBoostRegressor
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator

class EnhancedIndicators:
    """
    Class for adding advanced technical indicators to a stock data DataFrame.
    """

    @staticmethod
    def calculate_all_indicators(df):
        """
        Add a comprehensive set of indicators using the `ta` library.
        """
        df = add_all_ta_features(
            df, open_col='Open', high_col='High', low_col='Low', close_col='Close', volume_col='Volume', fillna=True
        )
        return df

    @staticmethod
    def calculate_custom_indicators(df):
        """
        Add additional, custom technical indicators.
        """
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_middle'] = bb.bollinger_mavg()

        # Relative Strength Index (RSI)
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi.rsi()

        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['Stochastic_K'] = stoch.stoch()
        df['Stochastic_D'] = stoch.stoch_signal()

        # Moving Averages
        df['SMA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()

        # MACD
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()

        return df.dropna()

class AdditionalModels:
    """
    Class for additional predictive models.
    """

    @staticmethod
    def train_extra_trees(X_train, y_train, X_test):
        """
        Train and predict with Extra Trees Regressor.
        """
        model = ExtraTreesRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return predictions

    @staticmethod
    def train_catboost(X_train, y_train, X_test):
        """
        Train and predict with CatBoost Regressor.
        """
        model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=8, verbose=0)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return predictions

# Example Usage
if __name__ == "__main__":
    # Load sample data
    df = pd.DataFrame({
        'Open': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100,
        'Low': np.random.rand(100) * 100,
        'Close': np.random.rand(100) * 100,
        'Volume': np.random.randint(100, 1000, size=100)
    })

    # Add indicators
    indicator_processor = EnhancedIndicators()
    df = indicator_processor.calculate_custom_indicators(df)

    # Split data for training additional models
    features = df.drop(columns=['Close'])
    target = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train models and predict
    model_processor = AdditionalModels()
    et_predictions = model_processor.train_extra_trees(X_train, y_train, X_test)
    cb_predictions = model_processor.train_catboost(X_train, y_train, X_test)

    print("Extra Trees Predictions:", et_predictions)
    print("CatBoost Predictions:", cb_predictions)
