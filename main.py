import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

class StockPricePredictor:
    def __init__(self, ticker='PETR4.SA', period='1y', interval='1d', time_steps=30, future_steps=10):
        self.ticker = ticker
        self.period = period
        self.interval = interval
        self.time_steps = time_steps
        self.future_steps = future_steps

        self.download_data()
        self.preprocess_data()
        self.create_datasets()
        self.build_model()
        self.train_model()
        self.evaluate_model()
        self.predict_future()

    def download_data(self):
        self.data = yf.download(self.ticker, period=self.period, interval=self.interval)['Close']

    def preprocess_data(self):
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1, 1))

    def create_datasets(self):
        X, y = [], []
        for i in range(len(self.scaled_data) - self.time_steps - self.future_steps):
            X.append(self.scaled_data[i:(i + self.time_steps)])
            y.append(self.scaled_data[i + self.time_steps: i + self.time_steps + self.future_steps])
        split = int(0.8 * len(X))
        self.X_train, self.y_train = np.array(X[:split]), np.array(y[:split])
        self.X_test, self.y_test = np.array(X[split:]), np.array(y[split:])

    def build_model(self):
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=self.future_steps)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, epochs=50, batch_size=32, verbose=1):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def evaluate_model(self):
        loss = self.model.evaluate(self.X_test, self.y_test, verbose=1)
        print(f'Mean Squared Error on test data: {loss}')

    def predict_future(self):
        predictions = []
        current_batch = self.scaled_data[-self.time_steps:].reshape((1, self.time_steps, 1))
        for i in range(self.future_steps):
            future_pred = self.model.predict(current_batch)[0]
            predictions.append(future_pred)
            current_batch = np.append(current_batch[:, 1:, :], future_pred[-1].reshape((1, 1, 1)), axis=1)

        future_predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        print("Previsões para os próximos 10 dias:")
        for i, pred in enumerate(future_predictions):
            print(f"Dia {i + 1}: {pred[0]}")

app = StockPricePredictor()
