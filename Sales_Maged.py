import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

class SalesForecastingModel:
    def __init__(self, df, target_col='Sales Count'):
        self.df = df
        self.target_col = target_col
        self.results = []

    def mean_absolute_percentage_error(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        epsilon = 1e-10
        y_true = np.where(y_true == 0, epsilon, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def run_models(self):
        # Drop the target column to create features
        X = self.df.drop(columns=[self.target_col])

        # Drop Timestamp if not used as a feature
        if 'Timestamp' in X.columns:
            X = X.drop(columns=['Timestamp'])

        y = self.df[self.target_col]

        tscv = TimeSeriesSplit(n_splits=5)
        r2_scores = []
        mape_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            r2 = r2_score(y_val, y_pred)
            mape = self.mean_absolute_percentage_error(y_val, y_pred)

            r2_scores.append(r2)
            mape_scores.append(mape)

            print(f"Fold {fold + 1}: R2 = {r2:.4f}, MAPE = {mape:.2f}%")

        self.results = {
            'R2': r2_scores,
            'MAPE': mape_scores
        }
