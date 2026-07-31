from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class SurrogateRF:
    def __init__(self, n_estimators: int = 100, max_depth: int = 10) -> None:
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) < 2:
            return
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SurrogateRF not fitted")
        return self._model.predict(X)

    def is_ready(self) -> bool:
        return self._fitted
