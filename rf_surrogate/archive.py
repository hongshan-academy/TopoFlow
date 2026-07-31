from collections import deque
from typing import Deque, Tuple

import numpy as np

from rf_surrogate.features import extract_features


class SurrogateArchive:
    def __init__(self, maxlen: int = 5000) -> None:
        self._samples: Deque[Tuple[Tuple[int, ...], float]] = deque(maxlen=maxlen)

    def add(self, chromosome: Tuple[int, ...], fitness_error: float) -> None:
        self._samples.append((chromosome, fitness_error))

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        for chrom, fitness in self._samples:
            X_list.append(extract_features(chrom))
            y_list.append(fitness)
        return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64)

    def size(self) -> int:
        return len(self._samples)
