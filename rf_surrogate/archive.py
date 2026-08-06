from typing import Deque, List, Tuple

from collections import deque
import numpy as np

from graph import Edge
from rf_surrogate.features import extract_features


class SurrogateArchive:
    def __init__(self, maxlen: int = 5000) -> None:
        self._samples: Deque[Tuple[Tuple[Edge, ...], float]] = deque(maxlen=maxlen)

    def add(self, edges_tuple: Tuple[Edge, ...], fitness_error: float) -> None:
        self._samples.append((edges_tuple, fitness_error))

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        for chrom, fitness in self._samples:
            X_list.append(extract_features(chrom))
            y_list.append(fitness)
        return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64)

    def get_raw_samples(self) -> List[Tuple[Tuple[Edge, ...], float]]:
        return list(self._samples)

    def size(self) -> int:
        return len(self._samples)
