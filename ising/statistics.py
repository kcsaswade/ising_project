from collections import deque


class RunningStats:
    def __init__(self, window_size=300):
        self.window_size = window_size
        self.E_vals = deque(maxlen=window_size)
        self.M_vals = deque(maxlen=window_size)

    def add(self, E: float, M: float):
        self.E_vals.append(E)
        self.M_vals.append(M)

    def clear(self):
        self.E_vals.clear()
        self.M_vals.clear()

    def _mean(self, data):
        return sum(data) / len(data) if data else 0.0

    def _mean_sq(self, data):
        return sum(x * x for x in data) / len(data) if data else 0.0

    def heat_capacity(self, T: float, N_spins: int):
        if len(self.E_vals) < 2:
            return 0.0
        E_mean = self._mean(self.E_vals)
        E2_mean = self._mean_sq(self.E_vals)
        return N_spins * (E2_mean - E_mean * E_mean) / (T * T)

    def susceptibility(self, T: float, N_spins: int):
        if len(self.M_vals) < 2:
            return 0.0
        M_mean = self._mean(self.M_vals)
        M2_mean = self._mean_sq(self.M_vals)
        return N_spins * (M2_mean - M_mean * M_mean) / T