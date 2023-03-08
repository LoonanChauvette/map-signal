import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.io import wavfile
from scipy.optimize import minimize

class Signal:
    def __init__(self, duration=5, sample_rate=44100):
        self._duration    = self._validate_return_duration(duration)
        self._sample_rate = self._validate_return_sample_rate(sample_rate)
        self._num_samples = self._compute_num_samples()
        self._time_array  = self._compute_time_array()
    
    @property
    def duration(self):
        return self._duration
    
    @duration.setter
    def duration(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("duration must be a positive number")
        self._duration    = self._validate_return_duration(value)
        self._num_samples = self._compute_num_samples()
        self._time_array  = self._compute_time_array()
    
    @property
    def sample_rate(self): 
        return self._sample_rate
    
    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = self._validate_return_sample_rate(value)
        self._num_samples = self._compute_num_samples()
        self._time_array  = self._compute_time_array()

    @property
    def num_samples(self):
        return self._num_samples
    
    @property
    def time_array(self):
        return self._time_array

    def _compute_num_samples(self):
        return int(self._duration * self._sample_rate)
    
    def _compute_time_array(self):
        arr = np.arange(0, self._num_samples) / self._sample_rate
        return arr.reshape(1, self._num_samples)
    
    def _validate_return_duration(self, duration):
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError("duration must be a positive number")
        return float(duration)
    
    def _validate_return_sample_rate(self, sample_rate):
        if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive number")
        if sample_rate not in [44100, 48000]: 
            warnings.warn("Non standard sample rate")
        return float(sample_rate)

class Sinusoid(Signal):
    def __init__(self, frequency=440, amplitude=1.0, phase=0, 
                 duration=5, sample_rate=44100):
        super().__init__(duration=duration, sample_rate=sample_rate)
        self.frequency = int(frequency)
        self.amplitude = float(amplitude)
        self.phase     = float(phase)
        self.samples   = self.generate()

    def generate(self):
        return np.sin(2 * np.pi * self.frequency * self.time_array + self.phase) * self.amplitude
    

class HarmonicComplex(Signal):
    def __init__(self, frequency=440, amplitude=1.0, phase=0, 
                 num_harmonics=4, coef = "uniform", h_phase=0,
                 duration=5, sample_rate=44100):
        super().__init__(duration=duration, sample_rate=sample_rate)
        self.frequency       = int(frequency)
        self.amplitude       = float(amplitude)
        self.phase           = float(phase)
        self.n_harm          = int(num_harmonics)
        self.harmonics       = self.get_harmonics()
        self.harmonics_coef  = self.get_coefficients(coef)
        self.harmonics_phase = self.get_phase(h_phase)
        self.harmonics_array = self.get_harmonics_array()
        self.samples         = np.sum(self.harmonics_array, axis=0)

    def get_harmonics(self):
        harmonics = np.arange(1, self.n_harm + 1, dtype=float) * self.frequency
        return harmonics.reshape(self.n_harm, 1)
    
    def get_coefficients(self, coef):
        if coef != "uniform":
            raise ValueError("coef must be 'uniform'(other not implemented yet)")
        if coef == "uniform":
            return np.ones((self.n_harm, 1)) / self.n_harm

    def get_phase(self, h_phase):
        if isinstance(h_phase, (int, float)): 
            return np.full((self.n_harm, 1), h_phase, dtype=float)
        elif h_phase == "zero":
            return np.zeros((self.n_harm, 1))
        elif h_phase == "random":
            return np.random.uniform(0, 2*np.pi, (self.n_harm, 1))

    def get_harmonics_array(self):
        return np.sin(2 * np.pi * self.harmonics * self.time_array + self.harmonics_phase) * self.harmonics_coef

if __name__ == "__main__":
    h = HarmonicComplex()
    print(h.duration)
    print(h.num_samples)
    h.duration = 1
    print(h.duration)
    print(h.num_samples)
    




