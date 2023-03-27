import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.io import wavfile
from scipy.optimize import minimize
from functools import cached_property

class Signal:
    def __init__(self, duration=5, sample_rate=44100):
        self.duration    = np.float32(duration)
        self.sample_rate = np.float32(sample_rate)    
        self.num_sample  = int(self.duration * self.sample_rate)
        self.time_array  = self.compute_time_array()

    def compute_time_array(self):
        """Array of all the discrete time samples of the signal."""
        return np.linspace(0, self.duration, self.num_sample).reshape((1, self.num_sample))

    def to_int(self, arr):
        return np.int16((arr/np.max(np.abs(arr))) * 32767)

class Periodic(Signal):
    def __init__(self, frequency=440, amplitude=1.0, phase=0, **kwargs):
        super().__init__(**kwargs)
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase     = phase
        self.period    = 1 / self.frequency

    @property
    def sinusoid(self):
        return np.sin(2 * np.pi * self.frequency * self.time_array + self.phase)
    
    def plot_period(self, n_period=3):
        periods = int(self.sample_rate / self.frequency) * n_period
        x = self.time_array[0][:periods]
        y = self.samples[0][:periods]
        plt.plot(x, y)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'{self.frequency} Hz Periodic Tone Over {n_period} Periods')
        plt.show()

class SineWave(Periodic):
    @cached_property
    def samples(self):
        return self.amplitude * super().sinusoid

class SquareWave(Periodic):
    @cached_property
    def samples(self):
        return self.amplitude * np.sign(super().sinusoid)

class TriangleWave(Periodic):
    @cached_property
    def samples(self):
        return ((2 * self.amplitude) / np.pi) * np.arcsin(super().sinusoid)
    
class SawtoothWave(Periodic):
    @cached_property
    def samples(self):
        return self.amplitude * np.arctan(np.tan(np.pi * self.frequency * self.time_array + self.phase))

class ReverseSawtoothWave(SawtoothWave):
    @cached_property
    def samples(self):
        return 1 - super().samples
    
class PulseTrain(Periodic):
    def __init__(self, duty_cycle = 0.5, **kwargs):
        super().__init__(**kwargs)
        if duty_cycle < 0 or duty_cycle > 1:
            raise ValueError("Duty cycle must be between 0 and 1")
        self.duty_cycle = duty_cycle * np.pi

    @cached_property
    def samples(self):
        sawtooth = np.arctan(np.tan(np.pi * self.frequency * self.time_array))
        delayed = np.arctan(np.tan(np.pi * self.frequency * self.time_array + self.duty_cycle))
        difference = sawtooth - delayed
        return self.amplitude * (difference / max(abs(difference)))
    

class HarmonicComplex(Periodic):
    """"
    centroid_shift multiplies the centroid by a value (1 is same, 0.5 is half, 2 is twice, etc.)
    """
    def __init__(self, num_harmonics=4, centroid_shift = 1, distribution = "uniform", h_phase=0, **kwargs):
        super().__init__(**kwargs)
        self._n_harm          = np.int16(self._is_positive_number(num_harmonics, "number of harmonics"))
        self._centroid_shift  = centroid_shift
        self._harmonics_coef  = self._compute_harmonics_coef(distribution)
        self._harmonics_phase = self._compute_harmonics_phase(h_phase)

    @property
    def harmonics(self):
        return np.arange(1, self._n_harm + 1, dtype=np.float32) * self.frequency
    
    @cached_property
    def harmonics_array(self):
        h = self.harmonics.reshape(self._n_harm, 1)
        return np.sin(2 * np.pi * h * (self.time_array + self._harmonics_phase) + self.phase) * self._harmonics_coef

    @cached_property
    def samples(self):
        return np.sum(self.harmonics_array, axis=0).reshape((1, self.num_samples))     
    
    def _compute_harmonics_coef(self, distribution, ):
        if distribution not in ("uniform", "normal"):
            raise ValueError("coef must be 'uniform'(other not implemented yet)")
        
        elif distribution == "uniform":
            return np.ones((self._n_harm, 1)) / self._n_harm
        
        elif distribution == "normal":
            mean = np.mean(self.harmonics) * self._centroid_shift
            print(mean)
            sd = np.std(self.harmonics)
            prob = stats.norm(mean, sd).pdf(self.harmonics)
            prob /= np.sum(prob)
            return np.array(prob).reshape((self._n_harm, 1))

    def _compute_harmonics_phase(self, h_phase):
        if isinstance(h_phase, (int, float)): 
            return np.full((self._n_harm, 1), h_phase, dtype=np.float32)
        elif h_phase == "zero":
            return np.zeros((self._n_harm, 1))
        elif h_phase == "random":
            return np.random.uniform(0, 2*np.pi, (self._n_harm, 1))
    


if __name__ == "__main__":
    h = HarmonicComplex(distribution = "normal", centroid_shift=0.5)
    h.plot_period()


