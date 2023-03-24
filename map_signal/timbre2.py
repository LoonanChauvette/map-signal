import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.io import wavfile
from scipy.optimize import minimize
from functools import cached_property

class Signal:
    def __init__(self, duration=5, sample_rate=44100):

        # Input validation
        if self._is_positive_number(duration, "duration") > 30:
            warnings.warn("Longer durations (>30 sec) may be inefficient, proceed with caution")
        if self._is_positive_number(sample_rate, "sample_rate") not in [44100, 48000]: 
            warnings.warn("Non standard sample rate")

        self._duration    = np.float32(duration)
        self._sample_rate = np.float32(sample_rate)    
    
    @property
    def duration(self):
        """Duration of the signal in seconds."""
        return self._duration
    
    @property
    def sample_rate(self):
        """Sampling rate of the signal in Hz."""
        return self._sample_rate
    
    @property
    def num_samples(self):
        """Total number of discrete time samples in the signal."""
        return int(self._duration * self._sample_rate)

    @cached_property
    def time_array(self):
        """Array of all the discrete time samples of the signal."""
        return np.linspace(0, self.duration, self.num_samples).reshape((1, self.num_samples))

    def _is_positive_number(self, value, name):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive number")
        return value

    def to_int(self, arr):
        return np.int16((arr/np.max(np.abs(arr))) * 32767)


class Periodic(Signal):
    def __init__(self, frequency=440, amplitude=1.0, phase=0, **kwargs):
        super().__init__(**kwargs)
        self._frequency = np.float32(self._is_positive_number(frequency, "frequency"))
        self._amplitude = np.float32(self._is_positive_number(amplitude, "amplitude"))
        self._phase     = np.float32(phase)

    @property
    def frequency(self):
        """Number of periods per second in Hz."""
        return self._frequency
    
    @property
    def amplitude(self):
        return self._amplitude
    
    @property
    def phase(self):
        return self._phase
    
    @property
    def period(self):
        return 1 / self._frequency

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

class Sinusoidal(Periodic):
    @cached_property
    def samples(self):
        return self.amplitude * super().sinusoid

class Square(Periodic):
    @cached_property
    def samples(self):
        return self.amplitude * np.sign(super().sinusoid)

class Triangle(Periodic):
    @cached_property
    def samples(self):
        return ((2 * self.amplitude) / np.pi) * np.arcsin(super().sinusoid)
    
class Sawtooth(Periodic):
    @cached_property
    def samples(self):
        return self.amplitude * np.arctan(np.tan(np.pi * self.frequency * self.time_array + self.phase))

class ReverseSawtooth(Sawtooth):
    @cached_property
    def samples(self):
        return 1 - super().samples
    
class PulseTrain(Periodic):
    def __init__(self, duty_cycle = 0.5, **kwargs):
        super().__init__(**kwargs)
        if duty_cycle < 0 or duty_cycle > 1:
            raise ValueError("Duty cycle must be between 0 and 1")
        self._duty_cycle = duty_cycle * np.pi

    @property
    def duty_cycle(self):
        return self._duty_cycle

    @cached_property
    def samples(self):
        sawtooth = np.arctan(np.tan(np.pi * self.frequency * self.time_array))
        delayed = np.arctan(np.tan(np.pi * self.frequency * self.time_array + self._duty_cycle))
        difference = sawtooth - delayed
        return self.amplitude * (difference / max(abs(difference)))
    

class HarmonicComplex(Periodic):
    def __init__(self, num_harmonics=4, coef = "uniform", h_phase=0, **kwargs):
        super().__init__(**kwargs)
        self._n_harm          = np.int16(self._is_positive_number(num_harmonics, "number of harmonics"))
        self._harmonics_coef  = self._compute_harmonics_coef(coef)
        self._harmonics_phase = self._compute_harmonics_phase(h_phase)

    @cached_property
    def harmonics(self):
        h = np.arange(1, self._n_harm + 1, dtype=np.float32) * self.frequency
        return h.reshape(self._n_harm, 1)
    
    @cached_property
    def harmonics_array(self):
        return np.sin(2 * np.pi * self.harmonics * (self.time_array + self._harmonics_phase) + self.phase) * self._harmonics_coef

    @cached_property
    def samples(self):
        return np.sum(self.harmonics_array, axis=0).reshape((1, self.num_samples))     
    
    def _compute_harmonics_coef(self, coef):
        if coef != "uniform":
            raise ValueError("coef must be 'uniform'(other not implemented yet)")
        elif coef == "uniform":
            return np.ones((self._n_harm, 1)) / self._n_harm

    def _compute_harmonics_phase(self, h_phase):
        if isinstance(h_phase, (int, float)): 
            return np.full((self._n_harm, 1), h_phase, dtype=np.float32)
        elif h_phase == "zero":
            return np.zeros((self._n_harm, 1))
        elif h_phase == "random":
            return np.random.uniform(0, 2*np.pi, (self._n_harm, 1))
    


if __name__ == "__main__":
    h = PulseTrain()
    h.plot_period()
