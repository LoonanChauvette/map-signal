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
    def __init__(self, n_harmonics=4, h_phase=0, **kwargs):
        super().__init__(**kwargs)
        self.n_harm      = n_harmonics 
        self.harm_index  = np.arange(1, self.n_harm + 1)  # ex. [1, 2, 3, ..., n_harm]
        self.harm_coef   = np.ones((self.n_harm, 1)) / self.n_harm # shape (n_harm, 1)
        self.harm_phase  = self.compute_harmonics_phase(h_phase)   # shape (n_harm, 1)
        self.harm_freqs  = self.compute_harmonics_frequencies()    # shape (n_harm, 1)
        self.harm_matrix = self.compute_harmonics_matrix() # shape (n_harm, num_sample)
        self.samples     = self.compute_samples()  # shape (1, num_sample)

    def uniform_coef(self):
        self.harm_coef = np.ones((self.n_harm, 1)) / self.n_harm 
    
    def normal_coef(self, mean_harmonic=None, sd_scaling=1):
        mean = np.median(self.harm_index) if mean_harmonic is None else mean_harmonic
        sd   = np.std(self.harm_index) * sd_scaling
        coef = stats.norm(mean, sd).pdf(self.harm_index)
        coef /= sum(coef)
        self.harm_coef = coef.reshape((self.n_harm, 1))
    
    def compute_harmonics_frequencies(self):
        harmonic_frequencies = self.harm_index * self.frequency
        return harmonic_frequencies.reshape((self.n_harm, 1))

    def compute_samples(self):
        return np.sum(self.harm_matrix, axis=0).reshape((1, self.num_sample))     

    def compute_harmonics_phase(self, h_phase):
        if isinstance(h_phase, (int, float)): 
            return np.full((self.n_harm, 1), h_phase, dtype=np.float32)
        elif h_phase == "zero":
            return np.zeros((self.n_harm, 1))
        elif h_phase == "random":
            return np.random.uniform(0, 2*np.pi, (self.n_harm, 1))
    
    def compute_harmonics_matrix(self):
        time_phase_matrix = self.harm_phase + self.time_array 
        return np.sin(2*np.pi * self.harm_freqs * time_phase_matrix + self.phase) * self.harm_coef


if __name__ == "__main__":
    h = HarmonicComplex()
    print(h.harm_freqs)
    h.plot_period()


