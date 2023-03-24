import numpy as np
from map_signal.timbre2 import *

def test_signal_init():
    signal = Signal(duration=0.5, sample_rate=44100)
    assert signal.duration == 0.5
    assert signal.sample_rate == 44100
    assert signal.num_samples == 22050
    assert signal.time_array.shape == (1, 22050)

def test_sinusoid_init():
    sinusoid = Sinusoidal(frequency=440, amplitude=0.5, phase=np.pi/2, duration=0.5, sample_rate=44100)
    assert sinusoid.duration == 0.5
    assert sinusoid.sample_rate == 44100
    assert sinusoid.num_samples == 22050
    assert sinusoid.time_array.shape == (1, 22050)
    assert sinusoid.frequency == 440
    assert sinusoid.amplitude == 0.5

    assert isinstance(sinusoid.samples, np.ndarray)
    assert sinusoid.samples.dtype == np.float64
    assert sinusoid.samples.shape == (1, 22050)
