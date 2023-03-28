import numpy as np
import pytest
from map_signal.timbre import *

def test_signal_init():
    signal = Signal(duration=0.5, sample_rate=44100)
    assert signal.duration == 0.5
    assert signal.sample_rate == 44100
    assert signal.num_sample == 22050
    assert signal.time_array.shape == (1, 22050)

def test_sinewave_init():
    x = SineWave(frequency=440, amplitude=0.5, phase=np.pi/2, duration=0.5, sample_rate=44100)
    assert x.duration == 0.5
    assert x.sample_rate == 44100
    assert x.num_sample == 22050
    assert x.time_array.shape == (1, 22050)
    assert x.frequency == 440
    assert x.amplitude == 0.5
    assert x.phase == np.pi/2
    assert x.period == 1 / 440
    assert isinstance(x.samples, np.ndarray)
    assert x.samples.dtype == np.float64
    assert x.samples.shape == (1, 22050)

def test_squarewave_init():
    x = SquareWave(frequency=440, amplitude=0.5, phase=np.pi/2, duration=0.5, sample_rate=44100)
    assert x.duration == 0.5
    assert x.sample_rate == 44100
    assert x.num_sample == 22050
    assert x.time_array.shape == (1, 22050)
    assert x.frequency == 440
    assert x.amplitude == 0.5
    assert x.phase == np.pi/2
    assert x.period == 1 / 440
    assert isinstance(x.samples, np.ndarray)
    assert x.samples.dtype == np.float64
    assert x.samples.shape == (1, 22050)

def test_trianglewave_init():
    x = TriangleWave(frequency=440, amplitude=0.5, phase=np.pi/2, duration=0.5, sample_rate=44100)
    assert x.duration == 0.5
    assert x.sample_rate == 44100
    assert x.num_sample == 22050
    assert x.time_array.shape == (1, 22050)
    assert x.frequency == 440
    assert x.amplitude == 0.5
    assert x.phase == np.pi/2
    assert x.period == 1 / 440
    assert isinstance(x.samples, np.ndarray)
    assert x.samples.dtype == np.float64
    assert x.samples.shape == (1, 22050)

def test_sawtoothwave_init():
    x = SawtoothWave(frequency=440, amplitude=0.5, phase=np.pi/2, duration=0.5, sample_rate=44100)
    assert x.duration == 0.5
    assert x.sample_rate == 44100
    assert x.num_sample == 22050
    assert x.time_array.shape == (1, 22050)
    assert x.frequency == 440
    assert x.amplitude == 0.5
    assert x.phase == np.pi/2
    assert x.period == 1 / 440
    assert isinstance(x.samples, np.ndarray)
    assert x.samples.dtype == np.float64
    assert x.samples.shape == (1, 22050)

def test_reversesawtoothwave_init():
    x = ReverseSawtoothWave(frequency=440, amplitude=0.5, phase=np.pi/2, duration=0.5, sample_rate=44100)
    assert x.duration == 0.5
    assert x.sample_rate == 44100
    assert x.num_sample == 22050
    assert x.time_array.shape == (1, 22050)
    assert x.frequency == 440
    assert x.amplitude == 0.5
    assert x.phase == np.pi/2
    assert x.period == 1 / 440
    assert isinstance(x.samples, np.ndarray)
    assert x.samples.dtype == np.float64
    assert x.samples.shape == (1, 22050)

def test_pulsetrain_init():
    x = PulseTrain(duty_cycle=0.5, frequency=440, amplitude=0.5, phase=np.pi/2, duration=0.5, sample_rate=44100)
    assert x.duration == 0.5
    assert x.sample_rate == 44100
    assert x.num_sample == 22050
    assert x.time_array.shape == (1, 22050)
    assert x.frequency == 440
    assert x.amplitude == 0.5 
    assert x.phase == np.pi/2
    assert x.period == 1 / 440
    assert x.duty_cycle == 0.5 * np.pi
    assert isinstance(x.samples, np.ndarray)
    assert x.samples.dtype == np.float64
    assert x.samples.shape == (1, 22050)

def test_pulsetrain_error():
    with pytest.raises(ValueError):
        x = PulseTrain(duty_cycle=-0.5)
        x = PulseTrain(duty_cycle=2)

def test_harmoniccomplex_init():
    x = HarmonicComplex(frequency=440, amplitude=0.5, phase=np.pi/2, duration=0.5, sample_rate=44100,
                        n_harmonics=4, h_phase=3)
    assert x.duration == 0.5
    assert x.sample_rate == 44100
    assert x.num_sample == 22050
    assert x.time_array.shape == (1, 22050)
    assert x.frequency == 440
    assert x.amplitude == 0.5
    assert x.phase == np.pi/2
    assert x.period == 1 / 440

    assert x.n_harm == 4
    assert np.array_equal(x.harm_index, np.array([1, 2, 3, 4]))
    assert np.array_equal(x.harm_coef, np.full((4, 1), 0.25))
    assert np.array_equal(x.harm_phase, np.full((4, 1), 3.0))
    assert np.array_equal(x.harm_freqs, np.array([[440], [880], [1320], [1760]]))
    x_time_phase_matrix = x.harm_phase + x.time_array
    assert x_time_phase_matrix.shape == (4, 22050) 
    assert np.allclose(x.harm_matrix, np.sin(2*np.pi * x.harm_freqs * x_time_phase_matrix + x.phase) * x.harm_coef)
    assert x.harm_matrix.shape == (4, 22050)
    assert isinstance(x.samples, np.ndarray)
    assert x.samples.dtype == np.float64
    assert x.samples.shape == (1, 22050)