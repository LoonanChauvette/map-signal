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