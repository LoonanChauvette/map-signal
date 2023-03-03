"""
Code adapted from : https://crackedbassoon.com/writing/ripple-sounds
"""

import numpy as np


def ripple_sound(n_sinus: int, f_min: float, f_max: float, duration: float = 0.5, sr: int = 44100,
                 velocity: float = 0, density: float = 1, depth: float = 1, phase: float = 0,
                 noise: str = "white"):
    """
    Generates moving ripple sounds.

    Args:
        n_sinus (int): number of sinusoids to generate.
        f_min (float): minimum frequency of the sinusoids in Hz.
        f_max (float): maximum frequency of the sinusoids in Hz.
        duration (float): total duration of the ripple sound.
        sr (int, optional): sampling rate of the sinusoids in Hz. Defaults to 44100.
        velocity (float): direction and slope of the ripples (w).
        density (float): modulation rate of the ripples (omega).
        depth (float): modulation depth of the ripples (delta) (if set to 0, no ripple just noise).
        phase (float): starting phase of the ripple
        noise (str): type of noise used, either "white" (flat long term average) or "pink" (equal energy per octave)

    Not implemented:
        accepting time varying functions for velocity, density, and depth

    Returns:
        envelope
        ripples
    """

    # Generate and array of log_spaced sinusoids with random phases
    sinus, freqs, times = sinusoid_array(n_sinus, f_min, f_max, duration, sr)

    # Compute the indices and phases of the spectral amplitude modulations
    spectral_indices = np.log2(freqs / f_min) * density
    spectral_phases = times * velocity

    # Compute the envelope of the ripples
    envelope = 1 + depth * np.sin(2 * np.pi * (spectral_indices + spectral_phases) + phase)

    # Compute ripples and scales the long-term average spectrum energy
    ripples = envelope * sinus
    if noise == "pink":
        ripples /= np.sqrt(freqs)

    # Sum over the rows to get a one dimensional array and convert to 16bit audio
    sound = to_int(ripples.sum(axis=0))

    return envelope, sound


def sinusoid_array(n_sinus: int, f_min: float, f_max: float, duration: float = 0.5, sr: int = 44100):
    """
    Generates an array of sinusoids with random phases.

    Args:
        n_sinus (int): number of sinusoids to generate.
        f_min (float): minimum frequency of the sinusoids in Hz.
        f_max (float): maximum frequency of the sinusoids in Hz.
        duration (float): total duration of the ripple sound.
        sr (int, optional): sampling rate of the sinusoids in Hz. Defaults to 44100.

    Returns:
        sinus (np.array): shape (n_sinus x n_sample), each row is a sinusoid with random phase.
        freqs (np.array): shape (n_sinus x 1), frequency value of each row
        times (np.array): shape (1 x n_sample), time value of each column
    """

    n_sample = int(duration * sr)  # total number of samples
    times = np.linspace(0, duration, num=n_sample).reshape((1, n_sample))  # Shape (1 x n_sample)
    freqs = np.geomspace(f_min, f_max, num=n_sinus).reshape((n_sinus, 1))  # Shape (n_sinus x 1)
    phases = 2 * np.pi * np.random.random((n_sinus, 1))  # Shape (n_sinus x 1)

    sinus = np.sin(2 * np.pi * freqs * times + phases)  # Shape (n_sinus x n_sample)

    return sinus, freqs, times

def to_int(waveform: np.array):
    """
    Converts an array of floats to 16bit integers

    Args:
        waveform: np.array

    Returns:
        waveform: np.array
    """
    m = np.max(np.abs(waveform))
    return np.int16((waveform / m) * 32767)
