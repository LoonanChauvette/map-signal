import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.io import wavfile
from scipy.optimize import minimize

def get_harmonics_array(frequency, num_harmonics):
    return np.arange(1, num_harmonics + 1, dtype=np.float32) * frequency

def discrete_normal_distribution(frequency, num_harmonics, mean=None):
    """Generate the coefficients of a harmonic complex using a discrete normal distribution.

    Args:
        frequency (int|float): Fundamental frequency of the harmonic complex.
        num_harmonics (int|float): Number of harmonics of the harmonic complex.
        mean (int|float, optional): Spectral centroid of the harmonic complex, in units of harmonic number. 
        e.g, if mean = 1, then the spectral centroid is the fundamental frequency.
             if mean = 4, then the spectral centroid is the fourth harmonic.
        Defaults to None, will use the median harmonic.

    Returns:
        tuple (np.array, np.array): harmonic values and coefficient values of the harmonic complex
    """
    array = np.arange(1, num_harmonics + 1, dtype=np.float32)
    mean  = np.median(array) if mean is None else mean
    coefficients = stats.norm(mean, np.std(array)).pdf(array)
    coefficients /= sum(coefficients)
    harmonics = array * frequency
    return (harmonics, coefficients)

def discrete_chisquared_distribution(array, df = 4):
    array = array / np.min(array)
    probabilities = [stats.chi2(df).pdf(num) for num in array]
    probabilities = [p/sum(probabilities) for p in probabilities] 
    return probabilities

def discrete_exponential_distribution(array, lambd = 0.5):
    array = array / np.min(array)
    probabilities = [stats.expon(scale=1/lambd).pdf(num) for num in array]
    probabilities = [p/sum(probabilities) for p in probabilities] 
    return probabilities

def generate_waveform(harmonics, fourier_coeffs, sample_rate=44100, duration=5):

    length = int(sample_rate * duration)
    discrete_times = np.linspace(0, duration, length, endpoint=False)
    waveform = np.zeros(length)
    
    for freq, coeff in zip(harmonics, fourier_coeffs):
        waveform += (np.sin(2 * np.pi * freq * discrete_times) * coeff)
    
    # Normalize the waveform between -1 and 1
    waveform /= np.max(np.abs(waveform))
    
    # Convert the waveform to a 16-bit integer format
    waveform_int = np.int16(waveform * 32767)

    return waveform_int

def odd_even_ratio(coefficients):
    # Odds and evens are reversed with respect to the modulus, since python index starts at 0
    odd_sum = sum(coeff for i, coeff in enumerate(coefficients) if i % 2 == 0)
    even_sum = sum(coeff for i, coeff in enumerate(coefficients) if i % 2 != 0)
    ratio = odd_sum / even_sum if even_sum != 0 else float('inf')
    return ratio

def objective_function(coefficients, desired_ratio):
    return abs(odd_even_ratio(coefficients) - desired_ratio)

def modify_coefficients(coefficients, desired_ratio):
    result = minimize(objective_function, coefficients, args=(desired_ratio,))
    return result.x
    
if __name__ == "__main__":
    fundamental = 100
    num_harmonics = 10
    harmonics, coefs = discrete_normal_distribution(fundamental, num_harmonics, mean=8)
    #m_coef = modify_coefficients(coefs, 1)


    plt.bar(harmonics, coefs, width=50)
    plt.xlabel("Array")
    plt.ylabel("Coefficient")
    plt.show()

