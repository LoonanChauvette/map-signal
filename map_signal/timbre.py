import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.io import wavfile
from scipy.optimize import minimize

def discrete_normal_distribution(array, mean, std_dev):
    probabilities = [stats.norm(mean, std_dev).pdf(num) for num in array]
    probabilities = [p/sum(probabilities) for p in probabilities] 
    return probabilities

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
    sample_rate = 44100
    duration = 5

    array = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000])
    coefs = discrete_normal_distribution(array, 1000, 300)
    m_coef = modify_coefficients(coefs, 1.5)
    waveform = generate_waveform(array, m_coef, sample_rate=sample_rate, duration=duration)
    wavfile.write("output_odd.wav", sample_rate, waveform)

    plt.bar(array, m_coef, width=50)
    plt.xlabel("Array")
    plt.ylabel("M Coefficient")
    plt.show()

