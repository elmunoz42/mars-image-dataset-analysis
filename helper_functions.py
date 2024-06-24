import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm

# Function to generate sample means from a given distribution and range
def generate_sample_means(dist, sample_num, random_state=None):
    samples = []
    sample_mu = []
    np.random.seed(random_state)
    #loop over values 1 - 500
    for i in range(sample_num):
        #generate samples 
        samples.append(dist.rvs())
        #find sample mean
        #append mean to sample_means
        sample_mu.append(np.mean(samples))
    return sample_mu


# Function to compare and plot sample means to population mean
def plot_sample_means_to_population_mean(sample_means, dist, sample_num):
    mean = dist.mean()
    plt.plot(range(1, sample_num+1), sample_means, label = 'sample mean', color = 'purple')
    plt.axhline(mean, label = 'true mean', color = 'green')
    plt.xlabel('Sample Size')
    plt.legend();

# Function to compare and plot sample means to population mean against a threshold
def actual_vs_sample_mean(dist, sample_mu, threshold):
    actual_mu = dist.mean()
    return True if abs(actual_mu - sample_mu) < threshold else False


# Function to generate sample means from a given distribution and range
def generate_sample_means_w_threshold(dist, sample_num, size_threshold=30, random_state=None):
    samples = []
    sample_mu = []
    np.random.seed(random_state)
    #loop over values 1 - 500
    for i in range(sample_num):
        #generate samples 
        samples.append(dist.rvs())
        #find sample mean
        #append mean to sample_means
        if i >= size_threshold:
            sample_mu.append(np.mean(samples))
    return sample_mu
