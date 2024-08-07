import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits import mplot3d

# Stats
from scipy.stats import uniform, norm

# Clustering
from sklearn.cluster import KMeans, DBSCAN


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

def conditional_probability_calculator(data_frame, condition_column, condition_value, target_column, target_condition):
    """
    Calculate conditional probability P(target_condition | condition_column = condition_value)
    
    Parameters:
    data_frame (pd.DataFrame): The DataFrame containing the data
    condition_column (str): The column name for the condition
    condition_value (str/int/float): The value for the condition
    target_column (str): The column name for the target
    target_condition (str): A string representing the condition for the target column (e.g., "> 40", "== 'Yes'")
    
    Returns:
    float: The calculated conditional probability
    """
    # Filter the DataFrame based on the condition
    condition_df = data_frame[data_frame[condition_column] == condition_value]
    
    # Count of rows meeting the condition
    condition_count = len(condition_df)
    
    # Filter further based on the target condition
    target_df = condition_df.query(f"{target_column} {target_condition}")
    
    # Count of rows meeting both conditions
    target_count = len(target_df)
    
    # Calculate conditional probability
    conditional_prob = target_count / condition_count if condition_count > 0 else 0
    
    return conditional_prob

# Example usage:
# p_over_40_given_first_class = conditional_probability_calculator(
#     titanic, 'class', 'First', 'age', '> 40'
# )


def check_inertia_of_range_of_clusters(upper_n_limit, init_value, random_stat_val):
    inertia_array = [] 
    #for each value 1 - 10
    for n in range(1,upper_n_limit):
        #instantiate new KMeans instance
        #Don't Forget to set the random_state!!!
        #fit the model
        kmeans_temp = KMeans(n_clusters=n, init=init_value, random_state=random_stat_val).fit(X)  
        i = kmeans_temp.inertia_
        #append inertia score to inertias list
        inertia_array.append(i)
        
    return inertia_array

# Example usage:
# inertias = check_inertia_of_range_of_clusters(11, 'k-means++', 42)


#examine the number of clusters created with 
#each epsilon value. 
#Plot the results
def exampine_number_of_clusters_for_array_of_eps(epsilons):
    n_clusters_list = []
    for eps in epsilons:
        db = DBSCAN(eps = eps).fit(X)
        n_clusters = len(np.unique(db.labels_))
        n_clusters_list.append(n_clusters)
    plt.plot(epsilons, n_clusters_list)
    plt.xlabel('Epsilon')
    plt.ylabel('Number of Clusters')
    plt.title('How the Number of Clusters varies with eps')
    
# Example usage:
# exampine_number_of_clusters_for_array_of_eps(epsilons)
