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

# Models, MSE, Pipelines
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV


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

def plot_hist_with_mean_std(df, feature):
    # Calculate mean and standard deviation
    mean_val = df[feature].mean()
    std_val = df[feature].std()

    # Plot histogram
    plt.hist(df[feature], bins=50)

    # Add dotted line for the mean
    plt.axvline(mean_val, color='r', linestyle='dotted', linewidth=2, label='Mean')

    # Add dotted lines for one standard deviation above and below the mean
    plt.axvline(mean_val - std_val, color='g', linestyle='dotted', linewidth=2, label='1 std dev below')
    plt.axvline(mean_val + std_val, color='b', linestyle='dotted', linewidth=2, label='1 std dev above')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

# Example usage:
# plot_hist_with_mean_std(df_pha, 'diameter')

def l2_loss(theta, x, y):
    """
    This function accepts a single theta value
    and calculates the mean squared error based
    on (theta*x - y)^2

    Arguments
    ---------
    theta: float
    The value to use for the parameter of the
    regression model.

    Returns
    -------
    mse: float
    Mean Squared Error
    """

    return np.mean((theta*x - y)**2)


def mae_loss(theta,x,y):
    """
    This function accepts an array of thetas
    and returns the mean absolute error based
    on np.mean(|(theta*x - y)|)
    
    Arguments
    ---------
    theta: float
           Values to use for parameter
           of regression model.
            
    Returns
    -------
    mse: np.float
         Mean Absolute Error
    """

    return np.mean(np.abs((theta * x) - y))


def huber_loss(theta, x, y, delta = 1.5):
    """
    This function accepts a value for theta
    and returns the sum of the huber loss.
    
    Arguments
    ---------
    theta: float
           Values to use for parameter
           of regression model.
           
    delta: float
           Value for delta in Huber Loss
            
    Returns
    -------
    huber: np.float
         Sum of huber loss
    """
    y_pred = theta * x
    y_err = np.abs(y-y_pred)
    return sum(np.where(y_err <= delta, 1/2*(y_err)**2, delta*(y_err - 1/2*delta)))

def mse_for_different_degrees(X,y,range_low, range_stop):
    mses = []
    for i in range(range_low,range_stop):
    #for 1, 2, 3, ..., 10

        #create pipeline
        pipeline = Pipeline([
            ('features', PolynomialFeatures(degree=i, include_bias=False)),
            ('model', LinearRegression())
        ])
        #fit pipeline
        pipeline.fit(X,y)
        #make predictions
        predictions = pipeline.predict(X)
        #compute mse
        mse = mean_squared_error(y,predictions)
        #append mse to mses
        mses.append(mse)
        
    return mses

def predictions_for_range_of_degrees(X_train, y_train, X_pred, range_start, range_stop):
    predictions = []
    #for 1, 2, 3, ..., 10
    for i in range(range_start, range_stop):
        #create pipeline
        pipe = Pipeline([
            ('quad_features', PolynomialFeatures(degree=i, include_bias=False)),
            ('quad_model', LinearRegression())
        ])
        #fit pipeline on training data
        pipe.fit(X_train, y_train)
        #make predictions on all data
        preds = pipe.predict(X_pred)
        #assign to model_predictions
        predictions.append(preds)
        
    return predictions

def get_mse_for_test_and_val_on_lr_degrees(training_X, training_y, validation_X, validation_y, start, stop):
    train_mses = []
    test_mses = []

    # For complexity from 'start' to 'stop':
    for i in range(start, stop + 1):
        # Create pipeline with PolynomialFeatures and LinearRegression
        # Remember to set include_bias = False
        pipe = Pipeline([
            ('pfeat', PolynomialFeatures(degree=i, include_bias=False)),
            ('linreg', LinearRegression())
        ])
        # Fit pipeline on training data
        pipe.fit(training_X, training_y)
        # Predict against training data
        train_preds = pipe.predict(training_X)
        # Predict against validation data
        test_preds = pipe.predict(validation_X)
        # MSE of training data
        train_mses.append(mean_squared_error(training_y, train_preds))
        # MSE of validation data
        test_mses.append(mean_squared_error(validation_y, test_preds))

    # Find the best model complexity based on the lowest validation MSE
    best_model_complexity = np.argmin(test_mses) + start  # Adjusting for the range start

    return train_mses, test_mses, best_model_complexity


def simple_cross_validation(X_train, y_train, X_test, y_test, start, stop):
    best_pipe = None  # Placeholder for the best model
    best_mse = np.inf  # Set best mse to infinity to begin

    # For complexity start to stop
    for i in range(start, stop + 1):
        # Create pipeline with PolynomialFeatures and LinearRegression
        # Remember to set include_bias = False
        pipe = Pipeline([
            ('pfeat', PolynomialFeatures(degree=i, include_bias=False)),
            ('linreg', LinearRegression())
        ])
        # Fit pipeline on training data
        pipe.fit(X_train, y_train)
        # Predict against test data
        test_preds = pipe.predict(X_test)
        # MSE of test data
        test_mse = mean_squared_error(y_test, test_preds)

        # If mse is the best so far, update best_mse and best_pipe
        if test_mse < best_mse:
            best_mse = test_mse
            best_pipe = pipe

    # Return the best pipeline
    return best_pipe

def find_highest_correlation_against_feature(data_frame, feature):
    # Compute the correlation matrix
    correlation_matrix = data_frame.corr()

    # Get the correlation values for a given feature
    feature_correlation = correlation_matrix[feature]

    # Drop the original feature to avoid self-correlation
    feature_correlation = feature_correlation.drop(feature)

    # Sort the correlations
    sorted_correlation = feature_correlation.sort_values(ascending=False)
    
    # Return the feature name of the highest correlating feature and the correlation value
    return sorted_correlation.index[0], sorted_correlation.iloc[0]


def create_test_and_train_dataframes(degree, features, X_train, X_test):
    """
    Generates polynomial features for training and test datasets.

    Parameters:
    degree (int): The degree of the polynomial features to be created.
    features (list of str): The names of the features to be transformed into polynomial features.
    X_train (DataFrame): The training dataset containing the features specified.
    X_test (DataFrame): The test dataset containing the features specified.

    Returns:
    tuple: A tuple containing two DataFrames with polynomial features for the training and test datasets.
    """
    
    # Instantiate the PolynomialFeatures object with the given degree,
    # setting include_bias to False to avoid generating a bias column.
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Fit the polynomial features to the training data and transform it,
    # creating the polynomial feature set for the training data.
    X_train_poly = poly_features.fit_transform(X_train[features])
    
    # Transform the test data using the same polynomial features,
    # creating the polynomial feature set for the test data.
    X_test_poly = poly_features.transform(X_test[features])
    
    # Retrieve the new feature names generated by the polynomial transformation.
    columns = poly_features.get_feature_names_out(features)
    
    # Create a DataFrame for the training data with the new polynomial features
    # and the corresponding feature names.
    train_df = pd.DataFrame(X_train_poly, columns=columns)
    
    # Create a DataFrame for the test data with the new polynomial features
    # and the corresponding feature names.
    test_df = pd.DataFrame(X_test_poly, columns=columns)
    
    # Return the two DataFrames containing the new polynomial features
    # for both the training and test sets.
    return train_df, test_df

def compare_different_ridge_alphas(alphas, X_train, y_train):
    # alphas is an array f.e. [0.001, 1.0, 10.0, 100.0]
    coef_list = []
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        model_coefs = model.coef_
        coef_list.append(list(model_coefs))
        
    return coef_list