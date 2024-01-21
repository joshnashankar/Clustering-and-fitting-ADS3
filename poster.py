
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# Read the dataset
df = pd.read_csv('C:/Users/Nunna Venkatesh/Documents/jyothsna/world-data-2023.csv.xls')

# Select only numeric columns for clustering
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_columns]

# Fill missing values with the mean
df_numeric = df_numeric.fillna(df_numeric.mean())

# Normalize the data
def scaler(df):
    """ Expects a dataframe and normalises all 
        columns to the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    df_min = df.min()
    df_max = df.max()

    df = (df - df_min) / (df_max - df_min)

    return df, df_min, df_max

df_norm, df_min, df_max = scaler(df_numeric)
df_norm.fillna(0, inplace=True)  # replace NaN values with 0

# Find the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)

# Plot the elbow method
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-Cluster Sum of Squares')
plt.show()


print(df_numeric.head())



from sklearn.metrics import silhouette_score

# Define the range of clusters to try
n_clusters_range = range(2, 11)

# Compute silhouette scores for each number of clusters
silhouette_scores = []
for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(df_norm)
    silhouette_scores.append(silhouette_score(df_norm, labels))

# Plot the silhouette scores
plt.plot(n_clusters_range, silhouette_scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis: Optimal Number of Clusters")
plt.show()



def exponential_growth(x, a, b, c):
    return a * np.exp(b * x) + c

# Error ranges function
def err_ranges(popt, pcov, x, model_func):
    perr = np.sqrt(np.diag(pcov))
    y = model_func(x, *popt)
    lower = model_func(x, *(popt - perr))
    upper = model_func(x, *(popt + perr))
    return y, lower, upper

# Sample x values (replace this with your actual x values)
x_values = np.arange(len(df_norm))

# Sample y values (replace this with the column you want to fit)
y_values = df_norm['Birth Rate']

# Perform curve fitting
params, covariance = curve_fit(exponential_growth, x_values, y_values)

# Get the fitted parameters
a, b, c = params

# Generate fitted curve
fitted_curve = exponential_growth(x_values, a, b, c)

# Calculate error ranges
y_fit, lower, upper = err_ranges(params, covariance, x_values, exponential_growth)

# Plot the original data, fitted curve, and error ranges
plt.plot(x_values, y_values, label='Original Data')
plt.plot(x_values, fitted_curve, label='Fitted Curve', linestyle='--')
plt.fill_between(x_values, lower, upper, color='gray', alpha=0.3, label='Error Range')
plt.title('Curve Fitting with Error Range')
plt.xlabel('Time')
plt.ylabel('Birth Rate (Normalized)')
plt.legend()
plt.grid(True)
plt.show()


# Plot bar graphs for each column
for column in df_norm.columns:
    plt.figure(figsize=(10, 6))
    plt.bar(df_norm.index, df_norm[column])
    plt.xlabel('Countries')
    plt.ylabel(f'Normalized {column}')
    plt.title(f'Distribution of {column}')
    plt.show()

    # Choose the optimal number of clusters (e.g., from the elbow method)
optimal_clusters = 3

# Perform clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df_norm['Cluster'] = kmeans.fit_predict(df_norm)

# Scatter plot for 'Life expectancy' with cluster centers
plt.figure(figsize=(12, 8))

# Scatter plot for 'Life expectancy'
plt.scatter(df_norm.index, df_norm['Life expectancy'], c=df_norm['Cluster'], cmap='viridis', label='Life Expectancy')

# Scatter plot for cluster centers
for cluster_center in kmeans.cluster_centers_:
    plt.scatter(len(df_norm) + 1, cluster_center[4], marker='X', s=100, c='red', label='Cluster Center')

plt.xlabel('Countries')
plt.ylabel('Normalized Life Expectancy')
plt.title('Clustering Results for Life Expectancy with Cluster Centers')
plt.legend()
plt.grid(True)
plt.show()



# Sample data for illustration purposes (replace this with your actual data)
selected_countries = ['Australia', 'Belgium', 'Brazil', 'France', 'Germany', 'Norway', 'Turkey', 'Sweden', 'United States' 'United Kingdom']  # Replace with the countries you want to include
subset_df = df[df['Country'].isin(selected_countries)]

# Create a line graph
plt.plot(subset_df['Country'], subset_df['Calling Code'], marker='o', color='green', linestyle='-', linewidth=2, markersize=8)
plt.xlabel('Country Name')
plt.ylabel('Calling Code')
plt.title('Calling Codes for Selected Countries (Line Graph)')
plt.xticks(rotation=90)

plt.grid(True)
plt.show()


