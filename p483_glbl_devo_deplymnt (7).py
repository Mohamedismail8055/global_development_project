# -*- coding: utf-8 -*-
"""P483_glbl_devo_deplymnt.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wzp36IgWq2TSWU602k1R11H-_9UP6J7E

# Import Libaries
"""

import subprocess

subprocess.check_call(["pip", "install", "matplotlib"])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Cleaned_World_Development_Data.csv")

df.head()

"""# Basic Overview"""

# Check dataset dimensions and data types
print("Dataset Dimensions:", df.shape)
print("\nData Types:")
print(df.dtypes)

# Check data types
df.info()

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

"""#  Univariate Analysis"""

# List numeric columns
numeric_columns = df.select_dtypes(include=np.number).columns

# Plot histograms for numeric features
df[numeric_columns].hist(figsize=(15, 10), bins=15, color='skyblue', edgecolor='black')
plt.suptitle('Numeric Features Distributions', fontsize=16)
plt.tight_layout()
plt.show()

"""# Bivariate Analysis"""

# Convert percentage strings to numeric values where applicable
def clean_percentage(column):
    if column.dtypes == 'object':  # Only process object (string) columns
        try:
            return column.str.replace('%', '').astype(float) / 100
        except:
            return column  # Return the column as is if conversion fails
    return column  # Return numeric columns as is

# Apply the cleaning function selectively
df_cleaned = df.copy()  # Create a copy to avoid modifying the original DataFrame
df_cleaned = df_cleaned.apply(clean_percentage)

# Check for non-numeric columns after cleaning
non_numeric_columns = df_cleaned.select_dtypes(include=['object']).columns
print("Non-numeric Columns:", non_numeric_columns)

# Drop non-numeric columns for correlation analysis
df_numeric = df_cleaned.drop(columns=non_numeric_columns)

# Calculate correlations
correlation_matrix = df_numeric.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

"""# Insights into Specific Features"""

# Scatter plot for GDP vs CO2 Emissions
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='GDP', y='CO2 Emissions', color='blue')
plt.title('GDP vs CO2 Emissions')
plt.xlabel('GDP')
plt.ylabel('CO2 Emissions')
plt.grid()
plt.show()

# Percentage of missing data in each column
missing_percentage = (df_cleaned.isnull().sum() / len(df)) * 100
print("Percentage of Missing Data:")
print(missing_percentage[missing_percentage > 0])

# Separate numeric and non-numeric columns
numeric_columns = df_cleaned.select_dtypes(include=['number']).columns
non_numeric_columns = df_cleaned.select_dtypes(exclude=['number']).columns

# Fill missing values for numeric columns with their mean
df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())

# Fill missing values for non-numeric columns with a placeholder or mode
df_cleaned[non_numeric_columns] = df_cleaned[non_numeric_columns].fillna("Unknown")

# Verify there are no missing values left
print("Missing Values After Imputation:")
print(df_cleaned.isnull().sum())

# Create numeric_df by selecting numeric columns from df
numeric_df = df_cleaned[numeric_columns]

# heatmap of corr > (+ or -) 0.5

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'numeric_df' is already defined from your previous code.
# If not, replace numeric_df with the actual DataFrame containing numeric columns.

# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

# Filter for correlations greater than 0.5 (absolute value)
correlation_matrix = correlation_matrix[abs(correlation_matrix) > 0.7]

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap (Correlation > 0.5)')
plt.show()

# Removing the lesser correlating features

columns_to_remove = ['Business Tax Rate', 'Days to Start Business', 'Health Exp % GDP', 'Hours to do Tax','Lending Interest', 'Population Total', 'Population Urban']
nu_df = numeric_df.drop(columns=columns_to_remove, errors='ignore')

nu_df.columns

from sklearn.preprocessing import StandardScaler
# Standardization
scaler = StandardScaler()

# Now we can use nu_df for scaling
scaled_df = scaler.fit_transform(nu_df)

nu_df.isnull().sum()

# Pairplot for initial insights
sns.pairplot(data=nu_df.iloc[:, :5])  # Plotting only first 5 columns for simplicity
plt.show()

#K-Means Clustering
# Finding the optimal number of clusters using Elbow Method
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    # Use the correct variable name 'scaled_df' here
    kmeans.fit(scaled_df)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()



"""#KMEANS"""

# Final K-Means Model
optimal_clusters = 3  # Based on the Elbow plot
kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans_model.fit_predict(scaled_df)

kmeans_labels

len(kmeans_labels)

from sklearn.metrics import silhouette_score, davies_bouldin_score
# K-Means Metrics
kmeans_silhouette = silhouette_score(scaled_df, kmeans_labels)
kmeans_davies = davies_bouldin_score(scaled_df, kmeans_labels)
print(f"K-Means Silhouette Score: {kmeans_silhouette}")
print(f"K-Means Davies-Bouldin Score: {kmeans_davies}")

#saving the model
import joblib

joblib.dump(kmeans_model, 'kmeans_model.joblib')



"""#HIERARCHICAL CLUSTERING (agglomerative approach)"""

# Assigning clusters from Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
agglomerative = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
agglomerative_labels = agglomerative.fit_predict(scaled_df)

agglomerative_labels

# prompt: find the silhoutte score of agglomerative

# Calculate and print the silhouette score for agglomerative clustering
agglomerative_silhouette = silhouette_score(scaled_df, agglomerative_labels)
print(f"Agglomerative Silhouette Score: {agglomerative_silhouette}")

"""#DBSCAN"""

from sklearn.neighbors import NearestNeighbors
nearest_n = NearestNeighbors(n_neighbors=2)
nearest_n.fit(scaled_df)
distances, indices = nearest_n.kneighbors(scaled_df)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.title('K-distance Graph')
plt.xlabel('Data Points sorted by distance')
plt.ylabel('Epsilon')
plt.show()

#DBSCAN Clustering
from sklearn.cluster import DBSCAN
dbscan_model = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan_model.fit_predict(scaled_df)

dbscan_labels

import numpy as np
dbscancluster_labels = np.unique(dbscan_labels)
print(f"Unique values in dbscan_labels: {dbscancluster_labels}")

# DBSCAN Metrics
if len(set(dbscan_labels)) > 1:
    dbscan_silhouette = silhouette_score(scaled_df, dbscan_labels)
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")
else:
    dbscan_silhouette = None
    print("DBSCAN did not form multiple clusters.")

"""# TUNED DBSCAN MODEL"""

from sklearn.model_selection import ParameterGrid

# Define the parameter grid for DBSCAN
param_grid = {
    'eps': [0.3, 0.5, 0.7, 1.0],  # Adjust the range of epsilon values
    'min_samples': [1, 2, 3, 5, 10, 15] # Adjust the range of min_samples values
}

best_score = -1
best_params = {}

for params in ParameterGrid(param_grid):
    dbscan_model = DBSCAN(**params)
    Tuned_dbscan_labels = dbscan_model.fit_predict(scaled_df)

    # Handle cases where DBSCAN creates only one cluster
    if len(set(Tuned_dbscan_labels)) > 1:
        score = silhouette_score(scaled_df, Tuned_dbscan_labels)
        if score > best_score:
            best_score = score
            best_params = params

print(f"Best DBSCAN parameters: {best_params}")
print(f"Best Silhouette Score: {best_score}")

# Train final model with best hyperparameters
best_dbscan_model = DBSCAN(**best_params)
best_dbscan_model.fit(scaled_df)

#Comparative Analysis
results = {
    'Model': ['K-Means', 'Hierarchical', 'TUNED DBSCAN'],
    'Silhouette Score': [kmeans_silhouette, agglomerative_silhouette, best_score],
    'Davies-Bouldin Score': [kmeans_davies, None, None]
}
results_df = pd.DataFrame(results)
print("Comparative Analysis of Clustering Models:")
print(results_df)

"""#Visualization of clusters"""

# Plot the K-means clusters using scatter plot
plt.scatter(scaled_df[:, 0], scaled_df[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering', color='blue')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plot the Tuned DBscan clusters using scatter plot
plt.scatter(scaled_df[:, 0], scaled_df[:, 1], c=Tuned_dbscan_labels, cmap='viridis')
plt.title('Tuned DBscan Clustering', color='blue')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plot the Hierarchical clusters using scatter plot
plt.scatter(scaled_df[:, 0], scaled_df[:, 1], c=agglomerative_labels, cmap='viridis')
plt.title('Hierararchical Clustering', color='blue')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

nu_df.head()

import subprocess

subprocess.check_call(["pip", "install", "streamlit"])  # ! pip install streamlit is invalid code for .py, which we will be using for streamlit app.

nu_df.columns

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# 
# # Load the trained KMeans model and scaler
# kmeans_model = joblib.load('kmeans_model.joblib')
# 
# st.title('KMeans Clustering Prediction')
# 
# # Create input fields in the Streamlit app
# # Input fields for user
# Birth_Rate= st.number_input("Birth Rate", min_value=0.0)
# CO2_Emissions= st.number_input("CO2 Emissions", min_value=0.0)
# Energy_Usage= st.number_input("EnergyUsage", min_value=0.0)
# gdp= st.number_input("GDP", min_value=0.0)
# Health_Exp/Capita= st.number_input("Health Exp/Capita", min_value=0.0)
# Infant_Mortality Rate= st.number_input("Infant Mortality Rate", min_value=0.0)
# Internet_Usage= st.number_input("Internet Usage", min_value=0.0)
# Life_Expectancy_Female= st.number_input("Life Expectancy Female", min_value=0.0)
# Life_Expectancy_Male= st.number_input("Life Expectancy Male", min_value=0.0)
# Mobile_Phone_Usage= st.number_input("Mobile Phone Usage", min_value=0.0)
# Population_0-14= st.number_input("Population 0-14", min_value=0.0)
# Population_15-64= st.number_input("Population 15-64", min_value=0.0)
# Population_65+= st.number_input("Population 65+", min_value=0.0)
# Tourism_Inbound= st.number_input("Tourism Inbound", min_value=0.0)
# Tourism_Outbound= st.number_input("Tourism Outbound", min_value=0.0)
# 
# # Create a dictionary to store the input values
# input_data = {
#      'Birth Rate': Birth_Rate
#      'CO2 Emissions' : CO2_Emissions
#      'Energy Usage': Energy_Usage
#      'GDP': gdp
#       'Health Exp/Capita': Health_Exp/Capita
#      'Infant Mortality Rate': Infant_Mortality_Rate
#      'Internet Usage': Internet_Usage
#        'Life Expectancy Female': Life_Expectancy_Female
#       'Life Expectancy Male': Life_Expectancy_Male
#       'Mobile Phone Usage': Mobile_Phone_Usage
#        'Population 0-14': Population_0-14
#       'Population 15-64': Population_15-64
#       'Population 65+': Population_65+
#        'Tourism Inbound': Tourism_Inbound
#       'Tourism Outbound': Tourism_Outbound
# }
# 
# # Convert input data to DataFrame
# input_df = pd.DataFrame([input_data])
# 
# # Preprocess the input data
# try:
#     # Make the prediction
#     prediction = kmeans_model.predict(scaled_input_data)
# 
#     # Display the prediction
#     st.success(f'Predicted Cluster: {prediction[0]}')
# except Exception as e:
#     st.error(f"An error occurred: {e}")  # error handling
#