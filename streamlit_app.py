import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# Load Data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)


# K-Means Clustering
def kmeans_clustering(data, n_clusters):
    X_kmeans = data[['PC1', 'PC2']]
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_kmeans)
    data['KMeans_Cluster'] = kmeans.labels_
    return kmeans, data


# Streamlit UI
st.title("K-Means Model Implementation")

# Add a description about the expected data format
st.write("""
### Instructions:
Please upload a CSV file containing the following columns:
- **PC1**: Principal Component 1 (numerical data)
- **PC2**: Principal Component 2 (numerical data)

These columns are required for K-Means clustering.
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Load the data
    data = load_data(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    st.write("## K-Means Clustering")

    # K-Means Parameters
    n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)

    if st.button("Run K-Means"):
        kmeans_model, clustered_data = kmeans_clustering(data, n_clusters)

        st.write("### Clustered Data Preview")
        st.write(clustered_data.head())

        # Plot clusters
        st.write("### K-Means Clustering Plot")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='KMeans_Cluster', data=clustered_data, palette='viridis')
        st.pyplot(plt)

        # Save K-Means Model
        kmeans_model_path = 'kmeans_model.pkl'
        joblib.dump(kmeans_model, kmeans_model_path)
        st.success(f"K-Means model saved as {kmeans_model_path}")
