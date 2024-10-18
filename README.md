# Fraud Detection Using Isolation Forests and HDBSCAN

## Overview
This project focuses on detecting fraudulent transactions using machine learning, specifically implementing Isolation Forests and HDBSCAN clustering algorithms. The aim was to build a model that can accurately identify fraud with high recall and precision, both essential metrics in fraud detection.

## Project Details
### Tools and Libraries

* **pandas and numpy:** For data manipulation and analysis.
* **matplotlib and seaborn:** For data visualisation and exploratory analysis.
* **sklearn:** For machine learning models (Isolation Forest, classification metrics, standardization, PCA).
* **hdbscan:** For density-based clustering and subclustering of high-risk transactions.
* **NearestNeighbors (from sklearn.neighbors):** For mapping clusters from new data to original clusters using nearest neighbor search.

### The Dataset
The dataset used in this project was sourced from Kaggle usign the Kaggle API. Each transaction in the dataset is labeled as either:
* 0: Non-fraudulent.
* 1: Fraudulent.

The data is highly imbalanced, with fraudulent transactions representing a small fraction of the total transactions. This imbalance poses challenges for training robust models that generalise well.

### Modelling Approach

The goal was to get a recall above 50% and a precision above 25%.

Two main algorithms were used to detect fraud in the dataset:
* **Isolation Forests:** A tree-based anomaly detection method that works by isolating points in a dataset, treating anomalies as points that are more easily isolated.
* **HDBSCAN Clustering:** A density-based clustering approach that groups data points into clusters based on their density. After clustering with HDBSCAN, subclustering was performed to further isolate smaller, high-risk groups.

HDBSCAN clustering was found to give better initial results in terms of recall and precision, so was used for further testing.

#### Model Artifacts
Three artifacts were generated and saved during the project:
* ** cluster_centroids.pkl:** This file stores the centroids of the main clusters and subclusters formed by the HDBSCAN algorithm.
* **high_fraud_cluster_lists.pkl:** Contains the lists of high-risk clusters and subclusters that showed a higher concentration of fraud.
* **df_sampled.pkl:** Contains the sample of the data used in building the initial clustering model.

These files can be loaded for future analysis or model testing using the following code:

    import pickle
    
    with open('cluster_centroids.pkl', 'rb') as file:
        original_cluster_centroids = pickle.load(file)
        original_subcluster_centroids = pickle.load(file)

    with open('high_fraud_cluster_lists.pkl', 'rb') as file:
        high_fraud_clusters = pickle.load(file)
        high_fraud_subclusters = pickle.load(file)
        high_fraud_subclusters_62 = pickle.load(file)

    with open('df_sampled.pkl', 'rb') as file:
        df_sampled = pickle.load(file)
  
### Model Evaluation
The performance of both the Isolation Forest and HDBSCAN models was evaluated using recall and precision, critical metrics in fraud detection:
* **Recall:** The percentage of actual fraudulent transactions correctly identified by the model.
* **Precision:** The percentage of transactions flagged as fraudulent that were truly fraudulent.

#### Performance on Original Data
* Recall: 54%
* Precision: 34%

#### Performance on New Data
* Recall: 7%
* Precision: 1%

### Reflections
Despite achieving the initial goals of >50% recall and 25% precision during initial model building, applying the model to new data revealed a significant performance degradation, with recall and precision dropping to 7% and 1%, respectively.

To improve performance in future iterations, I would:
* Train the model on a larger, more representative sample of the data.
* Explore techniques for handling class imbalance more effectively, such as adjusting loss functions or applying oversampling techniques.
* Dedicate more time to subclustering different clusters and experimenting with alternative clustering techniques to ensure that the model captures general patterns rather than specific clusters.
* Apply more comprehensive validation techniques, such as k-fold cross-validation or testing on multiple datasets, to identify potential weaknesses earlier in the modeling process.

In conclusion, while the model did not generalise well to new data, this experience provides valuable insights into the limitations of the current approach. These reflections can guide the design of a more robust and adaptable fraud detection model in future projects.

## Repository Structure
* **Online Payment Fraud Detection Using Machine Learning.ipynb:** The main notebook detailing all steps of the project, including data preprocessing, exploratory data analysis, feature engineering, model building, and evaluation.
* **Testing Clustering Model on New Data.ipynb:** Notebook testing the model on new data by first running the HDBSCAN and then mapping the new clusters to the original clusters using KNN.
* **cluster_centroids.pkl:** Stores the centroids of the clusters and subclusters created by HDBSCAN.
* **high_fraud_cluster_lists.pkl:** Contains the high-risk cluster and subcluster lists identified during the project.
* **df_sampled.pkl:** Contains the sample of the data used in building the initial clustering model.

## How to Use the Project
To run the project:
* Clone the repository and download the notebook and .pkl files.
* Follow the 'Testing Clustering Model on New Data.ipynb' notebook to test the model.

Note: the 'Testing Clustering Model on New Data.ipynb' tests the clustering model on another 10% sample of the same dataset used in the 'Online Payment Fraud Detection Using Machine Learning.ipynb'. 

If you wish to test using another dataset, it will need to have similar initial columns, e.g. 'step', 'amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', etc.. Please see the prepare_features function in either of the notebooks for the columns that will be required. 
