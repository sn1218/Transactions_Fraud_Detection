# Fraud Detection Using Isolation Forests and HDBSCAN

## Overview

This project implements machine learning models to detect fraudulent transactions. It experiments with both Isolation Forests and HDBSCAN clustering, with HDBSCAN yielding better results in terms of fraud detection.

## The Project
The data was sourced from Kaggle, where each transaction is labeled as 0 for non-fraudulent or 1 for fraudulent.

The modelling and testing was all done in the 'Online Payment Fraud Detection Using Machine Learning.ipynb' file. This notebook includes:
* Initial data cleaning
* Exploratory data analysis
* Feature engineering
* Modelling using isolation forests and HDBSCAN clustering, followed by subclustering
* Evaluation of the HDBSCAN model using a separate sample from the dataset

The centroids for clustering and subclustering, as well as the high-risk cluster and subcluster lists, are saved in separate files: 'cluster_centroids.pkl' and 'high_fraud_cluster_lists.pkl'. These files are not required to run the notebook but are saved for future use. Use the following code to load them:

    import pickle
    
    with open('cluster_centroids.pkl', 'rb') as file:

        original_cluster_centroids = pickle.load(file)
    
        original_subcluster_centroids = pickle.load(file)

    with open('high_fraud_cluster_lists.pkl', 'rb') as file:

        high_fraud_clusters = pickle.load(file)
    
        high_fraud_subclusters = pickle.load(file)
    
        high_fraud_subclusters_62 = pickle.load(file)
  
### Model Performance
We focus on the recall and precision of the fraudulent data. Recall measures how well fraudulent transactions were identified, while precision measures how many of the flagged transactions were actually fraudulent.

#### Original Data
* Recall: 54%
* Precision: 34%

#### New Data
* Recall: 6%
* Precision: 0%

### Reflections
Despite achieving the initial goals of >50% recall and 25% precision during initial model building, applying the model to new data revealed a significant performance degradation, with recall and precision dropping to 6% and 0%, respectively.

To improve performance in future iterations, I would:
* Train the model on a larger, more representative sample of the data.
* Explore techniques for handling class imbalance more effectively, such as adjusting loss functions or applying oversampling techniques.
* Dedicate more time to subclustering different clusters and experimenting with alternative clustering techniques to ensure that the model captures general patterns rather than specific clusters.
* Apply more comprehensive validation techniques, such as k-fold cross-validation or testing on multiple datasets, to identify potential weaknesses earlier in the modeling process.

In conclusion, while the model did not generalise well to new data, this experience provides valuable insights into the limitations of the current approach. These reflections can guide the design of a more robust and adaptable fraud detection model in future projects.
