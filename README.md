# Fraud Detection Using Isolation Forests and HDBSCAN

## Overview

This project implements machine learning models to detect fraudulent transactions. This project experiments with both isolation forests and HDBSCAN clustering. HDBSCAN clustering was found to give better results in terms of fraud detection. 

## The Project
The data was sourced from Kaggle, and each transaction was labelled with 0 for non-fraudulent or 1 for fraudulent.

The modelling and testing was all done in the 'Online Payment Fraud Detection Using Machine Learning.ipynb' file. The file includes initial data cleaning, exploratory data analysis, feature engineering, and modelling using isolation forests and HDBSCAN clustering and then subclustering. The HBDSCAN model is then evaluated using another sample from the dataset.

The centroids for the clustering and subclustering, as well as the hihg-risk cluster and subcluster lists are saved separately in 'cluster_centroids_and_high_fraud_clusters.pkl'. They are not required to run the notebook, but are saved for future use.

### Model Performance
We are interested in the recall and precision of the fraudulent data. Recall measures how well fraudulent transactions were identified, while precision measures how many of the flagged transactions were actually fraudulent

#### Original Data
* Recall: 54%
* Precision: 34%

#### New Data
* Recall: 6%
* Precision: 0%

### Reflections
Despite achieving the initial goal of >50% recall and 25% precision when first building the model, when applying the model to new data, it became apparent that the model's performance significantly degraded, with recall and precision dropping to 6% and 0%, respectively.

To improve performance in future iterations, I would:
* Train the model on a larger, more representative sample of the data.
* Explore techniques for handling class imbalance more effectively, such as adjusting loss functions or applying oversampling techniques.
* Dedicate more time to subclustering different clusters and experimenting with alternative clustering techniques to ensure that the model captures general patterns rather than specific clusters.
* Apply more comprehensive validation techniques, such as k-fold cross-validation or testing on multiple datasets, to identify potential weaknesses earlier in the modeling process.

In conclusion, while the model did not generalise well to the new data, this experience provides valuable insights into the limitations of the current approach. With these reflections, I can design a more robust and adaptable fraud detection model in future projects.
