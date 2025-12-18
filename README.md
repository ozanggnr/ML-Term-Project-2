# ML-Term-Project-2

This project focuses on customer segmentation to maximize business insights and profitability. Using unsupervised learning techniques, customers are grouped based on their similarities. The objective is to maximize the silhouette score while identifying meaningful customer segments.

- Data
The dataset is provided via AIRS.
The data contains missing values and requires preprocessing.
Exploratory Data Analysis (EDA) is used to gain a deeper understanding of feature distributions and data quality.

- Methodology
The optimal number of clusters is determined using clustering evaluation techniques such as:

Elbow Method

Silhouette Analysis
Relevant plots are generated to justify the chosen number of clusters.

A single final pipeline is implemented using scikit-learn, including:

Preprocessing steps

Clustering model training

Silhouette score calculation

Hyperparameter search code is intentionally excluded, as required.

<h2>Output<h2>

The code prints:

- Number of clusters found
- Final silhouette score
- Number of instances in each cluster

<h3>Requirements<h3>

- Python
- scikit-learn
- numpy, pandas, matplotlib 
