# soybean_disease
Michalski's famous soybean disease database. The dataset has been obtained from UCI-Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/Soybean+(Large) ).

# Dataset 
The dataset contains 15 different diseases and 35 features of soybean.

# Data Preprocessing
To get the overview of the data, profile report of the data has been generated using pandas_profiling after and before imputing missing values. 
Missing values has been imputed by dropping the corresponding rows. To reduce the number of features, feature reduction technique "SelectKBest" based on estimate mutual information has been used. Using this method, only 15 best features has been selected to predict soybean disease. 
# Machine Learning 
K Nearest Neighbor algorithm has been used to classify the disease class. The algorithm's accuracy on the test set with original data is 86.57%, whereas with the feature reducted dataset it reached 92.54%.
