import numpy as np
import pandas as pd
from util import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas_profiling

np.random.seed(101)
col_names =['diseases','month','plant-stand','precip','temp','hail','crop-hist','area-damaged','severity',
            'seed-tmt','germination','plant-growth','leaves','leafspots-halo','leafspots-marg','leafspot-size','leaf-shread','leaf-malf',
            'leaf-mild','stem','lodging','stem-cankers','canker-lesion','fruiting-bodies','external-decay','mycelium','int-discolor',
            'sclerotia','fruit-pods','fruit-spots','seed','mold-growth','seed-discolor','seed-size','shriveling','roots']

df = pd.read_csv("soybean-large_data.csv",header = None, names = col_names, nrows=290)
print("--------------Data loaded successfully---------------------------- ")
print(len(df.columns))

#print(df.describe())
#### Data Profiling
#pfrep=pandas_profiling.ProfileReport(df)
#pfrep.to_file('profile_report.html')

col_list =['hail','severity','seed-tmt','germination','leafspots-halo','leafspots-marg',
'leafspot-size','leaf-shread','leaf-malf','leaf-mild','lodging','fruiting-bodies',
'fruit-pods','fruit-spots','seed','mold-growth','seed-discolor','seed-size','shriveling']
# Change the data types of columns
change_df_dtype(df,col_list)

#############      M I S S I N G      V A L U E S   ######################
# Print missing values
print(df.isnull().sum())
# Data Cleaning
df = drop_missing_values(df)
df['diseases'] = df['diseases'].astype('category')
#print('Number of diseases: ',df['diseases'].nunique()) # 15
print(len(df.columns))
#### Data Profiling after data cleaning
#df_pro=pandas_profiling.ProfileReport(df)
#df_pro.to_file('profile_report_after_cleaning.html')

##############   M A C H I N E    L E A R N I N G       ###################
X = df.drop(['diseases'], axis = 1)
y = df['diseases']

####  Feature Selection
df_score3,df_new = selectkbest_mutual_info(X= X,y=y, k= 15,dataframe=df)

df_new['diseases'] = df['diseases']
X = df_new.drop(['diseases'], axis = 1)
y = df_new['diseases']
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 101)

#### Compare the data among "Nearest Neighbors, Linear SVM, RBF SVM, Decision Tree, Random Forest, Naive Bayes
clf_comparison(X_train, X_test,y_train,y_test)


#############        K N N                ##########################
knn_clf_score, knn_clf_acc, knn_clf_pred = clf_knn(X_train, X_test,y_train,y_test)
print("KNN Training Score : ",knn_clf_score)
print("KNN Accuracy       : ", knn_clf_acc)

#### Plotting Confusion Matrix
class_names = df['diseases'].unique()
classes = class_names[unique_labels(y_test, knn_clf_pred)]
plot_confusion_matrix(y_test, knn_clf_pred, classes=class_names,
                          normalize=True,
                          title="Confusion Matrix with Feature Selection")
#plt.show()

#plt.savefig(fname="Confusion Matrix with Feature Selection" , dpi=120)

####### KNN on Original Dataset
X = df.drop(['diseases'], axis = 1)
y = df['diseases']
le= LabelEncoder()
y = le.fit_transform(y)

X_train_orig, X_test_orig,y_train_orig,y_test_orig = train_test_split(X,y,test_size = 0.25,random_state = 101)
knn_clf_score, knn_clf_acc, knn_clf_pred = clf_knn(X_train_orig, X_test_orig,y_train_orig,y_test_orig )
print("KNN Training Score (Original Data) : ",knn_clf_score)
print("KNN Accuracy (Original Data)       : ", knn_clf_acc)

##### Plotting Confusion Matrix
class_names = df['diseases'].unique()
classes = class_names[unique_labels(y_test_orig, knn_clf_pred)]
plot_confusion_matrix(y_test_orig, knn_clf_pred, classes=class_names,
                          normalize=True,
                          title="Confusion Matrix On Original Data")
#plt.savefig(fname="Confusion Matrix On Original Data" , dpi=120)
#plt.show()



