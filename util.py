from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import f_classif, SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def change_df_dtype(df,col_list):
    df[col_list] = df[col_list].apply(pd.to_numeric,errors='coerce')


def clf_comparison(X_train, X_test,y_train,y_test):
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree", "Random Forest", "Naive Bayes"]

    classifiers = [
        KNeighborsClassifier(algorithm='ball_tree', leaf_size=5, metric='minkowski', n_neighbors=3, p=1,
                                   weights='distance'),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        GaussianNB()]
    for name, clf in zip(names,classifiers):
        clf.fit(X_train, y_train)
        clf_pred = clf.predict(X_test)
        clf_score = round(clf.score(X_train, y_train) * 100, 2)
        clf_acc = round(accuracy_score(y_test, clf_pred) * 100, 2)
        print(name, " Training Score : ", clf_score)
        print(name, " Accuracy  : ", clf_acc)


def drop_missing_values(dataframe):
    df = dataframe.replace(to_replace=['?', 'dna'], value=np.nan)
    df = df.dropna()
    return df



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots( figsize=(12,8), dpi=120)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def clf_knn(X_train, X_test,y_train,y_test):
    clf = KNeighborsClassifier(algorithm='ball_tree', leaf_size=5, metric='minkowski', n_neighbors=3, p=1,
                                   weights='distance')
    clf.fit(X_train, y_train)
    clf_pred = clf.predict(X_test)
    clf_score = round(clf.score(X_train, y_train) * 100, 2)
    clf_acc = round(accuracy_score(y_test, clf_pred) * 100, 2)

    return clf_score, clf_acc,clf_pred


######## Univariate Feature Selection  #############
def selectkbest_fClassif(X,y, k, dataframe):
    selector = SelectKBest(score_func = f_classif, k = k)
    selector.fit(X,y)
    scores =  pd.DataFrame(selector.scores_)
    cols = pd.DataFrame(X.columns)
    featureScore = pd.concat([cols,scores], axis=1)
    featureScore.columns=['Features','Score']
    featureScore = featureScore.replace([np.inf, -np.inf], np.nan)
    featureScore = featureScore.nlargest(k, 'Score')
    print('Select K Best with score function - f_classif:\n',featureScore)
    feats = featureScore['Features'].to_list()
    col_new = [x for x in featureScore['Features']]
    df_new = dataframe.loc[:, col_new]

    return feats, df_new

def selectkbest_chi2(X,y,k, dataframe):
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X, y)
    scores = pd.DataFrame(selector.scores_)
    cols = pd.DataFrame(X.columns)
    featureScore=pd.concat([cols,scores], axis=1)
    featureScore.columns=['Features','Score']
    featureScore=featureScore.nlargest(k, 'Score')
    print('Select K Best with score function - chi2:\n',featureScore)
    print(featureScore.Features)
    feats = featureScore['Features'].to_list()
    col_new = [x for x in featureScore['Features']]
    df_new = dataframe.loc[:, col_new]

    return feats, df_new

def selectkbest_mutual_info(X,y, k, dataframe):
    selector = SelectKBest(score_func = mutual_info_classif, k = k)
    selector.fit(X,y)
    scores = pd.DataFrame(selector.scores_)
    cols = pd.DataFrame(X.columns)
    featureScore = pd.concat([cols,scores], axis=1)
    featureScore.columns=['Features','Score']
    featureScore = featureScore.nlargest(k, 'Score')
    #print('Select K Best with score function - mutual_info_classif:\n',featureScore)
    #print(featureScore.Features)
    feats = featureScore['Features'].to_list()
    col_new = [x for x in featureScore['Features']]
    df_new = dataframe.loc[:, col_new]

    return feats, df_new

