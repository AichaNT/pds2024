"""
@authors: cjep, kmah, feso, aith, nozo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix



label_data = "data/ground_truth.csv"
feature_data = "data/features.csv"


df_labels = pd.read_csv(feature_data)
df_feat= pd.read_csv(label_data)


df = pd.merge(df_labels, df_feat, left_on=["image_id"], right_on=["image_id"])[["melanoma","A","C", "DG"]]


X = df[list(df.columns)[1:]]
y = df["melanoma"]

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



# ---Decision tree---

DT_classifiers = [
    DecisionTreeClassifier(max_depth=None, max_leaf_nodes=2),
    DecisionTreeClassifier(max_depth=None, max_leaf_nodes=4),
    DecisionTreeClassifier(max_depth=None, max_leaf_nodes=6),
    DecisionTreeClassifier(max_depth=None, max_leaf_nodes=8),
    DecisionTreeClassifier(max_depth=None, max_leaf_nodes=10)
]

# Loop through each classifier
for j, clf in enumerate(DT_classifiers):

    print(f"\n\nFor {2 * (j + 1)} leaf nodes:\n")

    # Train model with different max_depth
    clf.fit(X_train[['A', 'C', 'DG']], y_train)

    # Plot decision tree structure
    plt.figure(figsize=(5, 3))
    tree.plot_tree(clf, feature_names=['A', 'C', 'DG'], class_names=['0', '1'])
    #plt.show()

    # Prediction
    y_pred = clf.predict_proba(X_train)[:, 1]  # Probability of class 1

    # Calculate AUC score
    auc_score = roc_auc_score(y_train, y_pred)
    print(f"AUC Score : {auc_score}\n")

    # Predict the labels for the training set
    y_pred_labels = clf.predict(X_train)

    # Compute confusion matrix
    conf_matrix_DT = confusion_matrix(y_train, y_pred_labels)
    print("\nConfusion Matrix:")
    print(conf_matrix_DT)



# ---K-Nearest Neighbors---

# Prepare cross-validation
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds)

# Different classifiers to test out
KNN_classifiers = [
    KNeighborsClassifier(n_neighbors=1),
    KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier(n_neighbors=5),
    KNeighborsClassifier(n_neighbors=7),
    KNeighborsClassifier(n_neighbors=9),
    KNeighborsClassifier(n_neighbors=11)
]

num_KNN_classifiers = len(KNN_classifiers)

AUC_val = np.empty([num_folds, num_KNN_classifiers])

true_labels = [[] for _ in range(num_KNN_classifiers)]
predicted_labels = [[] for _ in range(num_KNN_classifiers)]


# Loop through the folds
for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    
    # Extract the train and test data for this fold
    x_train_fold, x_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    for j, clf in enumerate(KNN_classifiers): 
        
        # Fit the classifier on the training data
        clf.fit(x_train_fold, y_train_fold)

        # Evaluate your metric of choice 
        y_KNN_pred = clf.predict_proba(x_val_fold)[:, 1] # Probability of class 1
        AUC_val[i,j] = roc_auc_score(y_val_fold, y_KNN_pred)

        # Predict the labels for the validation set
        pred_labels = clf.predict(x_val_fold)

        # Append true labels and predicted labels for this fold
        true_labels[j].extend(y_val_fold)
        predicted_labels[j].extend(pred_labels)

#Average over all folds
average_KNN_auc = np.mean(AUC_val, axis=0) 

# Loop through each classifier
for j, clf in enumerate(KNN_classifiers):
    
    print(f'\n\nFor KNN classifier with neighbors = {2 * j + 1}:\n')

    print(f'Average AUC: {average_KNN_auc[j]}')

    # Train the classifier on the entire training set
    clf.fit(X_train, y_train)

    # Compute confusion matrix
    conf_matrix_KNN = confusion_matrix(true_labels[j], predicted_labels[j])
    print("\nConfusion Matrix:")
    print(conf_matrix_KNN)


# Final classifier
classifier = KNeighborsClassifier(n_neighbors = 3)

# Training this classifier on the entire dataset
classifier = classifier.fit(X, y)

# Saving classifier
dir = 'features'
filename = os.path.join(dir, 'groupK_classifier.sav')

pickle.dump(classifier, open(filename, 'wb'))