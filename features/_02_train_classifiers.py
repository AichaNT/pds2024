"""
@authors: cjep, kmah, feso, aith, nozo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


gt_data = "data/ground_truth.csv"
feature_data = "data/features.csv"

df_gt = pd.read_csv(gt_data)
df_feat= pd.read_csv(feature_data)

df = pd.merge(df_feat, df_gt, on=["image_id"])[["melanoma","A","C", "DG"]]


X = df.iloc[:, 1:]
y = df["melanoma"]


# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Prepare cross-validation
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds)


# ---Decision tree---

DT_classifiers = [
    DecisionTreeClassifier(max_depth=None, max_leaf_nodes=2),
    DecisionTreeClassifier(max_depth=None, max_leaf_nodes=4),
    DecisionTreeClassifier(max_depth=None, max_leaf_nodes=6),
    DecisionTreeClassifier(max_depth=None, max_leaf_nodes=8),
    DecisionTreeClassifier(max_depth=None, max_leaf_nodes=10)
]

num_DT_classifiers = len(DT_classifiers)

AUC_val_DT = np.empty([num_folds, num_DT_classifiers])

true_labels_DT = [[] for _ in range(num_DT_classifiers)]
predicted_labels_DT = [[] for _ in range(num_DT_classifiers)]


# Loop through the folds
for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    
    # Extract the train and validation data for this fold
    x_train_fold, x_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    for j, clf in enumerate(DT_classifiers): 

        # Fit the classifier on the training data
        clf.fit(x_train_fold, y_train_fold)

        # Plot decision tree structure
        #plt.figure(figsize=(5, 3))
        #tree.plot_tree(clf, feature_names=['A', 'C', 'DG'], class_names=['0', '1'])
        #plt.show()

        # Evaluate your metric of choice 
        y_DT_pred = clf.predict_proba(x_val_fold)[:, 1] # Probability of class 1

        AUC_val_DT[i,j] = roc_auc_score(y_val_fold, y_DT_pred)

        # Predict the labels for the validation set
        pred_labels_DT = clf.predict(x_val_fold)

        # Append true labels and predicted labels for this fold
        true_labels_DT[j].extend(y_val_fold)
        predicted_labels_DT[j].extend(pred_labels_DT)

#Average over all folds
average_DT_auc = np.mean(AUC_val_DT, axis=0) 

# Loop through each classifier
for j, clf in enumerate(DT_classifiers):
    
    print(f'\n\nFor {2 * (j + 1)} leaf nodes:\n')

    print(f'Average AUC: {average_DT_auc[j]}')

    # Compute confusion matrix
    conf_matrix_DT = confusion_matrix(true_labels_DT[j], predicted_labels_DT[j])
    print("\nConfusion Matrix:")
    print(conf_matrix_DT)

    tn, fp, fn, tp = conf_matrix_DT.ravel()
    sens = tp/(tp + fn)
    spec= tn/(tn + fp)

    print(f'Sensitivity: {sens}\nSpecificity: {spec}')



# ---K-Nearest Neighbors---

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
    
    # Extract the train and validation data for this fold
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

    # Compute confusion matrix
    conf_matrix_KNN = confusion_matrix(true_labels[j], predicted_labels[j])
    print("\nConfusion Matrix:")
    print(conf_matrix_KNN)

    tn, fp, fn, tp = conf_matrix_KNN.ravel()
    sens = tp/(tp + fn)
    spec= tn/(tn + fp)

    print(f'Sensitivity: {sens}\nSpecificity: {spec}')



# Final classifier
classifier = KNeighborsClassifier(n_neighbors = 3)

# Training this classifier on the entire dataset
classifier = classifier.fit(X, y)

# Saving classifier
dir = 'features'
filename = os.path.join(dir, 'groupK_classifier.sav')

pickle.dump(classifier, open(filename, 'wb'))



# Prediction on test data
test_pred = classifier.predict(X_test)

# Compute and visualize confusion matrix
conf_matrix = confusion_matrix(y_test, test_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Not Melanoma', 'Melanoma'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout() 
plt.show()

print(classification_report(y_test, test_pred, target_names=['Not Melanoma', 'Melanoma']))