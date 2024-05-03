import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
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


# Prepare cross-validation
num_folds = 5
skf = StratifiedKFold(n_splits=num_folds)

#Different classifiers to test out
classifiers = [
    KNeighborsClassifier(n_neighbors=1),
    KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier(n_neighbors=5)
]
num_classifiers = len(classifiers)

AUC_val = np.empty([num_folds,num_classifiers])

true_labels = [[] for _ in range(num_classifiers)]
predicted_labels = [[] for _ in range(num_classifiers)]

# Loop through the folds
for i, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
    
    # Extract the train and test data for this fold
    x_train_fold, x_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

    for j, clf in enumerate(classifiers): 
        
        # Fit the classifier on the training data
        clf.fit(x_train_fold, y_train_fold)

        # Predict the labels for the validation set
        pred = clf.predict(x_val_fold)
    
        #Evaluate your metric of choice (accuracy is probably not the best choice)
        AUC_val[i,j] = roc_auc_score(y_val_fold, pred)

        # Append true labels and predicted labels for this fold
        true_labels[j].extend(y_val_fold)
        predicted_labels[j].extend(pred)

#Average over all folds
average_acc = np.mean(AUC_val, axis=0) 

print('Classifier 1 average accuracy={:.3f} '.format(average_acc[0]))
print('Classifier 2 average accuracy={:.3f} '.format(average_acc[1]))


# Loop through each classifier
for j, clf in enumerate(classifiers):
    
    # Train the classifier on the entire training set
    clf.fit(X_train[['A', 'C', 'DG']], y_train)

    # Make predictions on the test set
    y_pred_test = clf.predict_proba(X_test[['A', 'C', 'DG']])[:, 1]  # Probability of class 1

    # Calculate AUC score on test set
    auc_score_test = roc_auc_score(y_test, y_pred_test)
    print(f"\nAUC Score for classifier {j+1} on test set: {auc_score_test}")

    # Compute confusion matrix
    conf_matrix = confusion_matrix(true_labels[j], predicted_labels[j])
    print("\nConfusion Matrix:")
    print(conf_matrix)


# Final classifier
#classifier = KNeighborsClassifier(n_neighbors = 5)

# Training this classifier on the entire dataset
#classifier = classifier.fit(X,y)

# Saving classifier
#filename = 'groupK_classifier.sav'
#pickle.dump(classifier, open(filename, 'wb'))