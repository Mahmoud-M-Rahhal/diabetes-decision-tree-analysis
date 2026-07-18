import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# Load dataset and preview structure

pima_df = pd.read_csv('diabetes.csv')

# Preview first rows
pima_df.head()

# Descriptive statistics

pima_df.describe()

# Summary of column types and non-null counts
pima_df.info()

# Dataset dimensions
shape = pima_df.shape
print("Dataset shape (rows, columns):", shape)

# Missing value counts per column

pima_df.isnull().sum()

# Split dataset into features and target variable

# Feature matrix
X = pima_df.loc[:, ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]

# Target variable
y = pima_df["Outcome"]

# Train/test split (80/20) with fixed random state

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Configure Decision Tree classifier
clf = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)

# Fit model
clf = clf.fit(X_train, y_train)

# Generate predictions on test set
dtree_y_pred = clf.predict(X_test)

# Classification metrics (weighted where applicable)

dtree_y_true = y_test

accuracy = metrics.accuracy_score(y_test, dtree_y_pred)
precision = metrics.precision_score(y_test, dtree_y_pred, average = 'weighted')
recall = metrics.recall_score(y_test, dtree_y_pred, average = 'weighted')
f1_score = metrics.f1_score(y_test, dtree_y_pred, average = 'weighted')

print("Model Performance Summary")
print("Accuracy:", accuracy)
print("Precision (weighted):", precision)
print("Recall (weighted):", recall)
print("F1-score (weighted):", f1_score)

# ROC curve and AUC computation

from sklearn.metrics import roc_curve, auc

dtree_auc = 0

def plot_roc(dt_y_true, dt_probs):
    
    # Compute FPR, TPR, and thresholds
    
    dtree_fpr, dtree_tpr, threshold = metrics.roc_curve(dt_y_true, dt_probs)
    
    # Compute ROC-AUC score
    dtree_auc_val = metrics.auc(dtree_fpr, dtree_tpr)
    
    # Print ROC-AUC score
    print('ROC-AUC = %0.2f' % dtree_auc_val)
    
    
    # Plot ROC curve
    
    plt.plot(dtree_fpr, dtree_tpr, label = 'AUC=%0.2f'%dtree_auc_val, color = 'darkorange')
    plt.legend(loc = 'lower right')
    plt.plot([0,1], [0,1], 'b--')
    plt.xlim([0,1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return dtree_auc_val

# Predicted positive-class probabilities

dtree_probs = clf.predict_proba(X_test) [:,1]
dtree_auc = plot_roc(dtree_y_true, dtree_probs)

# Stratified K-Fold evaluation (k = 2 to 10)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

max_acc, max_k = 0, 0

for k in range(2, 11):

    skfold = StratifiedKFold(n_splits = k, random_state = 100, shuffle = True)

    results_skfold_acc = (cross_val_score(clf, X, y, cv = skfold)).mean() * 100.0
    
    # Track best mean accuracy and corresponding k
    if results_skfold_acc > max_acc:
        
        max_acc = results_skfold_acc
        max_k = k

    print("Cross-validation accuracy (%d folds): %.2f%%" % (k, results_skfold_acc))

best_accuracy =  max_acc
best_k_fold =  max_k

print("Best cross-validation accuracy (%):", best_accuracy)
print("Optimal number of folds:", best_k_fold)

