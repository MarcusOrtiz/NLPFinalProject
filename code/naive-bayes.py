import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

train_path = r'/workspaces/NLPFinalProject/data/train_formatted_output.json'
test_path = r'/workspaces/NLPFinalProject/data/test_formatted_output.json'

# Read the data from the JSON files
with open(train_path, 'r') as train_file:
    train_data = [json.loads(line) for line in train_file]

with open(test_path, 'r') as test_file:
    test_data = [json.loads(line) for line in test_file]

# Create DataFrames
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Bin the 'Score' values into discrete classes
num_bins = 5  # You can adjust the number of bins based on your preference
train_df['Score'] = pd.cut(train_df['Score'], bins=num_bins, labels=False)
test_df['Score'] = pd.cut(test_df['Score'], bins=num_bins, labels=False)


# Count the number of unique values in the 'Score' column
unique_score_values = train_df['Score'].nunique()

# Print the result
print("Number of unique values in 'Score':", unique_score_values)

# Convert 'Score' column to string type
train_df['Score'] = train_df['Score'].astype(str)
test_df['Score'] = test_df['Score'].astype(str)

# Concatenate question and answer for both train and test
X_train = train_df['Question'] + ' ' + train_df['Answer']
X_test = test_df['Question'] + ' ' + test_df['Answer']

# Use 'Score' as the target variable
y_train = train_df['Score']
y_test = test_df['Score']

# Convert text data to a bag-of-words representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

def find_best_alpha(X_train, y_train, X_test, y_test):
    best_alpha = 0
    best_score = 0
    for alpha in np.arange(0.1, 1.1, 0.1):
        model = MultinomialNB(alpha=alpha)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_alpha = alpha
            best_score = score
    print("Best alpha: {}".format(best_alpha))
    print("Best score: {}".format(best_score))
    return best_alpha

# Train a naive Bayes classifier
best_alpha = find_best_alpha(X_train_bow, y_train, X_test_bow, y_test)
classifier = MultinomialNB(alpha = best_alpha)
classifier.fit(X_train_bow, y_train)

# Probability distribution of rating categories in the train dataset
train_rating_distribution = train_df['Score'].value_counts(normalize=True).sort_index()
print("Probability distribution of rating categories in the train dataset:")
print(train_rating_distribution)

# Make predictions on the test set
y_pred = classifier.predict(X_test_bow)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report (including F1 score)
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# AUC Score
y_prob = classifier.predict_proba(X_test_bow)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print(f'AUC Score for Naive Bayes: {roc_auc:.2f}')



