import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score

train_path = r'/workspaces/NLPFinalProject/data/train_formatted_output.json'
test_path = r'/workspaces/NLPFinalProject/data/test_formatted_output.json'
valid_path = r'/workspaces/NLPFinalProject/data/valid_formatted_output.json'

# Read the data from the JSON files
with open(train_path, 'r') as train_file:
    train_data = [json.loads(line) for line in train_file]

with open(test_path, 'r') as test_file:
    test_data = [json.loads(line) for line in test_file]

with open(test_path, 'r') as valid_file:
    valid_data = [json.loads(line) for line in valid_file]

# Create DataFrames from the lists of dictionaries
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
valid_df = pd.DataFrame(valid_data)

# Combine the two datasets
final_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)

# Split the combined final_df into train and temp datasets (62.5% train, 37.5% temp)
train_df, temp_df = train_test_split(final_df, test_size=0.375, random_state=42)

# Further split the temp_df into test and valid datasets (22% test, 19.5% valid)
test_df, valid_df = train_test_split(temp_df, test_size=0.52, random_state=42)

# Print the sizes of the datasets
print("Train Dataset Size:", len(train_df))
print("Test Dataset Size:", len(test_df))
print("Valid Dataset Size:", len(valid_df))

save_path = '/workspaces/NLPFinalProject/code/synthetic/'

# Save the train_df as a JSON file
train_df.to_json(save_path + 'train_s.json', orient='records', lines=True)

# Save the test_df as a JSON file
test_df.to_json(save_path + 'test_s.json', orient='records', lines=True)

# Save the valid_df as a JSON file
valid_df.to_json(save_path + 'valid_s.json', orient='records', lines=True)
