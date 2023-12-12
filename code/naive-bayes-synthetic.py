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

# # Print the combined dataset
# print("Combined Dataset:")
# print(final_df)

# Combine 'Question' and 'Answer' columns into a single text column
final_df['Combined_Text'] = final_df['Question'] + ' ' + final_df['Answer']

# Given probability distribution for labels
label_probs = [0.326709, 0.176647, 0.178767, 0.317877]
labels = np.arange(len(label_probs))

# Use CountVectorizer to create a bag-of-words representation
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(final_df['Combined_Text'])

# Extract the vocabulary array from CountVectorizer
bow_vocab = np.array(vectorizer.get_feature_names_out())

def generate_text(label, bow_vocab, num_words_range=(5, 15)):
    num_words_question = np.random.randint(*num_words_range)
    num_words_answer = np.random.randint(*num_words_range)

    # Randomly split the total number of words between question and answer
    total_words = num_words_question + num_words_answer
    split = np.random.choice(total_words, num_words_question, replace=False)

    # Create a binary mask to indicate which words belong to the question
    mask = np.zeros(total_words, dtype=int)
    mask[split] = 1

    # Use the mask to split the words into question and answer
    words = np.random.choice(bow_vocab, size=total_words)
    question = " ".join(words[mask == 1])
    answer = " ".join(words[mask == 0])

    return {"Question": question, "Answer": answer, "Label": label}

# Number of samples to generate
num_samples = 17307

# Generate random questions and answers
generated_data = []
for _ in range(num_samples):
    sampled_label = np.random.choice(labels, p=label_probs)
    entry = generate_text(sampled_label, bow_vocab)
    generated_data.append(entry)

# # Display the generated data
# for i, entry in enumerate(generated_data):
#     print(f"Sample {i + 1}:")
#     print("Question:", entry["Question"])
#     print("Answer:", entry["Answer"])
#     print("Answer:", entry["Label"])
#     print("=" * 30)

# Save the generated data to a CSV file
synthetic_data_df = pd.DataFrame(generated_data)
synthetic_data_df.to_csv('/workspaces/NLPFinalProject/data/synthetic_data_full.csv', index=False)
print("Synthetic data saved to synthetic_data_NB.csv")

# Load the synthetic data from the CSV file
synthetic_data_path = '/workspaces/NLPFinalProject/data/synthetic_data_full.csv'
synthetic_data = pd.read_csv(synthetic_data_path)

# Split the data into features (X) and labels (y)
X = synthetic_data[['Question', 'Answer']]
y = synthetic_data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use CountVectorizer to create a bag-of-words representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train['Question'] + ' ' + X_train['Answer'])
X_test_bow = vectorizer.transform(X_test['Question'] + ' ' + X_test['Answer'])

# Train a Multinomial Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_bow, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_bow)

# Evaluate the accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Print the results
print("Accuracy:", accuracy)
print("F1 Score:", f1)


