import pandas as pd
import nltk
import json
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load data from JSON file
# change the path when you use it
json_file_path = r"C:\Users\suimp\OneDrive\Desktop\Duke University\Class\2023-Fall\IDS 703 Introduction to Natural Language Processing\Final Project\NLPFinalProject\data\train_formatted_output.json"

# Download WordNet resource
nltk.download("wordnet")

# ---------------------------------------------------------------
data = []
with open(json_file_path, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

# Convert data to DataFrame
df = pd.DataFrame(data)

# Set display width to show entire content of the 'Answer' column
pd.set_option("display.max_colwidth", None)

# ---------------------------------------------------------------
# Stemming
stemmer = PorterStemmer()
df["Question"] = df["Question"].apply(
    lambda x: " ".join([stemmer.stem(word) for word in x.split()])
)
df["Answer"] = df["Answer"].apply(
    lambda x: " ".join([stemmer.stem(word) for word in x.split()])
)
print(df)

"""
lemmatization (from the below code) is not performed successfully, returning the same df
does not seem to be a difference between stemming and lemmatization
"""

"""
# Lemmatization
lemmatizer = WordNetLemmatizer()
df["Question"] = df["Question"].apply(
    lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
)
df["Answer"] = df["Answer"].apply(
    lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
)

# Print the first few rows of the DataFrame to check lemmatization
print("Lemmatized DataFrame:")
print(df.head())
"""

default_filters = '!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'
filters = default_filters.replace(".", "")

# ---------------------------------------------------------------
# Tokenization
tokenizer_all = Tokenizer(filters=filters)
combined_text = df["Question"] + " " + df["Answer"] + " " + df["Average"].astype(str)
tokenizer_all.fit_on_texts(combined_text)
total_words = len(tokenizer_all.word_index) + 1

# Convert text to sequences
sequences = tokenizer_all.texts_to_sequences(combined_text)

# # Print the first few sequences
# print("Sequences:")
# print(sequences[:2])

"""
padding the sequence with zeroes so that the sequences are of the same length
"""

# ---------------------------------------------------------------
# Assuming 'sequences' is your list of tokenized sequences
# Set the maximum sequence length
# You can set this to the length of the longest sequence or a predefined length
max_sequence_length = max([len(seq) for seq in sequences])

# Pad the sequences
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding="post")
