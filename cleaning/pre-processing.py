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
json_file_path = r"C:\Users\suimp\OneDrive\Desktop\Duke University\Class\2023-Fall\IDS 703 Introduction to Natural Language Processing\Final Project\NLPFinalProject\data\train_formatted_output.json"

# Download WordNet resource
# nltk.download("wordnet")

data = []
with open(json_file_path, "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))


# Convert data to DataFrame
df = pd.DataFrame(data)

# Set display width to show entire content of the 'Answer' column
pd.set_option("display.max_colwidth", None)

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


# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["Question"] + " " + df["Answer"])
total_words = len(tokenizer.word_index) + 1


# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df["Question"] + " " + df["Answer"])

# # Print the first few sequences
# print("Sequences:")
# print(sequences[:2])

"""
padding the sequence with zeroes so that the sequences are of the same length
"""
# Padding sequences
# padded_sequences = pad_sequences(sequences)

# Print the padded sequences
# print("Padded Sequences:")
# print(padded_sequences[:2])
# print("the type of sequences are", type(padded_sequences))

# "In this rapidly changing jobs market the Australian Government is supporting businesses and those Australians looking for work. While many businesses have been adversely affected by COVID-19 and are reducing their workforces, there are some areas of the economy which have an increased demand for workers. This includes jobs in health and care sectors, transport and logistics, some areas of retail, mining and mining services, manufacturing, agriculture and government sectors, among others. The Jobs Hub helps you find advertised vacancies."
