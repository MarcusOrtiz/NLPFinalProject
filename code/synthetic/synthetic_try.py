import pandas as pd
from sklearn.model_selection import train_test_split

# Load the synthetic data from the CSV file
synthetic_data_path = '/workspaces/NLPFinalProject/data/synthetic_data_full.csv'
synthetic_data = pd.read_csv(synthetic_data_path)

# Split the data into train, test, and validation sets
train_data, temp_data = train_test_split(synthetic_data, test_size=0.375, random_state=42)
test_data, val_data = train_test_split(temp_data, test_size=0.52, random_state=42)

# Define paths for the new JSON files
train_json_path = '/workspaces/NLPFinalProject/code/synthetic/synthetic_train_data.json'
test_json_path = '/workspaces/NLPFinalProject/code/synthetic/synthetic_test_data.json'
val_json_path = '/workspaces/NLPFinalProject/code/synthetic/synthetic_val_data.json'

# Save the split data to JSON files as arrays
train_data.to_json(train_json_path, orient='records')
test_data.to_json(test_json_path, orient='records')
val_data.to_json(val_json_path, orient='records')

print("Data split and saved to JSON files as arrays.")

