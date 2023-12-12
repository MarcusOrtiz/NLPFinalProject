import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from .train import collate_batch
from .load_datasets import load_datasets
from .qa import create_model
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# Constants
MODEL_PATH = 'path/to/your/saved_model.pth'
TEST_DATA_NAME = 'path/to/your/test_data.json'
BATCH_SIZE = 16
PADDING_INDEX = 1

# Function to pad sequences

# Load test data
_, _, test_data, vocab = load_datasets(TRAIN_DATA_NAME, VAL_DATA_NAME, TEST_DATA_NAME)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# Load model
model = create_model(len(vocab))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Prepare for evaluation
all_labels = []
all_predictions = []

# Evaluate the model
with torch.no_grad():
    for questions, answers, labels in test_loader:
        predictions = model(questions, answers)
        _, predicted_classes = predictions.max(1)
        all_labels.extend(labels.tolist())
        all_predictions.extend(predicted_classes.tolist())

# Convert to numpy arrays for sklearn metrics
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
print(f"Accuracy: {accuracy}")
print(classification_report(all_labels, all_predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
