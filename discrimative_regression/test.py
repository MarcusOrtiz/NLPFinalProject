import torch
from torch.utils.data import DataLoader
from .load_datasets import load_datasets  # Adjust import according to your project structure
from .qa import create_model  # Adjust import according to your project structure
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .train import collate_batch
import numpy as np
from pathlib import Path
import math

# Constants
MODEL_PATH = Path('./model_ending_1.pth').resolve()
TRAIN_DATA_NAME = 'unique_answers/train_data.json'
VAL_DATA_NAME = 'unique_answers/val_data.json'
TEST_DATA_NAME = 'unique_answers/test_data.json'
BATCH_SIZE = 32


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
    for questions, answers, scores in test_loader:
        predictions = model(questions, answers)
        for i in range(len(scores)):
            if 1 <= scores[i] < 1.75:
                scores[i] = 1
            elif 1.75 <= scores[i] < 2.5:
                scores[i] = 2
            elif 2.5 <= scores[i] < 3.25:
                scores[i] = 3
            elif 3.25 <= scores[i] < 4:
                scores[i] = 4
        all_labels.extend(scores.tolist())
        all_predictions.extend(predictions.squeeze().tolist())

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Calculate metrics
mse = mean_squared_error(all_labels, all_predictions)
rmse = math.sqrt(mse)
mae = mean_absolute_error(all_labels, all_predictions)
print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}")

# Plotting Predictions vs Actuals
plt.scatter(all_labels, all_predictions, alpha=0.5)
plt.title('Predictions vs Actuals')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.show()
