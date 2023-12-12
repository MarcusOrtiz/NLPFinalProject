import torch.nn as nn
import torch.optim as optim
import torch
from .load_datasets import load_datasets
from torch.nn.utils.rnn import pad_sequence
from .qa import create_model
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import math

# TRAIN_DATA_NAME = 'train_marcus.json'
# VAL_DATA_NAME = 'train_marcus.json'
# TEST_DATA_NAME = 'train_marcus.json'
# TRAIN_DATA_NAME = 'train_formatted_output_w_comma.json'
# VAL_DATA_NAME = 'valid_formatted_output_w_comma.json'
# TEST_DATA_NAME = 'test_formatted_output_w_comma.json'
TRAIN_DATA_NAME = 'unique_answers/train_data_classification.json'
VAL_DATA_NAME = 'unique_answers/val_data_classification.json'
TEST_DATA_NAME = 'unique_answers/test_data_classification.json'

BATCH_SIZE = 16
SHUFFLE = True
EPOCHS = 40
LEARNING_RATE = 0.003
PATIENCE = 35
MOMENTUM = 0.9
WEIGHT_DECAY = 0.01
PADDING_INDEX = 1


def collate_batch(batch):
    questions, answers, scores = zip(*batch)

    # Pad questions and answers to have the same length within each batch
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=PADDING_INDEX)
    answers_padded = pad_sequence(answers, batch_first=True, padding_value=PADDING_INDEX)
    scores = torch.tensor(scores, dtype=torch.float)

    return questions_padded, answers_padded, scores

def batch_correct(predictions, scores):
    total_correct = 0
    for i in range(len(scores)):
        if scores[i] == 0 and 0 <= predictions[i] < 0.75:
            total_correct += 1
        elif scores[i] == 1 and 0.75 <= predictions[i] < 1.5:
            total_correct += 1
        elif scores[i] == 2 and 1.5 <= predictions[i] < 2.25:
            total_correct += 1
        elif scores[i] == 3 and 2.25 <= predictions[i] <= 3:
            total_correct += 1

    return total_correct


def train():
    train_data, val_data, test_data, vocab = load_datasets(TRAIN_DATA_NAME, VAL_DATA_NAME, TEST_DATA_NAME)

    model = create_model(len(vocab))

    qa_train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE, collate_fn=collate_batch)
    qa_val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    qa_test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # Define the loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    ### TRAINING ###
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for questions, answers, scores in qa_train_loader:
            # Forward pass
            predictions = model(questions, answers).squeeze()
            scores = scores.squeeze()
            # print(f'train predictions: {predictions}')
            # print(f'train scores: {scores}')
            # Compute the loss
            loss = loss_fn(10*predictions, 10*scores)

            # loss = torch.mean((2*predictions - 2*scores) ** 4 * torch.abs(2*predictions - 2*scores))
            # loss = torch.mean((predictions - scores) ** 4) + torch.mean((predictions - scores) ** 1/4)            # Backward pass and optimization
            optimizer.zero_grad()  # Clear existing gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            train_loss += loss.item()

        train_loss = train_loss / len(qa_train_loader)

        ### VALIDATION ###
        model.eval()
        val_accuracy = 0
        val_loss = 0
        total_scores = 0
        with torch.no_grad():
            printCount = 4
            for questions, answers, scores in qa_val_loader:
                predictions = model(questions, answers).squeeze()
                scores = scores.squeeze()
                # print(f'val predictions: {predictions}')
                # print(f'val scores: {scores}')
                loss = loss_fn(predictions, scores)

                # loss = torch.mean((2 * predictions - 2 * scores) ** 3 * torch.abs(2 * predictions - 2 * scores))
                # loss = torch.mean((predictions - scores) ** 8)  # Backward pass and optimization
                val_loss += loss.item()

                # accuracy
                val_accuracy += batch_correct(predictions, scores)
                total_scores += len(scores)
                if printCount > 0:
                    print(f'val predictions: {predictions}')
                    print(f'val scores: {scores}')
                    #     print(f'val diff: {diff}')
                    #     print('loss: ', loss.item())
                    #     print('val loss: ', val_loss)
                    #     print(f'val accurate: {accurate}')
                    #     print(f'val accuracy: {val_accuracy}')
                    #     print(f'val total_scores: {total_scores}')
                    printCount -= 1

        val_loss = val_loss / len(qa_val_loader)
        val_accuracy = val_accuracy / total_scores

        # Print statistics
        print(f"Epoch {epoch},"
              f"Training Loss: {train_loss}, Validation Loss: {val_loss}, "
              f"Val Accuracy: {val_accuracy}")

        # Early Stopping
        if val_loss < best_val_loss and epoch > 12:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the model if it's the best so far
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve == PATIENCE:
            print(f'Early stopping! Epoch: {epoch}')
            break


        ### TEST ###
        all_labels = []
        all_predictions = []
        for questions, answers, scores in qa_test_loader:
            predictions = model(questions, answers)
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




    torch.save(model.state_dict(), 'model_ending_1.pth')



if __name__ == '__main__':
    train()
