import torch.nn as nn
import torch.optim as optim
import torch
from .load_datasets import load_datasets
from torch.nn.utils.rnn import pad_sequence
from .qa import create_model
from torch.utils.data import Dataset, DataLoader

# TRAIN_DATA_NAME = 'train_marcus.json'
# VAL_DATA_NAME = 'train_marcus.json'
# TEST_DATA_NAME = 'train_marcus.json'
# TRAIN_DATA_NAME = 'train_formatted_output_w_comma.json'
# VAL_DATA_NAME = 'valid_formatted_output_w_comma.json'
# TEST_DATA_NAME = 'test_formatted_output_w_comma.json'
# TRAIN_DATA_NAME = 'unique_answers/train_data_classification.json'
# VAL_DATA_NAME = 'unique_answers/val_data_classification.json'
# TEST_DATA_NAME = 'unique_answers/test_data_classification.json'
TRAIN_DATA_NAME = '../code/synthetic/train_s.json'
VAL_DATA_NAME = '../code/synthetic/valid_s.json'
TEST_DATA_NAME = '../code/synthetic/test_s.json'


BATCH_SIZE = 32
SHUFFLE = True
EPOCHS = 30
LEARNING_RATE = 0.005
PATIENCE = 45
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
PADDING_INDEX = 1


def collate_batch(batch):
    questions, answers, scores = zip(*batch)

    # Pad questions and answers to have the same length within each batch
    questions_padded = pad_sequence(questions, batch_first=True, padding_value=PADDING_INDEX)
    answers_padded = pad_sequence(answers, batch_first=True, padding_value=PADDING_INDEX)
    scores = torch.tensor(scores, dtype=torch.long)

    return questions_padded, answers_padded, scores


def train():
    train_data, val_data, test_data, vocab = load_datasets(TRAIN_DATA_NAME, VAL_DATA_NAME, TEST_DATA_NAME)

    model = create_model(len(vocab))

    qa_train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE, collate_fn=collate_batch)
    qa_val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    qa_test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

    # Define the loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val_accuracy = float('-inf')
    epochs_no_improve = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for questions, answers, scores in qa_train_loader:
            # Forward pass
            predictions = model(questions, answers)
            scores = scores

            # print(torch.shape(predictions))
            # print(torch.shape(scores))

            loss = loss_fn(predictions, scores)

            optimizer.zero_grad()  # Clear existing gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            train_loss += loss.item()

        train_loss = train_loss / len(qa_train_loader)

        model.eval()
        val_accuracy = 0
        val_loss = 0
        total_scores = 0
        with torch.no_grad():
            printCount = 4
            for questions, answers, scores in qa_val_loader:
                predictions = model(questions, answers)
                scores = scores
                loss = loss_fn(predictions, scores)

                val_loss += loss.item()

                # accuracy
                _, predicted_classes = predictions.max(1)
                correct_predictions = predicted_classes.eq(scores)
                val_accuracy += correct_predictions.sum().item()
                total_scores += len(scores)

                # classes = (0, 1, 2, 3)
                # for cls in classes:
                #     if cls not in predicted_classes: loss += 10
                if printCount > 0:
                    print(f'probs: {predictions}')
                    print(f'val predictions: {predicted_classes}')
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
        if val_accuracy > best_val_accuracy and epoch > 0 and val_accuracy > 0.3:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            # Save the model if it's the best so far
            torch.save(model.state_dict(), 'best_model_classification.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve == PATIENCE:
            print(f'Early stopping! Epoch: {epoch}')
            break

    torch.save(model.state_dict(), 'model_ending_classification.pth')


if __name__ == '__main__':
    train()
