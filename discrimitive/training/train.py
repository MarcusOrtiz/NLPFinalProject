from discrimitive.model.TunedQA import model
from Dataset import QA_loader
import torch.nn as nn
import torch.optim as optim
import torch

# Assuming 'train_loader' is your DataLoader instance
# 'model' is an instance of QAModel
# 'criterion' is your loss function, and 'optimizer' is defined
def train():
    qa_train_loader = QA_loader('../../data/train_output.json')
    qa_val_loader = QA_loader('../../data/valid_output.json')

    num_epochs = 50
    # Define the loss function and optimizer
    rate_learning = 0.001
    optimizer = optim.Adam(model.parameters(), lr=rate_learning)
    loss_fn = nn.CrossEntropyLoss()



    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for questions, answers, scores in qa_train_loader:
            # Forward pass
            predictions = model(questions, answers)

            # Compute the loss
            loss = loss_fn(predictions, scores)

            # Backward pass and optimization
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
            for questions, answers, scores in qa_val_loader:
                predictions = model(questions, answers)
                loss = loss_fn(predictions, scores)
                val_loss += loss.item()

                # accuracy
                diff = torch.abs(predictions - scores)
                accurate = torch.where(diff < 0.5, torch.ones_like(diff), torch.zeros_like(diff))
                val_accuracy += torch.sum(accurate).item()
                total_scores += len(scores)

        val_loss = val_loss / len(qa_val_loader)
        val_accuracy = val_accuracy / total_scores



        # Print statistics
        print(f"Epoch {epoch},"
              f"Training Loss: {train_loss}, Validation Loss: {val_loss}, "
              f"Val Accuracy: {val_accuracy}")

        # Save the model checkpoint if needed
        # torch.save(model.state_dict(), 'model_checkpoint.pth')