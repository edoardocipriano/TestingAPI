import torch
import torch.nn as nn
import torch.optim as optim

class DiabetesModel(nn.Module):
    def __init__(self, input_size):
        super(DiabetesModel, self).__init__()
        # More balanced network structure
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        
        # Fine-tuned regularization - higher dropout rates to prevent overfitting
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)  # Increased dropout in the middle of the network
        self.dropout3 = nn.Dropout(0.3)
        
        # Batch normalization for training stability
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)
        self.batch_norm4 = nn.BatchNorm1d(16)
        
        # Residual connection weights - reduced complexity
        self.res1 = nn.Linear(input_size, 32)
        self.res2 = nn.Linear(32, 64)
        self.res3 = nn.Linear(64, 32)

    def forward(self, x):
        # First block with residual connection
        identity1 = self.res1(x)
        x = torch.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = x + identity1  # Residual connection
        x = self.dropout1(x)
        
        # Second block with residual connection
        identity2 = self.res2(x)
        x = torch.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = x + identity2  # Residual connection
        x = self.dropout2(x)  # Higher dropout here to prevent overfitting
        
        # Third block with residual connection
        identity3 = self.res3(x)
        x = torch.relu(self.fc3(x))
        x = self.batch_norm3(x)
        x = x + identity3  # Residual connection
        x = self.dropout3(x)
        
        # Final layers
        x = torch.relu(self.fc4(x))
        x = self.batch_norm4(x)
        x = self.output(x)
        return x
    
def create_model(input_size):
    return DiabetesModel(input_size)