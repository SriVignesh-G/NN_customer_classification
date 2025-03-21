# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/3689221b-1280-4ca9-9760-d594e5982d50)


## DESIGN STEPS

### STEP 1:
Data Preprocessing: Clean, normalize, and split data into training, validation, and test sets.

### STEP 2:
Model Design:

Input Layer: Number of neurons = features.

Hidden Layers: 2 layers with ReLU activation.

Output Layer: 4 neurons (segments A, B, C, D) with softmax activation.

### STEP 3:
Model Compilation: Use categorical cross-entropy loss, Adam optimizer, and track accuracy.

### STEP 4:
Training: Train with early stopping, batch size (e.g., 32), and suitable epochs.

### STEP 5:
Evaluation: Assess using accuracy, confusion matrix, precision, and recall.

### STEP 6:
Optimization: Tune hyperparameters (layers, neurons, learning rate, batch size).


## PROGRAM

### Name: 
### Register Number:

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x)
        return x
        
```
```python
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)
```
```python
def train(trainloader,model,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch,y_batch in train_loader:
      optimizer.zero_grad()
      y_pred = model(X_batch)
      loss = criterion(y_pred,y_batch)
      loss.backward()
      optimizer.step()
    if (epoch+1)%20 == 0:
      print(f"Epoch [{epoch+1}/{epochs}],Loss:{loss.item():.4f}")
```

## Dataset Information

![image](https://github.com/user-attachments/assets/d8718f6f-1464-4476-a599-45dd261c6fd4)


## OUTPUT

### Confusion Matrix

![download](https://github.com/user-attachments/assets/8de88631-256b-4b0c-8c07-dd1d4567596a)


### Classification Report

![image](https://github.com/user-attachments/assets/03608949-18a5-4ef6-9d75-e156258ff027)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/a9a35a40-c13a-46ec-947d-da48c148e700)


## RESULT
Thus a neural network classification model for the given dataset is executed successfully.
