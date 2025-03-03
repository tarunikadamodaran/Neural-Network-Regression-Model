## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Tarunika D
### Register Number: 212223040227
```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('Book1.csv')
X = dataset1[['INPUT']].values
y = dataset1[['OUTPUT']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)
    return x
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr=0.001)
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    ai_brain.history = {'loss': []}
    history = {'loss': []}
    for epoch in range(epochs):
        y_pred = ai_brain(X_train)
        loss = criterion(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history['loss'].append(loss.item())
        if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
    return history
with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
history = train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)
loss_df = pd.DataFrame(history)
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```
## Dataset Information

Include screenshot of the dataset

0![Screenshot 2025-03-03 114202](https://github.com/user-attachments/assets/e78caad8-8369-4441-94ab-7fdb07f09f3d)


## OUTPUT

![Screenshot 2025-03-03 115054](https://github.com/user-attachments/assets/7cb69d35-8df8-4914-83c0-aa143be0c117)


### New Sample Data Prediction


![Screenshot 2025-03-03 115109](https://github.com/user-attachments/assets/7c5317a5-84ca-434c-b974-e87ca3887531)


## RESULT

The neural network regression model was successfully trained and evaluated.
