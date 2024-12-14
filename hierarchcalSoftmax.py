import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from hierarchicalsoftmax import SoftmaxNode, HierarchicalSoftmaxLoss
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# Load the 20 newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X, y = newsgroups.data, newsgroups.target

# Preprocess the text data using a bag-of-words model with max 10,000 features
vectorizer = CountVectorizer(max_features=10000)
X = vectorizer.fit_transform(X).toarray()

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to compute softmax probabilities
def regular_softmax(X, W, b):
    scores = np.dot(X, W.T) + b
    return softmax(scores, axis=1)

# Initialize weights and biases for softmax classifier
num_classes = len(np.unique(y))
W = np.random.randn(num_classes, X.shape[1])
b = np.zeros(num_classes)

# Train softmax
learning_rate = 0.01
num_epochs = 10

softmax_start_time = time.time()

for epoch in range(num_epochs):
    probs = regular_softmax(X_train, W, b)  
    gradient_W = np.dot((probs - np.eye(num_classes)[y_train]).T, X_train) / X_train.shape[0]
    gradient_b = np.mean(probs - np.eye(num_classes)[y_train], axis=0)
    gradient_W = gradient_W.T
    W -= learning_rate * gradient_W.T
    b -= learning_rate * gradient_b

# Evaluate softmax on test data
y_pred_softmax = np.argmax(regular_softmax(X_test, W, b), axis=1)
accuracy_softmax = accuracy_score(y_test, y_pred_softmax)

softmax_end_time = time.time()
softmax_execution_time = softmax_end_time - softmax_start_time

# Model for Hierarchical Softmax
class HierarchicalSoftmaxModel(nn.Module):
    def __init__(self, input_size, root):
        super().__init__()
        self.linear = nn.Linear(input_size, root.layer_size)
        self.root = root

    def forward(self, x):
        return self.linear(x)

# Create a hierarchical tree structure for classes
root = SoftmaxNode("root")
for i in range(num_classes):
    SoftmaxNode(str(i), parent=root)

# Initialize hierarchical softmax model and loss function
model = HierarchicalSoftmaxModel(X.shape[1], root)
criterion = HierarchicalSoftmaxLoss(root=root)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Convert training data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

hierarchical_start_time = time.time()

# Train hierarchical softmax model using Adam optimizer
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Evaluate hierarchical softmax on test data
X_test_tensor = torch.FloatTensor(X_test)
outputs = model(X_test_tensor)
_, y_pred_hierarchical = torch.max(outputs, 1)
accuracy_hierarchical = accuracy_score(y_test, y_pred_hierarchical.numpy())

hierarchical_end_time = time.time()
hierarchical_execution_time = hierarchical_end_time - hierarchical_start_time

print(f"Regular Softmax Accuracy: {accuracy_softmax:.4f}")
print(f"Hierarchical Softmax Accuracy: {accuracy_hierarchical:.4f}")
print(f"Regular Softmax Execution Time: {softmax_execution_time:.2f} seconds")
print(f"Hierarchical Softmax Execution Time: {hierarchical_execution_time:.2f} seconds")

# Plotting the accuracies and execution times
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy plot
labels = ['Softmax', 'Hierarchical Softmax']
accuracies = [accuracy_softmax, accuracy_hierarchical]
ax1.bar(labels, accuracies, color=['red', 'black'])
ax1.set_ylim(0, 1)
ax1.set_ylabel('Accuracy')
ax1.set_title('Comparison of Softmax and Hierarchical Softmax Accuracies')

# Execution time plot
execution_times = [softmax_execution_time, hierarchical_execution_time]
ax2.bar(labels, execution_times, color=['red', 'black'])
ax2.set_ylabel('Execution Time (seconds)')
ax2.set_title('Comparison of Softmax and Hierarchical Softmax Execution Times')

plt.tight_layout()
plt.show()
