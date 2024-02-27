from flask import Flask, render_template, request, jsonify
import threading
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from model import SimpleCNN
import os
import sqlite3
import time
app = Flask(__name__)

# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Global variables for plotting:
accuracies = []


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = 100
# Training function
CHECKPOINT_FOLDER = "./saved_model"
def train_model(epochs, learning_rate, batch_size, model_id):
    global accuracies
    accuracies = []
    training_status['status'] = 'Training has started!'
    # Update the trainloader with the new batch size
    # Initialize the model
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    model = SimpleCNN()
    model.to(device)
    start_time = time.time()
    # Update the optimizer with the new learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_val_acc = 0
    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        if accuracy > best_val_acc:
            best_val_acc = accuracy
            if not os.path.exists(CHECKPOINT_FOLDER):
               os.makedirs(CHECKPOINT_FOLDER)
            # print("Saving ...")
            state = {'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'lr': learning_rate}
            torch.save(state, os.path.join(CHECKPOINT_FOLDER, f"best_model_{model_id}.pth"))
        training_status['status'] = f'Training on {device}, Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}%'
        print(f"Epoch {epoch + 1} - Training loss: {running_loss / len(trainloader)}, Accuracy: {accuracy:.2f}%")
    training_time = time.time() - start_time
    # Save the hyperparameters and best accuracy to the database
    conn = sqlite3.connect('models.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO model_accuracies (epochs, learning_rate, batch_size, training_time, accuracy)
        VALUES (?, ?, ?, ?, ?)
    ''', (epochs, learning_rate, batch_size, training_time, max(accuracies)))
    conn.commit()
    conn.close()
# In-memory storage for training status
training_status = {'status': 'Not started'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    return jsonify(accuracies)
# create a route to handle the /get_model_data request and return the data from the database:
@app.route('/get_model_data')
def get_model_data():
    conn = sqlite3.connect('models.db')
    cursor = conn.cursor()
    cursor.execute('SELECT epochs, learning_rate, batch_size, training_time, accuracy FROM model_accuracies ORDER BY accuracy DESC')
    data = cursor.fetchall()
    conn.close()
    # Convert the data to a list of dictionaries for easier JSON serialization
    models = [{'epochs': row[0], 'learning_rate': row[1], 'batch_size': row[2], 'training_time': round(row[3], 1), 'accuracy': round(row[4], 2)} for row in data]
    return jsonify(models)

@app.route('/start_training', methods=['POST'])
def start_training():
    epochs = int(request.form['epochs'])
    learning_rate = float(request.form['learning_rate'])
    batch_size = int(request.form['batch_size'])
    model_id = f"lr_{learning_rate}_bs_{batch_size}_ep_{epochs}"
    thread = threading.Thread(target=train_model, args=(epochs,learning_rate, batch_size, model_id))
    thread.start()
    # training_status['status'] = 'Training in progress'
    return jsonify({'message': 'Training started'})

@app.route('/status', methods=['GET'])
def status():
    return jsonify(training_status)

if __name__ == '__main__':
    app.run(debug=True)
