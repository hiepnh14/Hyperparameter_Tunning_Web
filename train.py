import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from model import SimpleCNN, CNN
import os
import sqlite3
import time
from loadData import load_data

# global variables
val_accuracies = []
# In-memory storage for training status
training_status = {'status': 'Not started'}

def train_model(epochs, learning_rate, dropout_rate, model_id, training_status, datafile, device, CHECKPOINT_FOLDER):
    global val_accuracies
    val_accuracies = []
    train_accuracies = []
    training_status['status'] = 'Training has started!'
    # Update the trainloader with the new batch size
    # Initialize the model
    trainloader, val_loader = load_data(batch_size = 16)
    model = CNN(dropout_rate=dropout_rate)
    model.to(device)
    start_time = time.time()
    # Update the optimizer with the new learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_val_acc = 0
    for epoch in range(epochs):
        training_status['status'] = f'Training with learning rate = {learning_rate}, dropout_rate = {dropout_rate} at Epoch {epoch + 1}/{epochs}'
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
        train_accuracies.append(accuracy)
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                output = model(images)
                _, predicted = torch.max(output.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            if not os.path.exists(CHECKPOINT_FOLDER):
               os.makedirs(CHECKPOINT_FOLDER)
            # print("Saving ...")
            state = {'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'lr': learning_rate}
            torch.save(state, os.path.join(CHECKPOINT_FOLDER, f"best_model_{model_id}.pth"))
        training_status['status'] = f'Training with learning rate = {learning_rate}, dropout_rate = {dropout_rate} at Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}%'
        print(f"{model_id}: Epoch {epoch + 1} - Training loss: {running_loss / len(trainloader)}, Accuracy: {accuracy:.2f}%")
    training_time = time.time() - start_time
    # Save the hyperparameters and best accuracy to the database
    conn = sqlite3.connect(datafile)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO model_accuracies (epochs, learning_rate, dropout_rate, training_time, accuracy)
        VALUES (?, ?, ?, ?, ?)
    ''', (epochs, learning_rate, dropout_rate, training_time, max(val_accuracies)))
    conn.commit()
    conn.close()

# Hyperparameter tuning function to train models in queue
def hyperparameter_tuning(datafile, training_status, device, CHECKPOINT_FOLDER):
    conn = sqlite3.connect(datafile)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM model_trainings')
    tasks = cursor.fetchall()
    for task in tasks:
        _, epochs, learning_rate, dropout_rate = task
        model_id = f"lr_{learning_rate}_drop_{dropout_rate}_ep_{epochs}"
        # thread = threading.Thread(target=train_model, args=(epochs, learning_rate, dropout_rate, model_id))
        # thread.start()
        train_model(epochs, learning_rate, dropout_rate, model_id,  training_status, datafile, device, CHECKPOINT_FOLDER)
        # training_status['status'] = 'Training in progress'
        # Remove the task from the queue after training
        cursor.execute('DELETE FROM model_trainings WHERE epochs=? AND learning_rate=? AND dropout_rate=?', (epochs, learning_rate, dropout_rate))
        conn.commit()
        conn.close()
    # Check for tasks added during training
    conn = sqlite3.connect(datafile)
    cursor = conn.cursor()
    if  cursor.execute('SELECT * FROM model_trainings').fetchone():
        conn.close()
        hyperparameter_tuning()
    training_status['status'] = 'Training completed'
    conn.close()