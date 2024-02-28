from flask import Flask, render_template, request, jsonify
import threading
import torch
import sqlite3
import train
from train import hyperparameter_tuning, training_status

app = Flask(__name__)

# In-memory storage for training status
datafile = 'models.db'


# Global variables for plotting:
val_accuracies = train.val_accuracies

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss = 100
checkpoint_folder = "./saved_model"



# Add task to queue function
def add_task_to_queue(epochs, learning_rates, dropout_rates):
    messages = []
    conn = sqlite3.connect(datafile)
    cursor = conn.cursor()
    for learning_rate in learning_rates:
        for dropout_rate in dropout_rates:
            cursor.execute('SELECT * FROM model_trainings WHERE epochs=? AND learning_rate=? AND dropout_rate=?', (epochs, learning_rate, dropout_rate))
            duplicate1 = cursor.fetchone()
            cursor.execute('SELECT * FROM model_accuracies WHERE epochs=? AND learning_rate=? AND dropout_rate=?', (epochs, learning_rate, dropout_rate))
            duplicate2 = cursor.fetchone()
            if duplicate1 or duplicate2:
                messages.append({'message': f'Duplicate entry for learning_rate {learning_rate} and dropout_rate {dropout_rate}'})
            else:
                cursor.execute('''
                    INSERT INTO model_trainings (epochs, learning_rate, dropout_rate)
                    VALUES (?, ?, ?)
                ''', (epochs, learning_rate, dropout_rate))
                conn.commit()
                messages.append({'message': f'Task added to queue for learning_rate {learning_rate} and dropout_rate {dropout_rate}'})
    conn.close()
    return jsonify(messages)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    # print(train.val_accuracies)
    return jsonify(train.val_accuracies)

# create a route to handle the /get_model_data request and return the data from the database:
@app.route('/get_model_data')
def get_model_data():
    conn = sqlite3.connect(datafile)
    cursor = conn.cursor()
    cursor.execute('SELECT epochs, learning_rate, dropout_rate, training_time, accuracy FROM model_accuracies ORDER BY accuracy DESC')
    data = cursor.fetchall()
    conn.close()
    # Convert the data to a list of dictionaries for easier JSON serialization
    models = [{'epochs': row[0], 'learning_rate': row[1], 'dropout_rate': row[2], 'training_time': round(row[3], 1), 'accuracy': round(row[4], 2)} for row in data]
    return jsonify(models)

# create a route to handle the /get_model_data request and return the data from the database:
@app.route('/get_queue_data')
def get_queue_data():
    conn = sqlite3.connect(datafile)
    cursor = conn.cursor()
    cursor.execute('SELECT epochs, learning_rate, dropout_rate FROM model_trainings')
    data = cursor.fetchall()
    conn.close()
    # Convert the data to a list of dictionaries for easier JSON serialization
    models = [{'epochs': row[0], 'learning_rate': row[1], 'dropout_rate': row[2]} for row in data]
    return jsonify(models)

@app.route('/add_to_queue', methods=['POST'])
def add_to_queue():
    epochs = int(request.form['epochs'])
    # print(f"Epochs: {epochs}")

    learning_rates = list(map(float, request.form.get('learning_rate').split(',')))
    # print(f"Learning Rates: {learning_rates}")

    dropout_rates = list(map(float, request.form.get('dropout_rate').split(',')))
    # print(f"Dropout Rates: {dropout_rates}")
    thread = threading.Thread(target=add_task_to_queue, args=(epochs,learning_rates, dropout_rates))
    thread.start()
    thread.join()
    # add_task_to_queue.delay(epochs, learning_rates, dropout_rates)
    return jsonify({'message': 'Task added to queue'})

@app.route('/start_training', methods=['POST'])
def start_training():
    thread = threading.Thread(target=hyperparameter_tuning, args=(datafile, training_status, device, checkpoint_folder))
    thread.start()
    return jsonify({'message': 'Tuning Training started'})

@app.route('/status', methods=['GET'])
def status():
    return jsonify(training_status)

if __name__ == '__main__':
    app.run(debug=True)
