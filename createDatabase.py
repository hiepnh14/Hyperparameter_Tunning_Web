import sqlite3

datafile = 'models.db'
# Connect to the SQLite database
conn = sqlite3.connect(datafile)

# Create a cursor object
cursor = conn.cursor()

# Create a table to store hyperparameters and accuracies
cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_accuracies (
        id INTEGER PRIMARY KEY,
        epochs INTEGER,
        learning_rate REAL,
        dropout_rate REAL,
        training_time REAL,
        accuracy REAL
    )
''')
# Create a table to store hyperparameters and accuracies
cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_trainings (
        id INTEGER PRIMARY KEY,
        epochs INTEGER,
        learning_rate REAL,
        dropout_rate REAL
    )
''')
# Commit the changes and close the connection
conn.commit()
conn.close()


# conn = sqlite3.connect(datafile)
# cursor = conn.cursor()
# cursor.execute('''
#     INSERT INTO model_accuracies (epochs, learning_rate, batch_size, training_time, accuracy)
#     VALUES (?, ?, ?, ?, ?)
# ''', (10, 0.1, 64, 1000, 98))
# cursor.execute('''
#     INSERT INTO model_accuracies (epochs, learning_rate, batch_size, training_time, accuracy)
#     VALUES (?, ?, ?, ?, ?)
# ''', (10, 0.01, 64, 10000, 90))
# cursor.execute('''
#     INSERT INTO model_accuracies (epochs, learning_rate, batch_size, training_time, accuracy)
#     VALUES (?, ?, ?, ?, ?)
# ''', (10, 0.001, 64, 1000, 78))
# conn.commit()
# conn.close()