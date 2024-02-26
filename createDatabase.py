import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('models.db')

# Create a cursor object
cursor = conn.cursor()

# Create a table to store hyperparameters and accuracies
cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_accuracies (
        id INTEGER PRIMARY KEY,
        learning_rate REAL,
        batch_size INTEGER,
        traning_time REAL,
        accuracy REAL
    )
''')

# Commit the changes and close the connection
conn.commit()
conn.close()
