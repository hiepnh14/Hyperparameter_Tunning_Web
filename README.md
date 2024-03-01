# Management System for ML/DL Hyperparameter Tuning

This project aims to build a mini experiment management system that helps researchers select the best model for a simple ML/DL task. The task involves solving the MNIST challenge using a PyTorch model.

In this project, I focused on developing a backend that is compact and efficient, allowing multi-Threading, concurrency. Frontend can be further developed to improve User Experience.    

## Features

This management system includes the following features:

### 1. User Interface (UI)

This simple web interface is designed and built to allow users to tune the DL task as a blackbox. The UI provides the following functionalities:

- Users can tune hyperparameters for the Deep Learning task.
- Users can run several jobs with different hyperparameters by simply clicking a button.
- The progress of currently running jobs is displayed, a plot of acccuracy/epoch is plotted in real-time.
- The results of all finished jobs are displayed in a table in descending order of accuracy (click refresh to update).
- The UI can be resumed, allowing users to close and reopen the web browser without losing the current state of experiments.
- Users can add new jobs, such as grid-searching on different hyperparameters by choosing sets of parameters.

### 2. Job Management

The experiment management system includes job management capabilities, such as:

- Checking if the exact same job has been run before to avoid duplication.
- Storing and organizing job configurations and results for easy retrieval and analysis.

## Getting Started

Here's a more detailed guide on getting started with the experiment management system:

1. Clone the repository: 
   - Open your terminal or command prompt.
   - Navigate to the directory where you want to clone the repository.
   - Run the following command: `git clone https://github.com/hiepnh14/Hyperparameter_Tunning_Web`

2. Install the required dependencies:
   - Make sure you have Python and pip installed on your system.
   - Create virtual environment is highly recommended.
   - Navigate to the cloned repository directory.
   - Run the following command to install the dependencies: `pip install -r requirements.txt`

3. Run the web interface locally:
   - In the terminal, navigate to the repository directory.
   - If the database 'models.db' has not been initialized, run `python createDatabase.py`
   - Run the following command: `python app.py`
   - This will start the local development server.

4. Access the web interface:
   - Open your web browser and go to `http://localhost:8000` (or the specified port).
   - You should see the experiment management system's user interface.

5. Tune hyperparameters and run jobs:
   - Use the provided UI to tune the hyperparameters for the DL task.
   - Click "Add" to add new values for parameters, hold Ctrl or Shift to select multiple values.
   - Click "Add task to queue" to add the job to the queue to be trained.
   - Click "Refresh Queue Table", "Refresh Table" to see the queue and trained results.
   - Click the "Start training" button to start running the job with the selected hyperparameters.
   - Click "Refresh Graph" to refresh the graph or it is automatically refreshed every 20 seconds.
   - Monitor the progress of running jobs and view the results of finished jobs.

## Future Improvement

There are a lot of room for improvements such as more intuitive UI/UX. Further Data Visualization/Data analysis can be displayed. More parameters can be added to be tuned. 

## Contributing

Contributions to this project are welcome. If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Some demo images
<!-- ![Image 3](images/Screenshot%202024-02-28%20040207.png)
![Image 2](images/Screenshot%202024-02-28%20033538.png)

A reference for graph plot
![Image 1](images/Screenshot%202024-02-27%20232720.png) -->
