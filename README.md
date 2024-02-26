# Experiment Management System for ML/DL Task

This project aims to build a mini experiment management system that helps researchers select the best model for a simple ML/DL task. The task involves solving the MNIST challenge using a PyTorch model.

## Features

The management system includes the following features:

### 1. User Interface (UI)

A simple web interface is designed and built to allow users to tune the DL task as a blackbox. The UI provides the following functionalities:

- Users can tune hyperparameters for the DL task.
- Users can run several jobs with different hyperparameters by simply clicking a button.
- The progress of currently running jobs is displayed.
- The results of all finished jobs are displayed.
- Experiments can be sorted by pre-defined metrics (e.g., accuracy, run time) for ease of comparison.
- The UI can be resumed, allowing users to close and reopen the web browser without losing the current state of experiments.
- Users can add new jobs, such as grid-searching on different hyperparameters.

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
   - Navigate to the cloned repository directory.
   - Run the following command to install the dependencies: `pip install -r requirements.txt`

3. Run the web interface locally:
   - In the terminal, navigate to the repository directory.
   - Run the following command: `python app.py`
   - This will start the local development server.

4. Access the web interface:
   - Open your web browser and go to `http://localhost:8000` (or the specified port).
   - You should see the experiment management system's user interface.

5. Tune hyperparameters and run jobs:
   - Use the provided UI to tune the hyperparameters for the DL task.
   - Click the "Run Job" button to start running the job with the selected hyperparameters.
   - Monitor the progress of running jobs and view the results of finished jobs.

That's it! You can now use the experiment management system to find the best model for the MNIST challenge.

## Contributing

Contributions to this project are welcome. If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

