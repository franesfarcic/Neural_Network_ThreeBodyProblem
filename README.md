# Neural_Network_ThreeBodyProblem
"Python codes for stochastically generated simulations of three-body problem trajectories and the implementation of a BiLSTM neural network to predict these trajectories."

Multiprocessing: The multiprocessing module is being used effectively to parallelize the simulation. Each worker runs a simulation independently with different initial conditions. This is particularly useful for large-scale simulations, as it utilizes multiple cores of the CPU.

State Vector and Variables: The state vector is defined as a collection of positions, velocities, and distances between three bodies. These are used in the Runge-Kutta method for updating the system over time.

Random Initial Conditions: The generiranje_state_vektora function generates random initial positions and velocities for the bodies. This randomness can help in exploring different initial conditions for the simulation.

Adaptive Step Size: The adaptive Runge-Kutta method used here adjusts the step size (h) based on the error, ensuring that the simulation maintains accuracy without unnecessarily small time steps.

Performance Considerations:

Since the number of simulations can be large (ukpuni_broj_globalnih_simulacija), be mindful of memory and computational limitations, especially when storing results in large lists and numpy arrays.
The use of tqdm for progress tracking helps monitor the progress of each simulation, which is important for such large computations.
Saving Results: After all simulations are complete, results are saved in CSV files within the "Simulacije" directory. It would be useful to handle file naming more robustly, especially when multiple simulations are saved.

![Image](https://github.com/user-attachments/assets/52ff993b-e6a3-4dd5-a164-84bb6a3f4df5)

![Image](https://github.com/user-attachments/assets/2453117f-3cd5-43a1-b986-8557f0f65676)

![Image](https://github.com/user-attachments/assets/9d9d38c4-2ec1-4d12-b197-3b223d6f86c4)

![Image](https://github.com/user-attachments/assets/e3b5f449-c8d0-460e-8638-252a2c98dbe2)


1. LSTM_NN class:
This defines a simple LSTM-based neural network.
The network has:
A bidirectional option for the LSTM layer.
A LayerNorm layer for normalization.
A Dropout layer to prevent overfitting.
A fully connected output layer (Linear) to map from the LSTM output to the target feature size.
2. Srednja_Vrijednosti_I_Standardna_Devijacija:
This function calculates the mean and standard deviation for each feature from the training data to normalize the dataset. It assumes the dataset is a pickle file that contains train, validation, and test data in a structured format.
3. MojDataloader class:
A custom PyTorch Dataset that loads the input and label data from pickle files. It also normalizes the data using the calculated means and standard deviations.
4. RMSELoss class:
A custom loss function that computes the Root Mean Squared Error (RMSE) between the predicted and true values.
5. TrainNetwork function:
This is the core function for training the model. It:
Loads the training and validation data.
Defines the LSTM model and optimizer.
Optionally uses a learning rate scheduler (exponential or cosine annealing).
Trains the model through multiple epochs, calculating the loss at each step and updating the model weights via backpropagation.
Implements early stopping if the validation loss does not improve after a certain number of epochs.
Key features:
Data Normalization: Essential for models like LSTM, as they are sensitive to the scale of the data.
Custom DataLoader: Ensures that the data is correctly normalized during training, validation, and testing.
Custom Scheduler and Loss Function: Helps in optimizing the model by adjusting the learning rate and tracking RMSE loss.
Early Stopping: Monitors the validation loss to stop training early if the model’s performance does not improve after several epochs.

![Image](https://github.com/user-attachments/assets/361a6ece-c48a-472f-89da-d3d26cccfc93)

![Image](https://github.com/user-attachments/assets/e2cde9a1-dd07-4a6d-bb8a-1827d9cad10b)

![Image](https://github.com/user-attachments/assets/5931b844-ec4c-48c6-a218-1bb022a2e21e)

![Image](https://github.com/user-attachments/assets/44349753-0896-4fb9-9851-3522f3de94e2)
