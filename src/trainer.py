from qiskit.primitives import Estimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.metrics import accuracy_score
import numpy as np
from scipy.optimize import minimize
import time

class QMLTrainer:
    # Quantum Machine Learning trainer for anomaly detection.
    
    def __init__(self, n_qubits, X_train, y_train):
        self.feature_map = create_enhanced_circuit(n_qubits)
        self.estimator = Estimator()
        self.qnn = EstimatorQNN(
            circuit=self.feature_map,
            input_params=self.feature_map.parameters[:n_qubits],
            weight_params=self.feature_map.parameters[n_qubits:],
            estimator=self.estimator
        )
        self.X_train = X_train
        self.y_train = y_train
        self.best_loss = float('inf')
        self.best_weights = None

    def loss_fn(self, weights):
        # Compute the loss function for the quantum model.
        try:
            y_pred = self.qnn.forward(self.X_train, weights)
            loss = -accuracy_score(self.y_train, np.sign(y_pred))

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_weights = weights.copy()

            return loss
        except Exception as e:
            print(f"Error in loss calculation: {e}")
            return float('inf')

    def train(self, initial_point, maxiter=300):
        # Train the quantum model using COBYLA optimizer.
        start_time = time.time()
        
        opt_result = minimize(
            self.loss_fn,
            initial_point,
            method='COBYLA',
            options={'maxiter': maxiter, 'tol': 1e-3, 'rhobeg': 1.0}
        )

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        return opt_result, self.best_weights
