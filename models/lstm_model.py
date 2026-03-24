"""
LSTM Model Implementation for Time Series Forecasting

This module provides a PyTorch-based LSTM model that inherits from BaseTimeSeriesModel,
enabling fair comparison with PyReCo models using the same interface.

Author: Beginner-friendly version with English comments
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any

from .base_model import BaseTimeSeriesModel


class LSTMNetwork(nn.Module):
    """
    PyTorch LSTM Network

    Architecture:
    Input → LSTM layers (with dropout) → Fully Connected → Output
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int, dropout: float = 0.2):
        """
        Initialize LSTM network

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout rate (0 to 1)
        """
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Dropout only for multi-layer
            batch_first=True  # Input shape: (batch, seq, feature)
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor, shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor, shape (batch_size, output_size)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last timestep's output for prediction
        # Shape: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layer
        # Shape: (batch_size, output_size)
        output = self.fc(last_output)

        return output


class LSTMModel(BaseTimeSeriesModel):
    """
    LSTM Model Wrapper - Inherits from BaseTimeSeriesModel

    This wrapper enables LSTM to use the same interface as PyReCo models,
    ensuring fair comparison with unified evaluation metrics.

    Training uses:
    - Adam optimizer
    - MSE loss
    - Mini-batch training
    - Early stopping (optional)

    Example:
        model = LSTMModel(hidden_size=64, num_layers=2, epochs=100)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results = model.evaluate(X_test, y_test)
    """

    def __init__(self,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32,
                 patience: int = 10,
                 device: str = None,
                 verbose: bool = True):
        """
        Initialize LSTM model

        Args:
            hidden_size: Number of LSTM hidden units (default: 64)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate between LSTM layers (default: 0.2)
            learning_rate: Learning rate for optimizer (default: 0.001)
            epochs: Maximum number of training epochs (default: 100)
            batch_size: Batch size for training (default: 32)
            patience: Early stopping patience (epochs without improvement, default: 10)
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            verbose: Whether to print training progress (default: True)
        """
        # Call parent class initialization
        super().__init__(
            name="LSTM",
            config={
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'batch_size': batch_size,
                'patience': patience,
            }
        )

        # Store hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose

        # Set device (CPU or GPU)
        # Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Model will be initialized during fit()
        self.network = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        # Training history
        self.train_losses = []
        self.best_loss = float('inf')
        self.best_epoch = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        """
        Train the LSTM model

        Args:
            X_train: Training input, shape (n_samples, n_timesteps, n_features)
            y_train: Training target, shape (n_samples, n_output_steps, n_features)
            X_val: Optional validation input for early stopping
            y_val: Optional validation target for early stopping

        Note:
            - If n_output_steps > 1, only the first timestep is used for training
            - Uses early stopping based on validation loss (if provided) or training loss
            - Saves and restores the best model weights (checkpoint)
        """
        if self.verbose:
            print(f"    Starting training {self.name}...")

        start_time = time.time()

        # Get data dimensions
        n_samples, n_timesteps, n_features = X_train.shape
        n_output_steps = y_train.shape[1]

        # For now, only predict the first output timestep
        # Shape: (n_samples, n_features)
        y_train_flat = y_train[:, 0, :]

        # Prepare validation data if provided
        use_val = X_val is not None and y_val is not None
        if use_val:
            y_val_flat = y_val[:, 0, :]
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_flat).to(self.device)

        # Initialize network if not already done
        if self.network is None:
            self.network = LSTMNetwork(
                input_size=n_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=n_features,
                dropout=self.dropout
            ).to(self.device)

            # Store input/output sizes in config for save/load
            self.config['input_size'] = n_features
            self.config['output_size'] = n_features

            # Initialize optimizer
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.learning_rate
            )

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train_flat).to(self.device)

        # Create DataLoader for mini-batch training
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True  # Shuffle within each epoch
        )

        # Training loop
        patience_counter = 0
        best_state_dict = copy.deepcopy(self.network.state_dict())

        for epoch in range(self.epochs):
            self.network.train()
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self.network(batch_X)
                loss = self.criterion(outputs, batch_y)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

            # Calculate average training loss for this epoch
            epoch_loss = epoch_loss / len(dataset)
            self.train_losses.append(epoch_loss)

            # Compute the loss used for early stopping
            if use_val:
                # Validation loss for early stopping
                self.network.eval()
                with torch.no_grad():
                    val_pred = self.network(X_val_tensor)
                    monitor_loss = self.criterion(val_pred, y_val_tensor).item()
            else:
                # Fallback: use training loss
                monitor_loss = epoch_loss

            # Early stopping check + best model checkpoint
            if monitor_loss < self.best_loss:
                self.best_loss = monitor_loss
                self.best_epoch = epoch + 1
                patience_counter = 0
                best_state_dict = copy.deepcopy(self.network.state_dict())
            else:
                patience_counter += 1

            # Print progress every 10 epochs
            if self.verbose and (epoch + 1) % 10 == 0:
                if use_val:
                    print(f"    Epoch [{epoch+1}/{self.epochs}], Train Loss: {epoch_loss:.6f}, Val Loss: {monitor_loss:.6f}")
                else:
                    print(f"    Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.6f}")

            # Early stopping
            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break

        # Restore best model weights
        self.network.load_state_dict(best_state_dict)

        # Update training state
        self.training_time = time.time() - start_time
        self.is_trained = True

        if self.verbose:
            print(f"    Training complete! Took {self.training_time:.2f} seconds")
            print(f"    Best {'val' if use_val else 'train'} loss: {self.best_loss:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input data, shape (n_samples, n_timesteps, n_features)

        Returns:
            Predictions, shape (n_samples, 1, n_features)

        Note:
            Currently only predicts 1 step ahead to match PyReCo's typical usage
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.name} model is not trained yet!")

        start_time = time.time()

        # Set model to evaluation mode
        self.network.eval()

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Make predictions (no gradient computation needed)
        with torch.no_grad():
            predictions = self.network(X_tensor)

        # Convert back to numpy and reshape to (n_samples, 1, n_features)
        predictions = predictions.cpu().numpy()
        predictions = predictions[:, np.newaxis, :]  # Add timestep dimension

        self.prediction_time = time.time() - start_time

        return predictions

    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history and statistics

        Returns:
            Dictionary containing training information
        """
        return {
            'train_losses': self.train_losses,
            'best_loss': self.best_loss,
            'num_epochs_trained': len(self.train_losses),
            'device': str(self.device),
        }

    def save_model(self, filepath: str) -> None:
        """
        Save model weights to file

        Args:
            filepath: Path to save the model (e.g., 'model.pth')
        """
        if self.network is None:
            raise RuntimeError("No model to save! Train the model first.")

        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
        }, filepath)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load model weights from file

        Args:
            filepath: Path to the saved model file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Reconstruct network from config
        config = checkpoint['config']
        self.network = LSTMNetwork(
            input_size=config.get('input_size', 1),
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=config.get('output_size', 1),
            dropout=config['dropout']
        ).to(self.device)

        # Load weights
        self.network.load_state_dict(checkpoint['model_state_dict'])

        # Recreate optimizer and load its state (for resume training)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), lr=config.get('learning_rate', 0.001)
            )
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load training history
        self.train_losses = checkpoint.get('train_losses', [])

        self.is_trained = True
        print(f"Model loaded from {filepath}")


# ============================================================================
# Helper function for hyperparameter search
# ============================================================================

def tune_lstm_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              param_grid: Dict[str, list],
                              lstm_device: str = 'auto',
                              verbose: bool = True,
                              layer_hidden_map: Dict[int, int] = None) -> Dict[str, Any]:
    """
    Simple grid search for LSTM hyperparameters

    Args:
        X_train: Training input
        y_train: Training target
        X_val: Validation input
        y_val: Validation target
        param_grid: Dictionary of parameters to search, e.g.:
            {
                'num_layers': [1, 2],
                'learning_rate': [0.001, 0.01],
            }
        verbose: Whether to print progress
        layer_hidden_map: Optional mapping from num_layers to hidden_size.
            When provided, hidden_size is computed from num_layers for each
            combination (no need to include hidden_size in param_grid).
            Example: {1: 48, 2: 28} means 1-layer gets hidden_size=48,
            2-layer gets hidden_size=28 (both matching the same parameter budget).

    Returns:
        Dictionary containing:
            - 'best_params': Best parameter combination
            - 'best_score': Best validation MSE
            - 'all_results': List of all results

    Example:
        param_grid = {
            'num_layers': [1, 2],
            'learning_rate': [0.001, 0.01],
        }
        layer_hidden_map = {1: 48, 2: 28}
        results = tune_lstm_hyperparameters(X_train, y_train, X_val, y_val,
                                           param_grid, layer_hidden_map=layer_hidden_map)
        best_model = LSTMModel(**results['best_params'])
    """
    from itertools import product

    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    best_score = float('inf')
    best_params = None
    all_results = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting hyperparameter search...")
        print(f"Total combinations: {len(combinations)}")
        print(f"{'='*60}\n")

    # Convert 'auto' to None for auto-detection
    device_arg = None if lstm_device == 'auto' else lstm_device

    for i, combo in enumerate(combinations):
        # Create parameter dictionary
        params = dict(zip(keys, combo))

        # Compute hidden_size from num_layers if layer_hidden_map provided
        if layer_hidden_map and 'num_layers' in params:
            params['hidden_size'] = layer_hidden_map[params['num_layers']]

        if verbose:
            print(f"[{i+1}/{len(combinations)}] Testing: {params}")

        # Create and train model (with validation data for early stopping + checkpoint)
        model = LSTMModel(**params, device=device_arg, verbose=False)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # Evaluate on validation set
        val_results = model.evaluate(X_val, y_val, metrics=['mse'])
        val_mse = val_results['mse']

        # Store results
        all_results.append({
            'params': params,
            'val_mse': val_mse,
            'train_time': model.training_time,
            'best_epoch': getattr(model, 'best_epoch', None),
        })

        if verbose:
            print(f"  → Validation MSE: {val_mse:.6f}, Train time: {model.training_time:.2f}s\n")

        # Update best params
        if val_mse < best_score:
            best_score = val_mse
            best_params = params

        # Memory optimization: explicitly delete model and free GPU memory
        del model
        del val_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()  # Free MPS (Apple GPU) cache

    # Final garbage collection after all hyperparameter search
    import gc
    gc.collect()

    if verbose:
        print(f"{'='*60}")
        print(f"Hyperparameter search complete!")
        print(f"Best params: {best_params}")
        print(f"Best validation MSE: {best_score:.6f}")
        print(f"{'='*60}\n")

    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': all_results,
    }
