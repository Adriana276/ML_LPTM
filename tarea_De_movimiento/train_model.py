"""
Train MLP model for HAR using sliding window approach with scikit-learn
Alternative to TensorFlow for Python 3.14 compatibility
"""

import pandas as pd
import glob
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import sklearn.model_selection
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from config import *

def load_dataset(dataset_path='./dataset/*.csv'):
    """Load all CSV files from dataset folder"""
    print(f"Loading datasets from {dataset_path}...")
    df = pd.concat([pd.read_csv(f) for f in glob.glob(dataset_path)], ignore_index=True)
    print(f"Loaded {len(df)} total samples")
    return df

def prepare_sliding_window_data(df):
    """
    Prepare data using sliding window approach
    Creates sequences of STEP_SIZE timesteps
    """
    print(f"\nPreparing sliding window data (window size: {STEP_SIZE})...")
    
    # Convert labels to numeric
    df['label'] = [LABEL_DICT[item] for item in df['label']]
    
    # Extract features and labels
    x = np.array(df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]])
    y = np.array(df["label"])
    
    # Create sliding windows
    modDataset = []
    modTruth = []
    
    for i in range(len(x) - STEP_SIZE):
        temp = []
        for j in range(i, i + STEP_SIZE):
            temp.append(x[j])
        modDataset.append(temp)
    
    # For labels, use the most common label in the window
    for i in range(len(y) - STEP_SIZE):
        temp = []
        for j in range(i, i + STEP_SIZE):
            temp.append(y[j])
        
        most_common_item = max(temp, key=temp.count)
        modTruth.append(most_common_item)
    
    # Reshape to (samples, features) - flatten timesteps
    modDataset = np.array(modDataset).reshape(-1, STEP_SIZE * SENSOR_NUM)
    modTruth = np.array(modTruth)
    
    print(f"Created {len(modDataset)} windowed samples")
    print(f"Data shape: {modDataset.shape}")
    
    return modDataset, modTruth

def create_model():
    """Create MLP model using scikit-learn"""
    model = MLPClassifier(
        hidden_layer_sizes=(128, 128),
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=100,
        shuffle=True,
        random_state=42,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    return model

def plot_confusion_matrix(y_true, y_pred, save_path='./results/confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES.values(),
                yticklabels=CLASS_NAMES.values())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_loss_curve(model, save_path='./results/training_history.png'):
    """Plot training loss curve"""
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(model.loss_curve_, label='Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        
        if hasattr(model, 'validation_scores_'):
            plt.subplot(1, 2, 2)
            plt.plot(model.validation_scores_, label='Validation Score', color='orange')
            plt.xlabel('Iteration')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    import os
    os.makedirs('./results', exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    print("=" * 60)
    print("HAR Model Training (scikit-learn MLP)")
    print("=" * 60)
    
    # Load data
    df = load_dataset()
    
    # Prepare sliding windows
    x, y = prepare_sliding_window_data(df)
    
    # Split data
    print("\nSplitting data...")
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")
    
    # Create model
    print("\nCreating MLP model...")
    print("Architecture: 120 inputs -> Dense(128) -> Dense(128) -> 4 outputs")
    model = create_model()
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    model.fit(x_train, y_train)
    
    # Save model
    model_file = os.path.join(MODEL_PATH, 'model.pkl')
    print(f"\nSaving model to {model_file}...")
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully!")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating model on test set...")
    print("=" * 60)
    
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=CLASS_NAMES.values()))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_test)
    
    # Plot training curves
    plot_loss_curve(model)
    
    # Show sample predictions
    print("\n" + "=" * 60)
    print("Sample Predictions (first 20):")
    print("=" * 60)
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(x_test)
    
    for i in range(min(20, len(y_test))):
        actual = CLASS_NAMES[y_test[i]]
        predicted = CLASS_NAMES[y_pred_test[i]]
        confidence = y_pred_proba[i][y_pred_test[i]] * 100
        status = "[OK]" if actual == predicted else "[X]"
        print(f"{status} Predicted: {predicted:4s} (conf: {confidence:5.1f}%)  |  Actual: {actual:4s}")
    
    # Calculate accuracy per class
    print("\n" + "=" * 60)
    print("Accuracy per Activity:")
    print("=" * 60)
    for label_id, label_name in CLASS_NAMES.items():
        mask = y_test == label_id
        if mask.sum() > 0:
            class_acc = (y_pred_test[mask] == y_test[mask]).sum() / mask.sum()
            print(f"{label_name}: {class_acc*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {model_file}")
    print(f"Results saved to: ./results/")
