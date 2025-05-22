import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

# Constants
NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
EPOCHS = 20
BATCH_SIZE = 32

def get_data():
    """Prepare CIFAR10 data."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    X_train /= 255
    X_test /= 255
    return (X_train, y_train), (X_test, y_test)

def create_model():
    """Create the same model architecture as in mia_attack.py"""
    model = tf.keras.models.Sequential([
        # Conv1
        layers.Conv2D(6, (5, 5), activation='relu', padding='valid', input_shape=(WIDTH, HEIGHT, CHANNELS)),
        layers.MaxPooling2D((2, 2)),
        
        # Conv2
        layers.Conv2D(16, (5, 5), activation='relu', padding='valid'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten
        layers.Flatten(),
        
        # FC layers
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_save_model():
    # Create directory for saving model
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Get data
    (X_train, y_train), (X_test, y_test) = get_data()
    
    # Create and train model
    model = create_model()
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Save model weights
    model_path = os.path.join(save_dir, "trained_model.npz")
    
    # Get weights and save them in the same format as your PyTorch models
    weights = {}
    
    # Map layer names to match the expected format
    layer_name_map = {
        'conv2d': 'conv1',
        'conv2d_1': 'conv2',
        'dense': 'fc1',
        'dense_1': 'fc2',
        'dense_2': 'fc3'
    }
    
    for layer in model.layers:
        if layer.weights:
            # Convert weights to numpy arrays
            w = layer.weights[0].numpy()
            b = layer.weights[1].numpy()
            
            # For convolutional layers, transpose from NHWC to NCHW format
            if isinstance(layer, layers.Conv2D):
                w = np.transpose(w, (3, 2, 0, 1))  # NHWC -> NCHW
            
            # For dense layers, transpose the weight matrix
            elif isinstance(layer, layers.Dense):
                w = w.T
            
            # Map the layer name to match the expected format
            mapped_name = layer_name_map.get(layer.name, layer.name)
            
            # Save with the same naming convention as your PyTorch models
            weights[f'model.{mapped_name}.weight'] = w
            weights[f'model.{mapped_name}.bias'] = b
    
    # Save weights
    np.savez(model_path, **weights)
    print(f"Model saved to {model_path}")
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return model_path

if __name__ == "__main__":
    model_path = train_and_save_model()
    print(f"\nModel saved at: {model_path}")