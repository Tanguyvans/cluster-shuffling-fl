# MIA works 

import numpy as np
from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data

import ssl
import matplotlib.pyplot as plt
ssl._create_default_https_context = ssl._create_unverified_context

from scikeras.wrappers import KerasClassifier

import torch

NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SHADOW_DATASET_SIZE = 4000
ATTACK_TEST_DATASET_SIZE = 4000


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 12, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 12, "Number of epochs to train attack models.")
flags.DEFINE_integer("num_shadows", 3, "Number of epochs to train attack models.")


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


def create_simplenet_model():
    """Create SimpleNet model that matches the target architecture."""
    model = tf.keras.Sequential([
        # Conv1
        layers.Conv2D(6, (5, 5), activation='relu', padding='valid', input_shape=(32, 32, 3)),
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
    return model


def target_model_fn():
    """Returns a SimpleNet model wrapped in a scikit-learn compatible interface."""
    # Use the same architecture as the target model
    model = KerasClassifier(
        model=create_simplenet_model, # Use SimpleNet architecture instead
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
        # epochs=FLAGS.target_epochs, # epochs for fit will be passed in fit_kwargs
        # verbose=0 # verbose for fit will be passed in fit_kwargs
    )
    return model


def create_attack_keras_model(): # Renamed, this will be our build_fn for the attack model
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(128, activation="relu", input_shape=(NUM_CLASSES,))) # Input is prediction vector
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid")) # Binary output: in or out
    # model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"]) # Optional here
    return model


def attack_model_fn():
    model = KerasClassifier(
        model=create_attack_keras_model,
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
        # epochs=FLAGS.attack_epochs, # epochs for fit will be passed in fit_kwargs
        # verbose=0
    )
    return model


def load_pytorch_model(npz_path):
    """Load PyTorch model weights from .npz file"""
    try:
        data = np.load(npz_path)
        # Convert numpy arrays to torch tensors
        weights = {k: torch.from_numpy(v) for k, v in data.items()}
        return weights
    except Exception as e:
        print(f"Error loading model from {npz_path}: {e}")
        return None


def create_tf_model_from_pytorch(pytorch_model_path, model_type="SimpleNet"):
    """Convert PyTorch model to TensorFlow model"""
    
    # Load PyTorch weights
    weights = load_pytorch_model(pytorch_model_path)
    if weights is None:
        raise ValueError(f"Failed to load weights from {pytorch_model_path}")
    
    if model_type == "SimpleNet":
        # Create equivalent TensorFlow model
        model = tf.keras.Sequential([
            # Conv1
            layers.Conv2D(6, (5, 5), activation='relu', padding='valid', input_shape=(32, 32, 3)),
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
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Map weights from PyTorch to TensorFlow
        # PyTorch weights are in NCHW format, TensorFlow expects NHWC
        tf_weights = []
        
        # Conv1 weights
        conv1_w = weights['model.conv1.weight'].numpy()
        conv1_w = np.transpose(conv1_w, (2, 3, 1, 0))  # NCHW -> NHWC
        conv1_b = weights['model.conv1.bias'].numpy()
        tf_weights.extend([conv1_w, conv1_b])
        
        # Conv2 weights
        conv2_w = weights['model.conv2.weight'].numpy()
        conv2_w = np.transpose(conv2_w, (2, 3, 1, 0))  # NCHW -> NHWC
        conv2_b = weights['model.conv2.bias'].numpy()
        tf_weights.extend([conv2_w, conv2_b])
        
        # FC layers weights
        fc1_w = weights['model.fc1.weight'].numpy()
        fc1_b = weights['model.fc1.bias'].numpy()
        tf_weights.extend([fc1_w.T, fc1_b])  # Transpose weight matrix
        
        fc2_w = weights['model.fc2.weight'].numpy()
        fc2_b = weights['model.fc2.bias'].numpy()
        tf_weights.extend([fc2_w.T, fc2_b])
        
        fc3_w = weights['model.fc3.weight'].numpy()
        fc3_b = weights['model.fc3.bias'].numpy()
        tf_weights.extend([fc3_w.T, fc3_b])
        
        # Set weights
        model.set_weights(tf_weights)
        
    return model


def prepare_model_for_mia(model):
    """Prepare the model for MIA by adding necessary layers"""
    # Add any preprocessing or additional layers needed for MIA
    return model


def demo(argv):
    del argv  # Unused.

    (X_train, y_train), (X_test, y_test) = get_data()

    # Train the target model.
    print("Training the target model...")
    # target_model = target_model_fn() # This now returns a KerasClassifier
    # The KerasClassifier's `fit` method is scikit-learn like.
    # We need to instantiate the Keras model first then fit it, or pass epochs to the wrapper.
    
    # Let's adjust how target_model is created and trained for clarity with KerasClassifier
    # The KerasClassifier is stateful after fit.
    target_model = KerasClassifier(
        model=create_simplenet_model,
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
        epochs=FLAGS.target_epochs, # Pass epochs here for the main target model
        validation_split=0.1,
        verbose=1
    )
    target_model.fit(X_train, y_train) # y_train should be one-hot for categorical_crossentropy


    # Train the shadow models.
    # ShadowModelBundle expects target_model_fn to return a new, callable model constructor
    # or an unfitted scikit-learn estimator.
    # Our modified target_model_fn already returns a KerasClassifier instance.
    # The KerasClassifier itself is the "estimator".
    smb = ShadowModelBundle(
        target_model_fn, # This function now returns a KerasClassifier instance factory
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=FLAGS.num_shadows,
    )
    
    # We assume that attacker's data were not seen in target's training.
    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        X_test, y_test, test_size=0.1, stratify=y_test # Added stratify
    )
    print(f"Attacker X_train shape for shadow: {attacker_X_train.shape}, y_train shape: {attacker_y_train.shape}")


    print("Training the shadow models...")
    # fit_kwargs will be passed to the .fit() method of each KerasClassifier shadow model
    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=FLAGS.target_epochs,
            verbose=1, # Make shadow training verbose for debugging
            validation_data=(attacker_X_test, attacker_y_test), # Keras .fit() expects this
            # For KerasClassifier, batch_size might also be a good fit_kwarg
            batch_size=32 # Example batch size
        ),
    )
    
    # ... rest of your demo function ...
    # The AttackModelBundle should also work with the KerasClassifier from attack_model_fn
    print("Initializing AttackModelBundle...")
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

    # Fit the attack models.
    print("Training the attack models...")
    amb.fit(
        X_shadow, y_shadow, fit_kwargs=dict(
            epochs=FLAGS.attack_epochs, 
            verbose=1,
            batch_size=32 # Example batch size
            )
    )

    # Test the success of the attack.
    print("Preparing attack data for evaluation...")
    # Prepare examples that were in the training, and out of the training.
    # Ensure y_train and y_test are correctly shaped (e.g. one-hot for target_model.predict_proba)
    data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE] 
    data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    # target_model here is the KerasClassifier instance we trained.
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )

    # Compute the attack accuracy.
    print("Making attack predictions...")
    attack_guesses = amb.predict(attack_test_data) # amb.predict should work if attack models are KerasClassifiers
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    print(f"Attack Accuracy: {attack_accuracy}")


def demo_with_converted_model(argv):
    del argv  # Unused.

    # Load and convert your PyTorch model
    model_path = "results/CFL/global_models/node_n1_round_5_global_model.npz"
    
    try:
        # First, let's inspect the .npz file
        data = np.load(model_path)
        print("Available keys in the .npz file:", data.files)
        
        # Convert the model
        converted_model = create_tf_model_from_pytorch(model_path, model_type="SimpleNet")
        
        # Get the data
        (X_train, y_train), (X_test, y_test) = get_data()
        
        # Train the shadow models
        print("Training the shadow models...")
        smb = ShadowModelBundle(
            target_model_fn,
            shadow_dataset_size=SHADOW_DATASET_SIZE,
            num_models=FLAGS.num_shadows,
        )
        
        # Split data for shadow models
        attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
            X_test, y_test, test_size=0.1, stratify=y_test
        )
        
        # Train shadow models
        X_shadow, y_shadow = smb.fit_transform(
            attacker_X_train,
            attacker_y_train,
            fit_kwargs=dict(
                epochs=FLAGS.target_epochs,
                verbose=1,
                validation_data=(attacker_X_test, attacker_y_test),
                batch_size=32
            ),
        )
        
        # Initialize and train attack model
        print("Training the attack model...")
        amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)
        amb.fit(
            X_shadow, y_shadow,
            fit_kwargs=dict(
                epochs=FLAGS.attack_epochs,
                verbose=1,
                batch_size=32
            )
        )
        
        # Prepare attack data
        print("Preparing attack data for evaluation...")
        data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
        data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]
        
        # Create a wrapper for the converted model that provides predict_proba
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
            
            def predict_proba(self, X):
                return self.model.predict(X)
        
        # Use the wrapper instead of KerasClassifier
        target_model = ModelWrapper(converted_model)
        
        # Prepare attack data
        attack_test_data, real_membership_labels = prepare_attack_data(
            target_model, data_in, data_out
        )
        
        # Make attack predictions
        print("Making attack predictions...")
        attack_guesses = amb.predict(attack_test_data)
        attack_accuracy = np.mean(attack_guesses == real_membership_labels)
        
        print(f"Attack Accuracy: {attack_accuracy}")
        
    except Exception as e:
        print(f"Error in demo_with_converted_model: {e}")
        raise


if __name__ == "__main__":
    app.run(demo_with_converted_model)