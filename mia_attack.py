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


def create_keras_model():
    """The architecture of the target (victim) model.
    This function will be passed to KerasClassifier.
    """
    model = tf.keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(WIDTH, HEIGHT, CHANNELS)))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    # Compilation is handled by KerasClassifier or can be done here
    # model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"]) # Optional here
    return model


def target_model_fn():
    """Returns a Keras model wrapped in a scikit-learn compatible interface."""
    # KerasClassifier will handle compilation if loss and optimizer are provided.
    # It automatically provides fit, predict, predict_proba.
    model = KerasClassifier(
        model=create_keras_model, # Pass the function that creates the Keras model
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
        model=create_keras_model,
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


if __name__ == "__main__":
    app.run(demo)