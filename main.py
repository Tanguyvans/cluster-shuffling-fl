from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.mps
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset

import config
from models import Net, MobileNetV2

DEVICE = "mps" if torch.mps.is_available() else "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

# Determine dataset-specific parameters
if config.DATASET == "flwrlabs/femnist":
    NUM_CLASSES = 62
    INPUT_DIMS = (3, 28, 28)  # FEMNIST is 28x28, converted to 3 channels
elif config.DATASET == "cifar10":
    NUM_CLASSES = 10
    INPUT_DIMS = (3, 32, 32)  # CIFAR-10 is 32x32, 3 channels
else:
    # Default or raise error for unknown dataset
    # For now, let's assume a default or handle as needed.
    # For safety, if these are critical, you might want to raise an error.
    print(f"Warning: Unknown dataset {config.DATASET}. Using default NUM_CLASSES=10, INPUT_DIMS=(3, 32, 32)")
    NUM_CLASSES = 10
    INPUT_DIMS = (3, 32, 32)

# Global variable to store the prepared test set
_GLOBAL_TEST_SET = None

def apply_transforms(batch):
    to_tensor_transform = transforms.ToTensor()
    normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # Handle image key
    if 'image' in batch:
        input_image_key = 'image'
    elif 'img' in batch:
        input_image_key = 'img'
    else:
        raise KeyError(f"Neither 'img' nor 'image' found in batch. Keys: {list(batch.keys())}")

    # Process images
    source_images = batch[input_image_key]
    processed_images = []
    for img_pil in source_images:
        tensor_img = to_tensor_transform(img_pil)
        if tensor_img.shape[0] == 1:  # If grayscale, convert to 3 channels
            tensor_img = tensor_img.repeat(3, 1, 1)
        tensor_img = normalize_transform(tensor_img)
        processed_images.append(tensor_img)
    
    batch['img'] = processed_images
    if input_image_key != 'img' and input_image_key in batch:
        del batch[input_image_key]

    # Handle label key
    if 'character' in batch:  # FEMNIST uses 'character'
        batch['label'] = batch['character']
        del batch['character']
    elif 'writer_id' in batch:
        batch['label'] = batch['writer_id']
        del batch['writer_id']
    
    return batch

def load_datasets(dataset: str, partition_id: int):
    global _GLOBAL_TEST_SET
    
    # Initialize the FederatedDataset with the original dataset
    fds = FederatedDataset(dataset=dataset, partitioners={"train": config.NUM_CLIENTS})
    
    # Try to load the predefined test split
    testloader = None
    try:
        test_dataset = fds.load_split("test")
        print(f"Using predefined test split for {dataset}")
        test_dataset = test_dataset.with_transform(apply_transforms)
        testloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    except ValueError:
        # No predefined test split, we'll create one from the training data
        if _GLOBAL_TEST_SET is None:
            print(f"No predefined test split for {dataset}. Creating one from client data.")
            
            # We'll use a portion of each client's data to create a global test set
            # This approach is simpler and should work with the current API
            # For the first client (partition_id 0), we'll create a test set from its validation data
            
    # Load the client's partition
    partition = fds.load_partition(partition_id)
    
    # Create an 80/20 split for train/val 
    partition_train_val = partition.train_test_split(test_size=0.2, seed=42)
    
    # Apply transforms
    partition_train_val = partition_train_val.with_transform(apply_transforms)
    
    # Create dataloaders
    trainloader = DataLoader(
        partition_train_val["train"], batch_size=config.BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_val["test"], batch_size=config.BATCH_SIZE)
    
    # If we don't have a testloader yet (for datasets without a predefined test split)
    if testloader is None and partition_id == 0:
        # For the first client only, we'll use its validation set as a global test set
        # This is a simplified approach but should work for demonstration purposes
        _GLOBAL_TEST_SET = partition_train_val["test"]
        testloader = DataLoader(_GLOBAL_TEST_SET, batch_size=config.BATCH_SIZE)
        print(f"Created global test set from client 0's validation data with {len(_GLOBAL_TEST_SET)} samples")
    elif testloader is None and _GLOBAL_TEST_SET is not None:
        # For other clients, reuse the test set created from client 0
        testloader = DataLoader(_GLOBAL_TEST_SET, batch_size=config.BATCH_SIZE)
    
    return trainloader, valloader, testloader

def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

####
main_trainloader, main_valloader, main_testloader = load_datasets(dataset=config.DATASET, partition_id=0)

if config.MODEL == "mobilenetv2":
    net = MobileNetV2(num_classes=NUM_CLASSES).to(DEVICE)
# Add elif for other models like ResNet18, EfficientNetB0 if you add them to config.MODEL options
# elif config.MODEL == "resnet18":
#     net = ResNet18(num_classes=NUM_CLASSES).to(DEVICE)
else:  # Default to Net
    net = Net(input_dims=INPUT_DIMS, num_classes=NUM_CLASSES).to(DEVICE)

# Perform some initial training epochs and validation
if len(main_trainloader.dataset) > 0 and len(main_valloader.dataset) > 0:
    for epoch in range(5): # Consider reducing epochs or making conditional for quick tests
        train(net, main_trainloader, 1)
        loss, accuracy = test(net, main_valloader)
        print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")
else:
    print("Skipping initial local training/validation: dataset for partition 0 is too small or empty.")

# Evaluate on the main test set, if available
if main_testloader and len(main_testloader.dataset) > 0 :
    loss, accuracy = test(net, main_testloader)
    print(f"Initial (local) test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
else:
    print("Initial (local) test set performance: No global testloader available or it's empty for this dataset.")

####

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

####

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    
####

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model based on config
    if config.MODEL == "mobilenetv2":
        model = MobileNetV2(num_classes=NUM_CLASSES).to(DEVICE)
    # Add other models here based on config.MODEL
    # elif config.MODEL == "resnet18":
    #     model = ResNet18(num_classes=NUM_CLASSES).to(DEVICE) 
    # elif config.MODEL == "efficientnetb0":
    #     model = EfficientNetB0(num_classes=NUM_CLASSES).to(DEVICE)
    else:  # Default to Net
        model = Net(input_dims=INPUT_DIMS, num_classes=NUM_CLASSES).to(DEVICE)


    # Load data for the specific client partition
    partition_id = context.node_config["partition-id"]
    # Pass the dataset name from config
    trainloader, valloader, _ = load_datasets(dataset=config.DATASET, partition_id=partition_id)

    # Create a single Flower client
    return FlowerClient(model, trainloader, valloader).to_client()

# Create the ClientApp
client = ClientApp(client_fn=client_fn)

####

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Create FedAvg strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=10,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=10,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # Ensure this is the strategy used
)

####

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    server_config = ServerConfig(num_rounds=5) # Renamed to server_config to avoid conflict with main config module

    return ServerAppComponents(strategy=strategy, config=server_config)


# Create a new server instance with the updated FedAvg strategy
server = ServerApp(server_fn=server_fn)

####

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=config.NUM_CLIENTS,
    backend_config=backend_config,
)