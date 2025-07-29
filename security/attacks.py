import numpy as np

def data_poisoning(data, poisoning_type, n_classes, poisoned_number, number_of_nodes=1, clients_per_node=1, target_class=None):

    if poisoning_type == "rand":
        poison_per_node = poisoned_number // number_of_nodes
        for node in range(number_of_nodes):
            for i in range(poison_per_node):
                client_index = node * clients_per_node + i
                data[client_index][1] = random_poisoning(n_classes, n=len(data[client_index][1]))
    
    elif poisoning_type == "targeted":
        # Targeted poisoning: poison only the first class
        for i in range(poisoned_number):
            data[i][1] = targeted_poisoning(data[i][1], n_classes)
    
    else:
        raise ValueError(f"Unknown poisoning type: {poisoning_type}. Choose from: 'rand' or 'targeted'")
    

def random_poisoning(n_classes, n):
    return np.random.randint(0, n_classes, size=n).tolist()

def targeted_poisoning(labels, n_classes, target_class=0):
    poisoned_labels = []
    for label in labels:
        if label == target_class:
            # Generate a random wrong label for the target class
            wrong_labels = list(range(n_classes))
            wrong_labels.remove(target_class)  # Remove the correct label
            poisoned_labels.append(np.random.choice(wrong_labels))
        else:
            poisoned_labels.append(label)
    return poisoned_labels