settings = {
    "name_dataset": "cifar10",  # "cifar10" or "cifar100" or "caltech256"
    "arch": "simplenet",  # "mobilenet" or "resnet18" or "shufflenet"
    "pretrained": True,
    "patience": 3,
    "batch_size": 32,
    "n_epochs": 2,
    "number_of_nodes": 1,
    "number_of_clients_per_node": 6,
    "min_number_of_clients_in_cluster": 3,

    "check_usefulness": True,
    "coef_useful": 1.05,   # 1.05
    "tolerance_ceil": 0.08,

    "poisoned_number": 0,
    "n_rounds": 5,
    "choice_loss": "cross_entropy",
    "choice_optimizer": "Adam",
    "lr": 0.001,
    "choice_scheduler": "StepLR",  # "StepLR" or None
    "step_size": 3,
    "gamma": 0.5,

    'use_clustering': False,
    "secret_sharing": "additif",  # "additif" or "shamir"

    "diff_privacy": False,
    "k": 3,
    "m": 3,
    "ts": 5,
}
