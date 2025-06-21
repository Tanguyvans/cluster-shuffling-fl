settings = {
    "name_dataset": "cifar10",  # "cifar10" or "cifar100" or "caltech256"
    "arch": "simplenet",  # "mobilenet" or "resnet18" or "shufflenet"
    "pretrained": True,
    "patience": 3,
    "batch_size": 32,
    "n_epochs": 10,
    "number_of_nodes": 1,
    "number_of_clients_per_node": 6,
    "min_number_of_clients_in_cluster": 3,

    "check_usefulness": False,
    "coef_useful": 1.05,   # 1.05
    "tolerance_ceil": 0.08,

    "poisoned_number": 0,
    "n_rounds": 10,
    "choice_loss": "cross_entropy",
    "choice_optimizer": "Adam",
    "lr": 0.001,              # Back to original LR
    "choice_scheduler": "StepLR",  # Re-enable StepLR to test the fix
    "step_size": 3,
    "gamma": 0.5,

    "diff_privacy": False,
    "noise_multiplier": 0.1,  
    "max_grad_norm": 0.5,     
    "delta": 1e-5,
    "epsilon": 5.0,           

    "clustering": False,       # RE-ENABLE MPC - Testing the fix!
    "type_ss": "additif",
    "threshold": 3,
    "m": 3,
    "ts": 5,                  

    "save_figure": True,
    "matrix_path": "results/CFL/matrix_path",
    "roc_path": "results/CFL/roc_path",
    "pretrained": True,
}
