import torch
from .metrics import sMAPE

def fct_loss(choice_loss):
    """
    A function to choose the loss function

    :param choice_loss: the choice of the loss function
    :return: the loss function
    """
    choice_loss = choice_loss.lower()
    print("choice_loss : ", choice_loss)

    if choice_loss == "cross_entropy":
        loss = torch.nn.CrossEntropyLoss()

    elif choice_loss == "binary_cross_entropy":
        loss = torch.nn.BCELoss()

    elif choice_loss == "bce_with_logits":
        loss = torch.nn.BCEWithLogitsLoss()

    elif choice_loss == "smape":
        loss = sMAPE

    elif choice_loss == "mse":
        loss = torch.nn.MSELoss(reduction='mean')

    elif choice_loss == "scratch":
        return None

    else:
        print("Warning problem : unspecified loss function")
        return None

    print("loss : ", loss)

    return loss

def choice_optimizer_fct(model, choice_optim="Adam", lr=0.001, momentum=0.9, weight_decay=1e-6):
    """
    A function to choose the optimizer

    :param model: the model
    :param choice_optim: the choice of the optimizer
    :param lr: the learning rate
    :param momentum: the momentum
    :param weight_decay: the weight decay
    :return: the optimizer
    """

    choice_optim = choice_optim.lower()

    if choice_optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    elif choice_optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,
                                     amsgrad='adam' == 'amsgrad', )

    elif choice_optim == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    else:
        print("Warning problem : unspecified optimizer")
        return None

    return optimizer

def choice_scheduler_fct(optimizer, choice_scheduler=None, step_size=30, gamma=0.1, base_lr=0.0001, max_lr=0.1):
    """
    A function to choose the scheduler

    :param optimizer: the optimizer
    :param choice_scheduler: the choice of the scheduler
    :param step_size: the step size
    :param gamma: the gamma
    :param base_lr: the base learning rate
    :param max_lr: the maximum learning rate
    :return: the scheduler
    """

    if choice_scheduler:
        choice_scheduler = choice_scheduler.lower()

    if choice_scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif choice_scheduler == "exponentiallr":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif choice_scheduler == "cycliclr":
        # use per batch (not per epoch)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr)

    elif choice_scheduler == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size, eta_min=0)

    elif choice_scheduler == "multisteplr":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=gamma)

    elif choice_scheduler == "plateaulr":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                                               threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                               min_lr=0, eps=1e-08, verbose=False)

    elif choice_scheduler is None:
        return choice_scheduler

    else:
        print("Warning problem : unspecified scheduler")
        # There are other schedulers like OneCycleLR, etc.
        # but generally, they are used per batch and not per epoch.
        # For example, OneCycleLR : total_steps = n_epochs * steps_per_epoch
        # https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling
        return None

    print("scheduler : ", scheduler)
    return scheduler