import numpy as np

def sMAPE(outputs, targets):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) for evaluating the model.
    It is the sum of the absolute difference between the predicted and actual values divided by the average of
    the predicted and actual value, therefore giving a percentage measuring the amount of error :
    100/n * sum(|F_t - A_t| / (|F_t| + |A_t|) / 2) with t = 1 to n

    :param outputs : predicted values
    :param targets: real values
    :return: sMAPE
    """

    return 100 / len(targets) * np.sum(np.abs(outputs - targets) / (np.abs(outputs + targets)) / 2)