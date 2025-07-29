import torch

def choice_device(device):
    """
        A function to choose the device

        :param device: the device to choose (cpu, gpu or mps)
        """
    if torch.cuda.is_available() and device != "cpu":
        # on Windows, "cuda:0" if torch.cuda.is_available()
        device = "cuda:0"

    elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and device != "cpu":
        """
        on Mac : 
        - torch.backends.mps.is_available() ensures that the current MacOS version is at least 12.3+
        - torch.backends.mps.is_built() ensures that the current PyTorch installation was built with MPS activated.
        """
        device = "mps"

    else:
        device = "cpu"

    print("The device is : ", device)
    return device