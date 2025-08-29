import torch
import numpy as np
import os
from typing import List, Dict, Optional


class PyTorchSecretSharing:
    """Pure PyTorch implementation of Secret Sharing for Secure Multi-Party Computation (SMPC)"""
    
    def __init__(self, method: str = "additif", threshold: int = 3, num_shares: int = 3):
        """
        Initialize PyTorch Secret Sharing
        
        Args:
            method: "additif" for additive secret sharing or "shamir" for Shamir's secret sharing
            threshold: Minimum number of shares needed for reconstruction (Shamir only)
            num_shares: Total number of shares to generate
        """
        self.method = method
        self.threshold = threshold
        self.num_shares = num_shares
        
    def create_shares(self, model_state_dict: Dict[str, torch.Tensor], 
                     device: Optional[torch.device] = None) -> List[Dict[str, torch.Tensor]]:
        """
        Create secret shares from model state dictionary
        
        Args:
            model_state_dict: PyTorch model state dictionary
            device: Device to perform operations on (cuda/cpu)
            
        Returns:
            List of shares, each containing encrypted tensors
        """
        if device is None:
            device = next(iter(model_state_dict.values())).device
            
        if self.method == "additif":
            return self._additive_sharing(model_state_dict, device)
        elif self.method == "shamir":
            return self._shamir_sharing(model_state_dict, device)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _additive_sharing(self, model_state_dict: Dict[str, torch.Tensor], 
                         device: torch.device) -> List[Dict[str, torch.Tensor]]:
        """
        Additive secret sharing: secret = share1 + share2 + ... + shareN
        """
        shares = []
        
        with torch.no_grad():
            # Convert any numpy arrays to tensors first
            converted_dict = {}
            for key, value in model_state_dict.items():
                if isinstance(value, np.ndarray):
                    converted_dict[key] = torch.from_numpy(value).to(device)
                elif isinstance(value, torch.Tensor):
                    converted_dict[key] = value.to(device)
                else:
                    # Try to convert to tensor
                    converted_dict[key] = torch.tensor(value).to(device)
            
            # Generate random shares for all but the last one
            for i in range(self.num_shares - 1):
                share = {}
                for key, tensor in converted_dict.items():
                    # Generate random values in a reasonable range for model weights
                    # Using smaller range to avoid numerical overflow
                    share[key] = torch.randn_like(tensor, device=device) * 0.01
                shares.append(share)
            
            # Final share = original - sum(all other shares)
            final_share = {}
            for key, tensor in converted_dict.items():
                sum_others = torch.zeros_like(tensor, device=device)
                for share in shares:
                    sum_others += share[key]
                final_share[key] = tensor - sum_others
            shares.append(final_share)
        
        return shares

    def _shamir_sharing(self, model_state_dict: Dict[str, torch.Tensor], 
                       device: torch.device) -> List[Dict[str, torch.Tensor]]:
        """
        Shamir's secret sharing using polynomial evaluation
        """
        shares = [{} for _ in range(self.num_shares)]
        
        with torch.no_grad():
            for key, tensor in model_state_dict.items():
                # Flatten tensor for polynomial operations
                flat_tensor = tensor.flatten()
                
                # Generate polynomial coefficients for each element
                # coeffs[i, :] = [secret_val, coeff1, coeff2, ..., coeff_{threshold-1}]
                coeffs = torch.randn(flat_tensor.numel(), self.threshold, 
                                   device=device, dtype=tensor.dtype)
                coeffs[:, 0] = flat_tensor  # First coefficient is the secret
                
                # Evaluate polynomial at points 1, 2, ..., num_shares
                for share_idx in range(self.num_shares):
                    x = share_idx + 1  # Evaluation point
                    
                    # Polynomial evaluation: y = c0 + c1*x + c2*x^2 + ...
                    x_powers = torch.pow(x, torch.arange(self.threshold, device=device, dtype=tensor.dtype))
                    share_values = torch.sum(coeffs * x_powers.unsqueeze(0), dim=1)
                    
                    # Reshape back to original tensor shape
                    shares[share_idx][key] = share_values.reshape(tensor.shape)
        
        return shares
    
    def reconstruct_secret(self, shares: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Reconstruct original secret from shares
        """
        if self.method == "additif":
            return self._reconstruct_additive(shares)
        elif self.method == "shamir":
            return self._reconstruct_shamir(shares)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _reconstruct_additive(self, shares: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Reconstruct from additive shares by simple summation"""
        if not shares:
            return {}
        
        reconstructed = {}
        first_share = shares[0]
        
        with torch.no_grad():
            for key in first_share.keys():
                reconstructed[key] = torch.zeros_like(first_share[key])
                for share in shares:
                    reconstructed[key] += share[key]
        
        return reconstructed
    
    def _reconstruct_shamir(self, shares: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Reconstruct from Shamir shares using Lagrange interpolation"""
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares, got {len(shares)}")
        
        # Use only the first 'threshold' shares
        shares = shares[:self.threshold]
        reconstructed = {}
        
        if not shares:
            return {}
        
        first_share = shares[0]
        device = next(iter(first_share.values())).device
        
        with torch.no_grad():
            # Evaluation points (1, 2, 3, ...)
            x_points = torch.arange(1, len(shares) + 1, device=device, dtype=torch.float32)
            
            for key in first_share.keys():
                # Stack all share values for this key
                y_values = torch.stack([share[key] for share in shares])  # [num_shares, *tensor_shape]
                
                # Flatten for easier computation
                original_shape = y_values.shape[1:]
                y_flat = y_values.view(len(shares), -1)  # [num_shares, num_elements]
                
                # Lagrange interpolation to find value at x=0 (the secret)
                result = torch.zeros_like(y_flat[0])
                
                for i in range(len(shares)):
                    # Compute Lagrange basis polynomial Li(0)
                    li_0 = torch.tensor(1.0, device=device, dtype=torch.float32)
                    for j in range(len(shares)):
                        if i != j:
                            li_0 *= (0 - x_points[j]) / (x_points[i] - x_points[j])
                    
                    result += li_0 * y_flat[i]
                
                # Reshape back to original shape
                reconstructed[key] = result.view(original_shape).to(first_share[key].dtype)
        
        return reconstructed
    
    def save_shares(self, shares: List[Dict[str, torch.Tensor]], client_id: str, 
                   round_num: int, save_dir: str) -> List[str]:
        """Save shares as .pt files"""
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        
        for i, share in enumerate(shares):
            filepath = f"{save_dir}/c{client_id}_round_{round_num}_share_{i}.pt"
            
            torch.save({
                'share': share,
                'client_id': client_id,
                'round': round_num,
                'share_index': i,
                'method': self.method,
                'threshold': self.threshold,
                'num_shares': self.num_shares
            }, filepath)
            
            saved_paths.append(filepath)
        
        return saved_paths
    
    def load_shares(self, filepaths: List[str]) -> List[Dict[str, torch.Tensor]]:
        """Load shares from .pt files"""
        shares = []
        
        for filepath in filepaths:
            data = torch.load(filepath, map_location='cpu')
            shares.append(data['share'])
        
        return shares


# Legacy compatibility function for existing code
def encrypt_tensor(secret, n_shares=3):
    """
    Legacy wrapper for tensor encryption using additive sharing - now uses PyTorch
    
    Args:
        secret: PyTorch tensor or numpy array to encrypt
        n_shares: Number of shares to generate
        
    Returns:
        List of share tensors (converted back to numpy for compatibility)
    """
    # Convert numpy to torch if needed
    if hasattr(secret, 'numpy') or str(type(secret)).find('numpy') != -1:
        import numpy as np
        tensor = torch.from_numpy(secret if isinstance(secret, np.ndarray) else np.array(secret))
        return_numpy = True
    else:
        tensor = secret
        return_numpy = False
    
    device = tensor.device if hasattr(tensor, 'device') else torch.device('cpu')
    
    # Use PyTorchSecretSharing
    temp_dict = {'tensor': tensor}
    ss = PyTorchSecretSharing(method="additif", num_shares=n_shares)
    shares = ss.create_shares(temp_dict, device)
    
    # Extract tensor and convert back to numpy if needed
    result = [share['tensor'] for share in shares]
    if return_numpy:
        result = [share.detach().cpu().numpy() for share in result]
    
    return result

def apply_additif(input_list, n_shares=3):
    """
    Legacy wrapper for apply_additif using PyTorch backend
    :param input_list: The secret to share (list of tensors)
    :param n_shares: the number of parts to divide the secret (so the number of participants)
    :return: the list of shares for each client where each share is a list of tensors
    """
    # Convert to state dict format for PyTorchSecretSharing
    model_dict = {f'layer_{i}': tensor for i, tensor in enumerate(input_list)}
    
    # Determine device
    device = input_list[0].device if hasattr(input_list[0], 'device') else torch.device('cpu')
    
    # Use PyTorchSecretSharing
    ss = PyTorchSecretSharing(method="additif", num_shares=n_shares)
    shares = ss.create_shares(model_dict, device)
    
    # Convert back to list format for compatibility
    encrypted_list = []
    for share in shares:
        share_list = [share[f'layer_{i}'] for i in range(len(input_list))]
        encrypted_list.append(share_list)
    
    return encrypted_list

def calculate_y(x, poly):
    """
    Function to calculate the value of y from a polynomial and a value of x:
    y = poly[0] + x*poly[1] + x^2*poly[2] + ...

    :param x: the value of x
    :param poly: the list of coefficients of the polynomial

    :return: the value of y
    """
    y = sum([poly[i] * x ** i for i in range(len(poly))])
    return y

def apply_shamir(input_list, n_shares=2, k=3):
    """
    Function to apply shamir's secret sharing algorithm
    :param input_list: secret to share (a list of tensors of values of type numpy.ndarray)
    :param n_shares: the number of parts to divide the secret (so the number of participants)
    :param k: the minimum number of parts to reconstruct the secret, called threshold (with a polynomial of order k-1)
    :return: the list of shares for each client
    """
    list_clients = [[]for _ in range(n_shares)]
    for weights_layer in input_list:
        y_i = apply_poly(weights_layer.flatten(), n_shares, k)[1]
        for i in range(n_shares):
            list_clients[i].append(y_i[i])

    return list_clients

def apply_poly(S, N, K):
    """
    Function to perform secret sharing algorithm and encode the given secret when the secret is a tensor of values
    :param S: The secret to share (a tensor of values of type numpy.ndarray)
    :param N: the number of parts to divide the secret (so the number of participants)
    :param K: the minimum number of parts to reconstruct the secret, called threshold (with a polynomial of order K-1)

    :return: the points generated from the polynomial we created for each part
    """
    # A tensor of polynomials to store the coefficients of each polynomial
    # The element i of the column 0 corresponds to the constant of the polynomial i which is the secret S_i
    # that we want to encrypt
    poly = np.array([[S[i]] + [0] * (K - 1) for i in range(len(S))])

    # Chose randomly K - 1 numbers for each row except the first column which is the secret
    poly[:, 1:] = np.random.randint(1, 996, np.shape(poly[:, 1:])) + 1

    # Generate N points for each polynomial we created
    points = np.array([
        [
            (x, calculate_y(x, poly[i])) for x in range(1, N + 1)
        ] for i in range(len(S))
    ]).T
    return points

def apply_smpc(input_list, n_shares=2, type_ss="additif", threshold=3):
    """
    Main function to apply secret sharing algorithm using PyTorch backend
    :param input_list: list of PyTorch tensors to share
    :param n_shares: the number of parts to divide the secret
    :param type_ss: the type of secret sharing algorithm ("additif" or "shamir")
    :param threshold: the minimum number of parts to reconstruct the secret
    :return: the list of shares for each client and the shape of the secret
    """
    if type_ss == "additif":
        return apply_additif(input_list, n_shares), None

    elif type_ss == "shamir":
        secret_shape = [tensor.shape for tensor in input_list]
        
        # Convert to state dict format
        model_dict = {f'layer_{i}': tensor for i, tensor in enumerate(input_list)}
        device = input_list[0].device if hasattr(input_list[0], 'device') else torch.device('cpu')
        
        # Use PyTorchSecretSharing
        ss = PyTorchSecretSharing(method="shamir", threshold=threshold, num_shares=n_shares)
        shares = ss.create_shares(model_dict, device)
        
        # Convert to (x, y) format expected by legacy code
        list_shares = []
        for i, share in enumerate(shares):
            x = i + 1  # x-coordinate
            y = [share[f'layer_{j}'] for j in range(len(input_list))]  # y-values
            list_shares.append((x, y))
        
        return list_shares, secret_shape

    else:
        raise ValueError("Type of secret sharing not recognized")

def sum_shares_additif(encrypted_list):
    """
    Function to sum the parts received by an entity when the secret sharing algorithm used is additif.
    Uses PyTorch tensors for computation.
    :param encrypted_list: list of shares to sum
    :return: the sum of the parts received as NumPy arrays for Flower compatibility
    """
    if not encrypted_list:
        return []
    
    decrypted_list = []
    n_shares = len(encrypted_list)

    for layer in range(len(encrypted_list[0])):  # for each layer
        # Initialize sum with zeros_like to preserve tensor properties
        sum_tensor = torch.zeros_like(encrypted_list[0][layer])
        
        # Sum all shares for this layer
        for i in range(n_shares):
            sum_tensor += encrypted_list[i][layer]

        # Convert back to NumPy for Flower compatibility
        decrypted_list.append(sum_tensor.detach().cpu().numpy())

    return decrypted_list

def sum_shares_shamir(encrypted_list):
    """
    Function to sum the parts received by an entity when the secret sharing algorithm used is Shamir.
    Uses PyTorch tensors for computation.
    :param encrypted_list: list of shares to sum where each element is a tuple (x, y)
    :return: the sum of the parts received
    """
    result_som = {}  # sum for a given client
    for (x, list_weights) in encrypted_list:
        if x in result_som:
            for layer in range(len(list_weights)):
                result_som[x][layer] += list_weights[layer]
        else:
            # Clone tensors to avoid modifying original data
            result_som[x] = [w.clone() if hasattr(w, 'clone') else w for w in list_weights]

    return result_som

def sum_shares(encrypted_list, type_ss="additif"):
    """
    Function to sum the parts received by an entity
    :param encrypted_list: list of lists to decrypt
    :param type_ss: type of secret sharing algorithm used
    :return: the sum of the parts received
    """
    if type_ss == "additif":
        return sum_shares_additif(encrypted_list)

    elif type_ss == "shamir":
        return sum_shares_shamir(encrypted_list)

    else:
        raise ValueError("Type of secret sharing not recognized")

def generate_secret_shamir(x, y, m):
    """
    Function to generate the secret from the given points
    :param x: list of x
    :param y: list of y
    :param m: number of points to use for the reconstruction

    :return: the secret
    """
    # Initialisation of the answer
    ans = 0

    # loop to go through the given points
    for i in range(m):
        l = y[i]
        for j in range(m):
            # Compute the Lagrange polynomial
            if i != j:
                temp = -x[j] / (x[i] - x[j])  # L_i(x=0)
                l *= temp

        ans += l

    # return the secret
    return ans

def combine_shares_node(secret_list):
    """
    Function to combine the shares of the secret - PyTorch compatible
    :param secret_list: list of shares of each client
    :return: dictionary of the combined secret
    """
    secret_dic_final = {}
    for id_client in range(len(secret_list)):
        for x, list_weights in secret_list[id_client].items():
            if x in secret_dic_final:
                for layer in range(len(list_weights)):
                    secret_dic_final[x][layer] += list_weights[layer]
            else:
                # Clone tensors to avoid modifying original data
                secret_dic_final[x] = [w.clone() if hasattr(w, 'clone') else w for w in list_weights]
    return secret_dic_final

def decrypt_shamir_node(secret_dic_final, secret_shape, m):
    """
    Function to decrypt the secret on the node side with Shamir secret sharing
    :param secret_dic_final: dictionary of the secret, so secret_dic_final[x][layer]
    :param secret_shape: list of shapes of the layers
    :param m: number of shares to use for the reconstruction of the secret
    :return: list of the decrypted secret, so decrypted_result[layer]  = weights_layer
    """
    x_combine = list(secret_dic_final.keys())
    y_combine = list(secret_dic_final.values())
    decrypted_result = []
    for layer in range(len(y_combine[0])):
        list_x = []
        list_y = []
        for i in range(m):  # (len(x_combine)):
            y = y_combine[i][layer]
            x = np.ones(y.shape) * x_combine[i]
            list_x.append(x)
            list_y.append(y)

        all_x_layer = np.array(list_x).T
        all_y_layer = np.array(list_y).T

        decrypted_result.append(
            np.round(
                [generate_secret_shamir(all_x_layer[i], all_y_layer[i], m) for i in range(len(all_x_layer))],
                4).reshape(secret_shape[layer]) / len(x_combine)
        )

    return decrypted_result

def aggregate_shamir(secret_list, secret_shape, m):
    """
    Aggregate Shamir shares and reconstruct secret using PyTorch
    
    :param secret_list: list of shares of each client
    :param secret_shape: list of shapes of the layers  
    :param m: number of shares to use for the reconstruction
    :return: list of reconstructed PyTorch tensors
    """
    # Combine shares from all clients
    secret_dic_final = combine_shares_node(secret_list)
    
    # Convert to format suitable for PyTorchSecretSharing
    x_points = list(secret_dic_final.keys())
    y_values = list(secret_dic_final.values())
    
    if not x_points or not y_values:
        return [torch.zeros(shape) for shape in secret_shape]
    
    # Convert back to state dict format for reconstruction
    shares = []
    for i in range(min(m, len(x_points))):
        share = {}
        for layer_idx in range(len(secret_shape)):
            if i < len(y_values) and layer_idx < len(y_values[i]):
                share[f'layer_{layer_idx}'] = y_values[i][layer_idx]
            else:
                # Create zero tensor if missing
                shape = secret_shape[layer_idx]
                device = y_values[0][0].device if y_values and len(y_values[0]) > 0 else torch.device('cpu')
                share[f'layer_{layer_idx}'] = torch.zeros(shape, device=device)
        shares.append(share)
    
    # Reconstruct using PyTorchSecretSharing
    ss = PyTorchSecretSharing(method="shamir", threshold=m)
    reconstructed_dict = ss.reconstruct_secret(shares)
    
    # Convert back to list format
    return [reconstructed_dict.get(f'layer_{i}', torch.zeros(secret_shape[i])) 
            for i in range(len(secret_shape))]