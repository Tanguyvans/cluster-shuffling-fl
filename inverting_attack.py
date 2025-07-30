import breaching
import torch
import logging, sys
import matplotlib.pyplot as plt
import os

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()

cfg = breaching.get_config(overrides=["case=4_fedavg_small_scale", "case/data=CIFAR10"])
          
device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
setup

cfg.case.data.partition="random"
cfg.case.user.user_idx = 1
cfg.case.model='resnet18'

cfg.case.user.provide_labels = True

# These settings govern the total amount of user data and how it is used over multiple local update steps:
cfg.case.user.num_data_points = 4
cfg.case.user.num_local_updates = 4
cfg.case.user.num_data_per_local_update_step = 2

# Total variation regularization needs to be smaller on CIFAR-10:
cfg.attack.regularization.total_variation.scale = 1e-3

user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)

server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)

reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)

metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload], 
                                    server.model, order_batch=True, compute_full_iip=False, 
                                    cfg_case=cfg.case, setup=setup)

print(metrics)

def save_reconstructed_images(user_data, save_dir='reconstructed_images'):
    """Save reconstructed images individually."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get dataset stats for denormalization (CIFAR-10 values)
    dm = torch.as_tensor([0.4914, 0.4822, 0.4465])[None, :, None, None]  # CIFAR-10 mean
    ds = torch.as_tensor([0.2023, 0.1994, 0.2010])[None, :, None, None]  # CIFAR-10 std
    
    data = user_data["data"].clone().detach()
    labels = user_data["labels"].clone().detach() if user_data["labels"] is not None else None
    
    # Denormalize
    data.mul_(ds).add_(dm).clamp_(0, 1)
    
    # Save each image
    for i, img in enumerate(data):
        img_np = img.permute(1, 2, 0).cpu().numpy()
        label = labels[i].item() if labels is not None else 'unknown'
        
        plt.figure(figsize=(8, 8))
        plt.imshow(img_np)
        plt.axis('off')
        plt.title(f'Reconstructed Image {i} (Label: {label})')
        plt.savefig(f'{save_dir}/image_{i}_label_{label}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Individual images saved in '{save_dir}' directory")

user.plot(reconstructed_user_data)
plt.savefig('reconstructed_images.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory
print("Images saved as 'reconstructed_images.png'")

# Also save individual images
save_reconstructed_images(reconstructed_user_data)