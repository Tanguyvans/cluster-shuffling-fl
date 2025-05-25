import breaching
import torch
import logging, sys
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()

cfg = breaching.get_config(overrides=["case=4_fedavg_small_scale", "case/data=CIFAR10"])

# Check available CUDA devices and use the first one
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    logger.info(f"Found {num_devices} CUDA devices")
    device = torch.device(f'cuda:0')  # Use the first GPU
else:
    device = torch.device('cpu')
    logger.info("No CUDA devices found, using CPU")

torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
print(setup)

# Rest of your configuration
cfg.case.data.partition="random"
cfg.case.user.user_idx = 1
cfg.case.model='resnet18'
cfg.case.user.provide_labels = True
cfg.case.user.num_data_points = 4
cfg.case.user.num_local_updates = 4
cfg.case.user.num_data_per_local_update_step = 2
cfg.attack.regularization.total_variation.scale = 1e-3

user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)

server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)

# Plot and save original data
plt.figure(figsize=(10, 8))
user.plot(true_user_data)
plt.title('Original User Data')
plt.savefig('user_data_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Perform reconstruction
reconstructed_user_data, stats = attacker.reconstruct([server_payload], [shared_data], {}, dryrun=cfg.dryrun)

# Calculate metrics
metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload], 
                                    server.model, order_batch=True, compute_full_iip=False, 
                                    cfg_case=cfg.case, setup=setup)

# Plot and save reconstruction
plt.figure(figsize=(10, 8))
user.plot(reconstructed_user_data)
plt.title('Reconstructed User Data')
plt.savefig('reconstructed_data_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Print metrics
print("\nReconstruction Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")