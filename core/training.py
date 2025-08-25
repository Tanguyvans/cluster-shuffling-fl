import numpy as np
import torch
import os
from tqdm.auto import tqdm


def train(node_id, model, train_loader, val_loader, epochs, loss_fn, optimizer, scheduler=None, device="cpu",
          dp=False, delta=1e-5, max_physical_batch_size=256, privacy_engine=None, patience=2, save_model=None, single_batch_training=False,
          capture_gradients=False):

    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "stopping_n_epoch": None}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm(range(epochs)):
        epoch_loss, epoch_acc = train_step(model, train_loader, loss_fn, optimizer, device, scheduler, privacy_engine, single_batch_training)
        tmp = ""

        val_loss, val_acc, _, _, _ = test(model, val_loader, loss_fn, device=device)

        # Print out what's happening
        print(
            f"\tNode: {node_id} \t"
            f"\tTrain Epoch: {epoch + 1} \t"
            f"Train_loss: {epoch_loss:.4f} | "
            f"Train_acc: {epoch_acc:.4f} % | "
            f"Validation_loss: {val_loss:.4f} | "
            f"Validation_acc: {val_acc:.4f} %" + tmp
        )

        # Update results dictionary
        results["train_loss"].append(epoch_loss)
        results["train_acc"].append(epoch_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model weights
            if save_model:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_model), exist_ok=True)
                
                # Save in PyTorch .pt format only
                torch.save(model.state_dict(), save_model)
                print(f"Model improved and saved to {save_model} (pytorch format)")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            results["stopping_n_epoch"] = epoch + 1
            break

    # Capture gradients if requested (similar to simple_federated.py)
    if capture_gradients:
        print(f"[Node {node_id}] Capturing gradients for gradient inversion attack evaluation...")
        model.eval()
        model.zero_grad()
        
        # Use the first batch from training data for gradient computation
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            
            # Compute gradients
            gradients = torch.autograd.grad(loss, model.parameters(), create_graph=False, retain_graph=False)
            
            # Store gradient information in results
            results["gradients"] = [grad.detach().clone().cpu() for grad in gradients]
            results["gradient_batch_images"] = x_batch.detach().clone().cpu()
            results["gradient_batch_labels"] = y_batch.detach().clone().cpu()
            results["gradient_loss"] = loss.item()
            
            # Calculate gradient norm
            grad_norms = [g.norm().item() for g in gradients]
            results["gradient_norm"] = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
            
            # Calculate accuracy for this batch
            _, predicted = outputs.max(1)
            correct = predicted.eq(y_batch).sum().item()
            results["gradient_accuracy"] = 100. * correct / len(y_batch)
            
            print(f"[Node {node_id}] Captured gradients from batch of {len(x_batch)} samples")
            print(f"[Node {node_id}] Gradient loss: {results['gradient_loss']:.4f}, accuracy: {results['gradient_accuracy']:.2f}%")
            break  # Only use first batch for gradient capture
        
        model.train()

    return results


def train_step(model, dataloader, loss_fn, optimizer, device, scheduler=None, privacy_engine=None, single_batch_training=False):
    model.train()
    accuracy = 0
    epoch_loss = 0
    total = 0
    correct = 0

    batch_count = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # 1. Forward pass
        y_pred = model(x_batch)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y_batch)
        epoch_loss += loss.item()

        # Zero the gradients before running the backward pass.
        optimizer.zero_grad()
        loss.backward()
        
        # The optimizer's step method is already modified by the privacy engine
        # if it was properly initialized
        optimizer.step()
        
        batch_count += 1
        
        # 3. Calculate accuracy
        _, predicted = torch.max(y_pred, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
        
        # For single batch training, stop after processing one batch
        if single_batch_training:
            print(f"    Single batch training: processed {len(x_batch)} samples")
            break

        # Print the current loss and accuracy
        if total > 0:
            accuracy = 100 * correct / total

    # Adjust metrics to get average loss and accuracy per batch
    epoch_loss = epoch_loss / batch_count  # Use batch_count for single batch training

    if scheduler:
        scheduler.step()

    return epoch_loss, accuracy


def test(model, dataloader, loss_fn, device="cpu"):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total = 0
    correct = 0
    y_pred = []
    y_true = []
    y_proba = []

    softmax = torch.nn.Softmax(dim=1)

    with torch.inference_mode():  # with torch.no_grad():  # Disable gradient computation
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # 1. Forward pass
            output = model(x_batch)

            # 2. Calculate and accumulate probas
            probas_output = softmax(output)
            y_proba.extend(probas_output.detach().cpu().numpy())

            # 3. Calculate and accumulate loss
            loss = loss_fn(output, y_batch)
            total_loss += loss.item() * x_batch.size(0)

            # 4. Calculate and accumulate accuracy
            _, predicted = torch.max(output, 1)  # np.argmax(output.detach().cpu().numpy(), axis=1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            y_true.extend(y_batch.detach().cpu().numpy().tolist())  # Save Truth

            y_pred.extend(predicted.detach().cpu().numpy())  # Save Prediction

    model.train()
    # Calculate average loss and accuracy
    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    return avg_loss, accuracy, y_pred, y_true, np.array(y_proba)
