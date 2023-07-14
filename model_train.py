from tqdm import tqdm
from utils import GetCorrectPredCount
import torch

# Data to plot accuracy and loss graphs
train_losses = []
train_acc = []

########################################################################################################################################################
def model_train(model, device, train_loader, optimizer, criterion, scheduler, path):
    """
        Training method
    """
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate Loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        scheduler.step()
        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
    torch.save(model.state_dict(), path)
########################################################################################################################################################

def get_lr(optimizer):
    """
        For tracking how the learning rate is changing throughout training
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]