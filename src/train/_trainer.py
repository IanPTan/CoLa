import torch
from tqdm import tqdm

def train_model(model, optimizer, criterion, train_loader, val_loader, num_epochs, device='cpu', logging=True):
    """
    Standard training loop for a PyTorch model.
    
    Args:
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        num_epochs (int): Number of epochs to train.
        device (str): Device to train on ('cpu' or 'cuda').
        logging (bool): If True, use tqdm and print epoch summaries.
        
    Returns:
        train_losses (Tensor): Tensor of average training losses per epoch.
        val_losses (Tensor): Tensor of average validation losses per epoch.
    """
    train_losses = []
    val_losses = []
    
    model.to(device)
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") if logging else train_loader
        
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Flatten images if they are 4D (B, C, H, W) for MLP compatibility
            if len(images.shape) == 4:
                images = images.view(images.size(0), -1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") if logging else val_loader
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                if len(images.shape) == 4:
                    images = images.view(images.size(0), -1)
                    
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if logging:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
    return torch.tensor(train_losses), torch.tensor(val_losses)
