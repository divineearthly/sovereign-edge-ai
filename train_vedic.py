"""
Training script for Vedic Transformer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from vedic_transformer import VedicTransformer

def train_vedic_model(
    model: VedicTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 3e-4,
    device: str = 'cuda'
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids)
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = criterion(
                shift_logits.view(-1, model.vocab_size),
                shift_labels.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                val_loss += criterion(
                    shift_logits.view(-1, model.vocab_size),
                    shift_labels.view(-1)
                ).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        scheduler.step()
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'vedic_transformer_epoch{epoch}.pt')
    
    return model


if __name__ == "__main__":
    # Model config for 300M parameter Vedic model
    config = {
        'vocab_size': 32000,
        'dim': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'dropout': 0.1,
        'max_seq': 2048
    }
    
    model = VedicTransformer(**config)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Expected: ~300M parameters
    # 24 layers × (1024² × 4) ≈ 100M attention
    # 24 layers × (1024² × 5 × 2) ≈ 250M FFN
    # Total: ~350M with embedding and output
