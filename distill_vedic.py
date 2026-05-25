"""
Knowledge Distillation: Standard 1.1B model → Vedic 300M model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from vedic_transformer import VedicTransformer

def distill_vedic(
    teacher_model,  # Standard 1.1B transformer
    student_model,  # Vedic 300M transformer
    train_loader,
    epochs: int = 3,
    temperature: float = 4.0,
    alpha: float = 0.5,  # Balance between hard and soft targets
    device: str = 'cuda'
):
    """Distill knowledge from standard model to Vedic model."""
    
    teacher = teacher_model.to(device).eval()
    student = student_model.to(device).train()
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    
    for epoch in range(epochs):
        total_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            
            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_logits = teacher(input_ids) / temperature
            
            # Student forward pass
            student_logits = student(input_ids) / temperature
            
            # Shift for next-token prediction
            teacher_logits = teacher_logits[:, :-1, :].contiguous()
            student_logits = student_logits[:, :-1, :].contiguous()
            labels = batch['labels'][:, 1:].to(device).contiguous()
            
            # Hard loss (standard cross-entropy)
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1)
            )
            
            # Soft loss (KL divergence from teacher)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits.view(-1, student_logits.size(-1)), dim=-1),
                F.softmax(teacher_logits.view(-1, teacher_logits.size(-1)), dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # Combined loss
            loss = alpha * hard_loss + (1 - alpha) * soft_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()
        
        print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}, "
              f"hard={total_hard_loss/len(train_loader):.4f}, "
              f"soft={total_soft_loss/len(train_loader):.4f}")
        
        torch.save(student.state_dict(), f'vedic_distilled_epoch{epoch}.pt')
    
    return student
