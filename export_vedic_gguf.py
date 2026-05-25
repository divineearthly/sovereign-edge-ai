"""Export trained Vedic Transformer weights for GGUF conversion."""
import torch
import numpy as np
import os
from vedic_transformer import VedicTransformer

def export_vedic_to_gguf(checkpoint_path: str, output_dir: str, 
                         vocab_size=8000, dim=512, num_layers=8, num_heads=8):
    model = VedicTransformer(vocab_size, dim, num_layers, num_heads)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    os.makedirs(output_dir, exist_ok=True)
    
    config = {"vocab_size": vocab_size, "dim": dim, 
              "num_layers": num_layers, "num_heads": num_heads}
    
    for name, param in model.named_parameters():
        data = param.detach().cpu().numpy()
        fname = name.replace(".", "_") + ".f32"
        with open(f"{output_dir}/{fname}", "wb") as f:
            f.write(data.tobytes())
    
    print(f"Exported {len(list(model.named_parameters()))} tensors to {output_dir}/")
    return config

if __name__ == "__main__":
    export_vedic_to_gguf("vedic_epoch4.pt", "vedic_gguf")
