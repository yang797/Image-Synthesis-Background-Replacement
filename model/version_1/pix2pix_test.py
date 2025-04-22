import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def generate_image(model, background, people):
    original_size = background.size
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transformed_bg = transform(background)
    transformed_people = transform(people)
    
    input_image = torch.cat([transformed_bg, transformed_people], dim=0)
    
    model.generator.eval()
    with torch.no_grad():
        input_tensor = input_image.unsqueeze(0).to(model.device)
        output = model.generator(input_tensor)

        # # Denormalize
        # output = output * 0.5 + 0.5
        # output = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        # output = np.clip(output, 0, 1)
        # return output
        
        # Convert tensor to PIL Image
        output = output.squeeze(0)  # Remove batch dimension [1, C, H, W] -> [C, H, W]
        output = output * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
        output = transforms.ToPILImage()(output.cpu())  # Direct conversion to PIL Image
        output = output.resize(original_size)
        
        return output
