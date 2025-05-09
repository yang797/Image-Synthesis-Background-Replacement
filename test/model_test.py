import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def generate_image(model, size=512, mask, background, people):
    original_size = background.size
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    m_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transformed_bg = transform(background)
    transformed_people = transform(people)
    transformed_mask = m_transform(mask)
    
    # input_image = torch.cat([transformed_bg, transformed_people], dim=0)
    
    model.generator.eval()
    with torch.no_grad():
        bg_tensor = transformed_bg.unsqueeze(0).to(model.device)
        people_tensor = transformed_people.unsqueeze(0).to(model.device)
        mask_tensor = transformed_mask.unsqueeze(0).to(model.device)

        input_tensor = (mask_tensor, bg_tensor, people_tensor)
        
        output = model.generator(input_tensor)
        
        # Convert tensor to PIL Image
        output = output.squeeze(0)  # Remove batch dimension [1, C, H, W] -> [C, H, W]
        output = output * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
        output = transforms.ToPILImage()(output.cpu())  # Direct conversion to PIL Image
        output = output.resize(original_size)
        
        return output
