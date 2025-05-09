import matplotlib.pyplot as plt
import os
from PIL import Image


def visualize_image(base_dir, composite_dir, save_dir=None, mode='val'):
    """
        Visualize composited images, people, background, and (optionally) real images.
        
        Args:
            base_dir: Directory containing input backgrounds, input people, ground truth images (if have).
            composite_dir: Directory containing composited images.
            save_dir: Directory to save visualizations (if None, just display).
            mode: 'val' (show real image) or 'test' (no real image).
    """
    name = [f for f in os.listdir(composite_dir) if f.endswith('png')]

    # under validation mode, output 4 rows :  composited images, people, background, real img 
    # under test mode (no real labels), output 3 rows :  composited images, people, background
    for f in name:
        com_img = Image.open(os.path.join(composite_dir, f))
        people = Image.open(os.path.join(base_dir, 'people', f))
        bg = Image.open(os.path.join(base_dir, 'backgrounds', f))
        
        if mode == 'val':
            real_img = Image.open(os.path.join(base_dir, 'images', f)))
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))  # 1行4列
            
            axes[3].imshow(real_img)
            axes[3].axis('off')
            
        elif mode == 'test':
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1行3列
        else:
            print('Wrong mode input!!')
    
        
        axes[0].imshow(com_img)
        axes[0].axis('off')
        
        axes[1].imshow(people)
        axes[1].axis('off')
        
        axes[2].imshow(bg)
        axes[2].axis('off')
    
        plt.tight_layout()
    
        if save_dir != None:
            os.makedirs(save_dir, exist_ok=True)
            plt.figsave(os.path.join(save_dir, f))
    
        plt.show()