import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
import random
import numpy as np
from PIL import Image


# Set random seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()


# Define the U-Net Generator
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, bn=True, dropout=False, act="relu"):
        super(UNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode='reflect')
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        )

        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leaky":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Tanh()

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x))))


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super(UNetGenerator, self).__init__()

        # Initial downsampling block without BatchNorm
        self.initial_down = UNetBlock(in_channels, features, down=True, bn=False, act="leaky")

        # Downsampling part
        self.down1 = UNetBlock(features, features * 2, down=True, bn=True, act="leaky")
        self.down2 = UNetBlock(features * 2, features * 4, down=True, bn=True, act="leaky")
        self.down3 = UNetBlock(features * 4, features * 8, down=True, bn=True, act="leaky")
        self.down4 = UNetBlock(features * 8, features * 8, down=True, bn=True, act="leaky")
        # self.down5 = UNetBlock(features * 8, features * 8, down=True, bn=True, act="leaky")
        # self.down6 = UNetBlock(features * 8, features * 8, down=True, bn=True, act="leaky")

        # Bottleneck
        self.bottleneck = UNetBlock(features * 8, features * 8, down=True, bn=False, act="leaky")

        # # concate layers
        # self.concat_layer1 = nn.conv2(features * 8, features * 8, 4, 1, 1)
        # self.concat_layer2 = nn.conv2(features * 8, features * 4, 4, 1, 1)
        # self.concat_layer3 = nn.conv2(features * 4, features * 4, 4, 1, 1)

        # Upsampling part
        self.up1 = UNetBlock(features * 8, features * 8, down=False, bn=True, dropout=True, act="relu")
        # self.up2 = UNetBlock(features * 8 * 2, features * 8, down=False, bn=True, dropout=True, act="relu")
        # self.up3 = UNetBlock(features * 8 * 2, features * 8, down=False, bn=True, dropout=True, act="relu")
        self.up4 = UNetBlock(features * 8 * 2, features * 8, down=False, bn=True, dropout=False, act="relu")
        self.up5 = UNetBlock(features * 8 * 2, features * 4, down=False, bn=True, dropout=False, act="relu")
        self.up6 = UNetBlock(features * 4 * 2, features * 2, down=False, bn=True, dropout=False, act="relu")
        self.up7 = UNetBlock(features * 2 * 2, features, down=False, bn=True, dropout=False, act="relu")

        
        # Upsampling part (modified)
        # self.up1 = UNetBlock(features * 4, features * 4, down=False, bn=True, dropout=True, act="relu")
        # # self.up2 = UNetBlock(features * 8 * 2, features * 8, down=False, bn=True, dropout=True, act="relu")
        # # self.up3 = UNetBlock(features * 8 * 2, features * 8, down=False, bn=True, dropout=True, act="relu")
        # self.up4 = UNetBlock(features * 4 * 2, features * 4, down=False, bn=True, dropout=False, act="relu")
        # self.up5 = UNetBlock(features * 4 * 2, features * 4, down=False, bn=True, dropout=False, act="relu")
        # self.up6 = UNetBlock(features * 4 * 2, features * 2, down=False, bn=True, dropout=False, act="relu")
        # self.up7 = UNetBlock(features * 2 * 2, features, down=False, bn=True, dropout=False, act="relu")

        # Final upsampling to output
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Down path with skip connections
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        # d6 = self.down5(d5)
        # d7 = self.down6(d6)
        bottleneck = self.bottleneck(d5)

        # Up path with skip connections
        up1 = self.up1(bottleneck)
        # print(f"up1 shape: {up1.shape}, d5 shape: {d5.shape}")
        up2 = self.up4(torch.cat([up1, d5], dim=1))

        # up3 = self.up3(torch.cat([up2, d6], dim=1))
        # up4 = self.up4(torch.cat([up3, d5], dim=1))
        up5 = self.up5(torch.cat([up2, d4], dim=1))
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        up7 = self.up7(torch.cat([up6, d2], dim=1))

        return self.final_up(torch.cat([up7, d1], dim=1))


# Define the PatchGAN Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=6+3, features=64):
        super(PatchGANDiscriminator, self).__init__()

        # Initial layer without BatchNorm
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Downsampling layers
        self.down1 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False,
                      padding_mode="reflect"),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1, bias=False,
                      padding_mode="reflect"),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Output layer for patch classification
        self.output = nn.Sequential(
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

    def forward(self, x, y):
        # x: input image, y: target image or generated image
        # Concatenate input and target along channel dimension
        combined = torch.cat([x, y], dim=1)

        # Forward pass
        x = self.initial(combined)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        # Return raw logits (no sigmoid, will use BCEWithLogitsLoss)
        return self.output(x)


# Pix2Pix Model combining Generator and Discriminator
class Pix2Pix:
    def __init__(self, in_channels=6, gen_features=64, disc_features=64, lr=0.0002, beta1=0.5, beta2=0.999,
                 lambda_L1=100):
        # Initialize models
        self.generator = UNetGenerator(in_channels, gen_features)
        self.discriminator = PatchGANDiscriminator(in_channels+3, disc_features)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Loss functions
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.lambda_L1 = lambda_L1

        # Optimizers
        # self.optim_G = torch.optim.SGD(self.generator.parameters(), lr=0.001)
        self.optim_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    def train_step(self, input_img, target_img):
        # Move inputs to device
        input_img = input_img.to(self.device)
        target_img = target_img.to(self.device)

        # Generate fake images
        fake_img = self.generator(input_img)

        # Train Discriminator
        # Real images
        disc_real = self.discriminator(input_img, target_img)
        real_label = torch.ones_like(disc_real)
        loss_d_real = self.criterion_gan(disc_real, real_label)

        # Fake images
        disc_fake = self.discriminator(input_img, fake_img.detach())
        fake_label = torch.zeros_like(disc_fake)
        loss_d_fake = self.criterion_gan(disc_fake, fake_label)

        # Total discriminator loss
        loss_d = (loss_d_real + loss_d_fake) * 0.5

        # Update discriminator
        self.optim_D.zero_grad()
        loss_d.backward()
        self.optim_D.step()

        # Train Generator
        # Try to fool the discriminator
        disc_fake_for_gen = self.discriminator(input_img, fake_img)
        loss_g_gan = self.criterion_gan(disc_fake_for_gen, real_label)

        # L1 loss between generated and target
        loss_g_l1 = self.criterion_l1(fake_img, target_img) * self.lambda_L1

        # Total generator loss
        loss_g = loss_g_gan + loss_g_l1

        # Update generator
        self.optim_G.zero_grad()
        loss_g.backward()
        self.optim_G.step()

        return {
            'loss_d': loss_d.item(),
            'loss_g_gan': loss_g_gan.item(),
            'loss_g_l1': loss_g_l1.item(),
            'loss_g': loss_g.item()
        }

    def save_models(self, path):
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D': self.optim_D.state_dict()
        }, path)

    def load_models(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optim_G.load_state_dict(checkpoint['optim_G'])
        self.optim_D.load_state_dict(checkpoint['optim_D'])


# Dataset for paired images
class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        self.bg_dir = os.path.join(root_dir, 'backgrounds')
        self.people_dir = os.path.join(root_dir, 'people')
        self.img_dir = os.path.join(root_dir, 'images')
        # self.bg_files = os.listdir(os.path.join(bg_dir, mode))
        # self.people_files = os.listdir(os.path.join(people_dir, mode))

    def __len__(self):
        return len([f for f in os.listdir(self.bg_dir) if f.endswith('.png')])

    def __getitem__(self, idx):
        bg_files = [f for f in os.listdir(self.bg_dir) if f.endswith('.png')]
        name = bg_files[idx]

        bg_path = os.path.join(os.path.join(self.bg_dir, name))
        people_path = os.path.join(os.path.join(self.people_dir, name))
        img_path = os.path.join(os.path.join(self.img_dir, name))

        # Load the paired image (assumed to be side-by-side)
        bg = Image.open(bg_path).convert('RGB')
        people = Image.open(people_path).convert('RGB')
        img = Image.open(img_path).convert('RGB')

        input_transformed = None
        img_transformed = None
        # Apply transformations
        if self.transform:
            input_transformed = torch.cat([self.transform(bg), self.transform(people)], dim=0)
            img_transformed = self.transform(img)

        return input_transformed, img_transformed


# Training function
def train_pix2pix(model, train_loader, val_loader=None, num_epochs=100, save_path='checkpoints'):
    os.makedirs(save_path, exist_ok=True)

    writer = SummaryWriter('tf-logs')
    
    for epoch in range(num_epochs):
        model.generator.train()
        model.discriminator.train()

        epoch_losses = {
            'loss_d': 0.0,
            'loss_g_gan': 0.0,
            'loss_g_l1': 0.0,
            'loss_g': 0.0
        }

        for i, (input_img, target_img) in enumerate(train_loader):
            losses = model.train_step(input_img, target_img)

            # Update running losses
            for k, v in losses.items():
                epoch_losses[k] += v

            if i % 20 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i}/{len(train_loader)}], "
                      f"D_loss: {losses['loss_d']:.4f}, G_loss: {losses['loss_g']:.4f}")
    
        # Print epoch stats
        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
            writer.add_scalar(k, epoch_losses[k], epoch)
        
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"D_loss: {epoch_losses['loss_d']:.4f}, "
              f"G_GAN_loss: {epoch_losses['loss_g_gan']:.4f}, "
              f"G_L1_loss: {epoch_losses['loss_g_l1']:.4f}, "
              f"G_total_loss: {epoch_losses['loss_g']:.4f}")

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            model.save_models(os.path.join(save_path, f'pix2pix_epoch_{epoch + 1}.pth'))

    writer.close()

        # # Validation
        # if val_loader is not None:
        #     model.generator.eval()
        #     val_l1_loss = 0.0
        #
        #     with torch.no_grad():
        #         for input_img, target_img in val_loader:
        #             input_img = input_img.to(model.device)
        #             target_img = target_img.to(model.device)
        #             fake_img = model.generator(input_img)
        #             val_l1_loss += F.l1_loss(fake_img, target_img).item()
        #
        #     val_l1_loss /= len(val_loader)
        #     print(f"Validation L1 Loss: {val_l1_loss:.4f}")



