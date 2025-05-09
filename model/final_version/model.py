from network_structure import Generator, PatchGANDiscriminator

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
import os


# Triple-U-net Model combining Generator and Discriminator
class Triple_U_Net:
    def __init__(self, m_in_channels=1, p_in_channels=3, bg_in_channels=3, gen_features=64, disc_features=64, lr=0.001, beta1=0.5, beta2=0.999, lambda_L1=7):
        # Initialize models
        self.generator = Generator(m_in_channels, p_in_channels, bg_in_channels, gen_features)
        self.discriminator = PatchGANDiscriminator(m_in_channels + p_in_channels + bg_in_channels + 3, disc_features)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Loss functions
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.lambda_L1 = lambda_L1

        vgg19 = models.vgg19(pretrained=True).features
        for p in vgg19.parameters():
            p.requires_grad = False
        vgg19 = vgg19.to(self.device)
        
        vgg19.eval()
        self.vgg = vgg19
        # 感知损失权重，可从 0.01～0.1 试验
        self.lambda_perc = 0.2

        # Optimizers
        self.optim_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, beta2))
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    
    def train_step(self, m_img, bg_img, people_img, target_img):
        # Move inputs to device
        m_img = m_img.to(self.device)
        bg_img = bg_img.to(self.device)
        people_img = people_img.to(self.device)
        target_img = target_img.to(self.device)

        input_img = (m_img, bg_img, people_img)

        fake_img = self.generator(input_img)

        # Train Discriminator
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
        for i in range(2):
            # Generate fake images
            fake_img = self.generator(input_img)
            
            disc_fake_for_gen = self.discriminator(input_img, fake_img.detach())
            loss_g_gan = self.criterion_gan(disc_fake_for_gen, real_label)

            # L1 loss between generated and target
            loss_g_l1 = self.criterion_l1(fake_img, target_img) * self.lambda_L1

            # Perception loss
            mean = torch.tensor([0.485,0.456,0.406], device=self.device).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225], device=self.device).view(1,3,1,1)
            fb = (fake_img + 1)/2;  fb = (fb - mean)/std
            rb = (target_img + 1)/2;  rb = (rb - mean)/std
            feat_fake = self.vgg(fb)
            feat_real = self.vgg(rb)
            loss_G_perc = F.l1_loss(feat_fake, feat_real) * self.lambda_perc

            # Total generator loss
            # loss_g = loss_g_gan + loss_g_l1 
            loss_g = loss_g_gan + loss_g_l1 + loss_G_perc

            # Update generator
            self.optim_G.zero_grad()
            loss_g.backward()
            self.optim_G.step() 


        return {
            'loss_d': loss_d.item(),
            'loss_g_gan': loss_g_gan.item(),
            'loss_g_l1': loss_g_l1.item(),
            'loss_g': loss_g.item(),
            'loss_G_perc': loss_G_perc
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
