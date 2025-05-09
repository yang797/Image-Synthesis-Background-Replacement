import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms




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
        self.dropout = nn.Dropout(0.2) if dropout else nn.Identity()

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leaky":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.act = nn.Tanh()

    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x))))



class Generator(nn.Module):
    def __init__(self, m_in_channels=1, p_in_channels=3, bg_in_channels=3, features=64):
        super(Generator, self).__init__()


        # people down-sampling
        # Initial downsampling block without BatchNorm
        self.p_initial_down = UNetBlock(p_in_channels, features, down=True, bn=False, act="leaky")
        # Downsampling part
        self.p_down1 = UNetBlock(features, features * 2, down=True, bn=True, act="leaky")
        self.p_down2 = UNetBlock(features * 2, features * 4, down=True, bn=True, act="leaky")
        self.p_down3 = UNetBlock(features * 4, features * 8, down=True, bn=True, act="leaky")
        # self.p_down4 = UNetBlock(features * 8, features * 8, down=True, bn=True, act="leaky")       
        # people Bottleneck
        self.p_bottleneck = UNetBlock(features * 8, features * 8, down=True, bn=False, act="leaky")

            
        # background down-sampling
        # Initial downsampling block without BatchNorm
        self.bg_initial_down = UNetBlock(bg_in_channels, features, down=True, bn=False, act="leaky")
        # Downsampling part
        self.bg_down1 = UNetBlock(features, features * 2, down=True, bn=True, act="leaky")
        self.bg_down2 = UNetBlock(features * 2, features * 4, down=True, bn=True, act="leaky")
        self.bg_down3 = UNetBlock(features * 4, features * 8, down=True, bn=True, act="leaky")
        # self.bg_down4 = UNetBlock(features * 8, features * 8, down=True, bn=True, act="leaky")   
        # background Bottleneck
        self.bg_bottleneck = UNetBlock(features * 8, features * 8, down=True, bn=False, act="leaky")

        
        # mask down-sampling
        self.mask_initial_down = UNetBlock(m_in_channels, features, down=True, bn=False, act='leaky')
        self.m_down1 = UNetBlock(features, features * 2, down=True, bn=True, act="leaky")
        self.m_down2 = UNetBlock(features * 2, features * 4, down=True, bn=True, act="leaky")
        self.m_down3 = UNetBlock(features * 4, features * 8, down=True, bn=True, act="leaky")
        # self.m_down4 = UNetBlock(features * 8, features * 8, down=True, bn=True, act="leaky")       
        # mask Bottleneck
        self.m_bottleneck = UNetBlock(features * 8, features * 8, down=True, bn=False, act="leaky")
        
    
        # Upsampling part
        self.up1 = UNetBlock(features * 8 * 3, features * 8 * 2, down=False, bn=True, dropout=True, act="relu")
        self.up2 = UNetBlock(features * 8 * 3, features * 4 * 3, down=False, bn=True, dropout=True, act="relu")
        # self.up3 = UNetBlock(features * 8 * 2, features * 4, down=False, bn=True, dropout=False, act="relu")
        self.up4 = UNetBlock(features * 4 * 4, features * 3 * 2, down=False, bn=True, dropout=True, act="relu")
        self.up5 = UNetBlock(features * 4 * 2, features * 2, down=False, bn=True, dropout=False, act="relu")

        # Final upsampling to output
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 3, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, input_):
        mask, bg, people = input_
        # down path with people
        p_d1 = self.p_initial_down(people)
        p_d2 = self.p_down1(p_d1)
        p_d3 = self.p_down2(p_d2)
        p_d4 = self.p_down3(p_d3)
        # p_d5 = self.p_down4(p_d4)
        p_bottleneck = self.p_bottleneck(p_d4)


        # down path with background
        bg_d1 = self.bg_initial_down(bg)
        bg_d2 = self.bg_down1(bg_d1)
        bg_d3 = self.bg_down2(bg_d2)
        bg_d4 = self.bg_down3(bg_d3)
        # bg_d5 = self.bg_down4(bg_d4)
        bg_bottleneck = self.bg_bottleneck(bg_d4)


        # down path with mask
        m_d1 = self.mask_initial_down(mask)
        m_d2 = self.m_down1(m_d1)
        m_d3 = self.m_down2(m_d2)
        m_d4 = self.m_down3(m_d3)
        m_bottleneck = self.m_bottleneck(m_d4)

        
        # concat people and background
        bottleneck = torch.cat([bg_bottleneck, p_bottleneck, m_bottleneck], dim=1)

        
        # Up path with skip connections
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, bg_d4], dim=1))
        # up3 = self.up3(torch.cat([up2, p_d4], dim=1))
        up3 = self.up4(torch.cat([up2, p_d3], dim=1))
        up4 = self.up5(torch.cat([up3, bg_d2], dim=1))
        output = self.final_up(torch.cat([up4, p_d1], dim=1))

        return output



# Define the PatchGAN Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1+3+3+3, features=64):
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

    def forward(self, input_, composited_):
        # Concatenate input and target along channel dimension
        m, bg, people = input_
        combined = torch.cat([m, bg, people, composited_], dim=1)

        # Forward pass
        x = self.initial(combined)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        # Return raw logits (no sigmoid, will use BCEWithLogitsLoss)
        return self.output(x)


