"""
Network architectures for Bi-Modality Medical Image Synthesis.
Ported from TensorFlow 1.x (semi/net.py) to PyTorch.

Architecture overview:
  - Encoder: maps 64x64 grayscale image -> 128-dim latent vector
  - GeneratorADC: 128-dim -> 64x64 ADC image (with shared layers)
  - GeneratorT2: 64x64 ADC -> 64x64 T2w image (U-Net with shared layers + skip connections)
  - DiscriminatorADC / DiscriminatorT2: 64x64 image -> scalar (WGAN critic)
  - SharedLayers: first two deconv layers shared between ADC generator and T2 translator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedLayers(nn.Module):
    """
    Shared transposed-conv layers between Generator_ADC and Generator_T2.
    In the original paper, the first two deconv layers share weights to ensure
    that spatial layouts of corresponding images in two modalities are consistent.
    """
    def __init__(self):
        super(SharedLayers, self).__init__()
        # dcov1: [batch, 512, 2, 2] => [batch, 512, 4, 4]
        self.dcov1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        # dcov2: [batch, 512, 4, 4] => [batch, 256, 8, 8]
        self.dcov2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

    def forward_dcov1(self, x):
        return F.relu(self.bn1(self.dcov1(x)))

    def forward_dcov2(self, x):
        return F.relu(self.bn2(self.dcov2(x)))


class Encoder(nn.Module):
    """
    Encoder: maps a 64x64 grayscale image to a 128-dim latent vector.
    Architecture: conv(64) -> conv(128) -> flatten -> FC(1024) -> FC(128)
    """
    def __init__(self, z_dim=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)   # -> 32x32
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # -> 16x16
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, z_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation on latent code
        return x


class GeneratorADC(nn.Module):
    """
    Generator for ADC images: z (128-dim) -> 64x64 ADC image.
    Architecture: FC(128) -> FC(1024) -> FC(2*2*512) -> reshape ->
                  SharedLayer.dcov1 -> SharedLayer.dcov2 ->
                  dcov3 (128) -> depth_to_space(2) -> dcov4 (1, tanh)
    """
    def __init__(self, shared_layers, z_dim=128):
        super(GeneratorADC, self).__init__()
        self.shared = shared_layers

        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 1024)
        self.fc3 = nn.Linear(1024, 2 * 2 * 512)

        # dcov3: [batch, 256, 8, 8] => [batch, 128, 16, 16]
        self.dcov3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # After depth_to_space(2): [batch, 128, 16, 16] => [batch, 32, 32, 32]
        # dcov4: [batch, 32, 32, 32] => [batch, 1, 64, 64]
        self.dcov4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 512, 2, 2)

        # Shared layers
        x = self.shared.forward_dcov1(x)   # -> [B, 512, 4, 4]
        x = self.shared.forward_dcov2(x)   # -> [B, 256, 8, 8]

        # Private layers
        x = F.relu(self.bn3(self.dcov3(x)))  # -> [B, 128, 16, 16]

        # depth_to_space with block_size=2: rearranges channels to spatial
        # [B, 128, 16, 16] -> [B, 32, 32, 32]
        x = F.pixel_shuffle(x, 2)

        x = torch.tanh(self.dcov4(x))       # -> [B, 1, 64, 64]
        return x


class GeneratorT2(nn.Module):
    """
    Generator/Translator for T2w images: ADC image (64x64) -> T2w image (64x64).
    Architecture: U-Net encoder-decoder with skip connections.
    Encoder shares the first two deconv layers with GeneratorADC.

    Encoder:
      layer1: [B,1,64,64] -> [B,64,32,32]
      layer2: [B,64,32,32] -> [B,128,16,16]
      layer3: [B,128,16,16] -> [B,256,8,8]
      layer4: [B,256,8,8] -> [B,512,4,4]
      layer5: [B,512,4,4] -> [B,512,2,2]

    Decoder (with shared layers + skip connections):
      SharedLayer.dcov1: [B,512,2,2] -> [B,512,4,4]
      SharedLayer.dcov2: [B,512,4,4] -> [B,256,8,8] (+ private dcov2 from layer4, summed)
      dcov3: concat with layer3 -> [B,128,16,16]
      dcov4: concat with layer2 -> [B,64,32,32]
      dcov5: concat with layer1 -> [B,1,64,64] (tanh)
    """
    def __init__(self, shared_layers):
        super(GeneratorT2, self).__init__()
        self.shared = shared_layers
        c = 64

        # Encoder layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 8), nn.ReLU(inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(c * 8, c * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 8), nn.ReLU(inplace=True)
        )

        # Private decoder layer (parallel to shared dcov2, outputs are summed)
        self.private_dcov2 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1)
        self.private_dcov2_bn = nn.BatchNorm2d(c * 4)

        # dcov3: input is concat of [decoder_output, enc3] = [256+256, 8, 8] -> [128, 16, 16]
        self.dcov3 = nn.ConvTranspose2d(c * 4 + c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.dcov3_bn = nn.BatchNorm2d(c * 2)

        # dcov4: input is concat of [decoder_output, enc2] = [128+128, 16, 16] -> [64, 32, 32]
        self.dcov4 = nn.ConvTranspose2d(c * 2 + c * 2, c, kernel_size=4, stride=2, padding=1)
        self.dcov4_bn = nn.BatchNorm2d(c)

        # dcov5: input is concat of [decoder_output, enc1] = [64+64, 32, 32] -> [1, 64, 64]
        self.dcov5 = nn.ConvTranspose2d(c + c, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # [B, 64, 32, 32]
        e2 = self.enc2(e1)   # [B, 128, 16, 16]
        e3 = self.enc3(e2)   # [B, 256, 8, 8]
        e4 = self.enc4(e3)   # [B, 512, 4, 4]
        e5 = self.enc5(e4)   # [B, 512, 2, 2]

        # Decoder with shared layers
        d1 = self.shared.forward_dcov1(e5)     # [B, 512, 4, 4]

        # Shared dcov2 + private dcov2 (sum, as in original code: output_1 + output_2)
        shared_out = self.shared.forward_dcov2(d1)  # [B, 256, 8, 8]
        private_out = F.relu(self.private_dcov2_bn(self.private_dcov2(e4)))  # [B, 256, 8, 8]
        d2 = shared_out + private_out               # [B, 256, 8, 8]

        # Skip connections
        d3 = F.relu(self.dcov3_bn(self.dcov3(torch.cat([d2, e3], dim=1))))   # [B, 128, 16, 16]
        d4 = F.relu(self.dcov4_bn(self.dcov4(torch.cat([d3, e2], dim=1))))   # [B, 64, 32, 32]
        d5 = torch.tanh(self.dcov5(torch.cat([d4, e1], dim=1)))              # [B, 1, 64, 64]

        return d5


class DiscriminatorADC(nn.Module):
    """
    WGAN critic for ADC images.
    Architecture: conv(64) -> conv(128) -> flatten -> FC(1024) -> FC(128) -> FC(1)
    No batch norm (as per WGAN-GP guidelines).
    """
    def __init__(self):
        super(DiscriminatorADC, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation (WGAN critic)
        return x


class DiscriminatorT2(nn.Module):
    """
    WGAN critic for T2w images.
    Same architecture as DiscriminatorADC but separate weights.
    """
    def __init__(self):
        super(DiscriminatorT2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation (WGAN critic)
        return x


class DiscriminatorZ(nn.Module):
    """
    Discriminator for latent z space (used in supervised training only).
    Distinguishes between z ~ N(0,1) and encoder output z = Enc(x).
    Architecture: FC(128) -> FC(1024) -> FC(1)
    """
    def __init__(self, z_dim=128):
        super(DiscriminatorZ, self).__init__()
        self.fc1 = nn.Linear(z_dim, 128)
        self.fc2 = nn.Linear(128, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = self.fc3(z)  # No activation (sigmoid applied in loss)
        return z
