import torch
import torch.nn as nn
import torch.nn.functional as F

class SRMConv(nn.Module):
    """
    Spatial Rich Model (SRM) Filter:
    Bộ lọc pháp y số (Forensic Filter) trích xuất nhiễu nén và vết cắt viền tần số cao.
    Giúp mô hình bắt được vết lem ghép nén H.264 mà mắt thường và mạng RGB thông thường bỏ qua.
    """
    def __init__(self):
        super().__init__()
        # 3 bộ lọc SRM kinh điển trong Steganalysis & Deepfake Detection
        filter1 = [[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]]
                   
        filter2 = [[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]
                   
        filter3 = [[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, -4, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0]]
                   
        filters = torch.tensor([filter1, filter2, filter3], dtype=torch.float32).unsqueeze(1) # (3, 1, 5, 5)
        # Lặp lại cho 3 kênh RGB (3 filter x 3 kênh = 3 kênh đầu ra sau tổng hợp)
        weights = filters.repeat(1, 3, 1, 1) / 12.0
        
        self.conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv.weight = nn.Parameter(weights, requires_grad=False) # Cố định trọng số lọc tần số

    def forward(self, x):
        return self.conv(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Module để điều phối trọng số giữa luồng RGB và Tần số"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class FreqMesoNet(nn.Module):
    """
    FreqMesoNet: Kiến trúc đề xuất mới (Proposed Model)
    Kết hợp 2 luồng:
    1. Spatial Stream (Không gian màu RGB thông thường)
    2. High-Frequency Stream (Miền tần số cao qua bộ lọc SRM)
    Hòa trộn bằng Channel Attention (SE Fusion)
    
    Tổng tham số: ~0.35M params (Siêu nhẹ, thiết kế riêng cho Web Extension)
    """
    def __init__(self, num_classes=1):
        super(FreqMesoNet, self).__init__()
        self.srm = SRMConv()

        # Nhánh 1: Không gian (Spatial Stream)
        self.spat_stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.spat_block1 = DepthwiseSeparableConv(16, 32, stride=2)
        self.spat_block2 = DepthwiseSeparableConv(32, 64, stride=2)
        self.spat_block3 = DepthwiseSeparableConv(64, 128, stride=2)

        # Nhánh 2: Tần số cao (High-Frequency Stream)
        self.freq_stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.freq_block1 = DepthwiseSeparableConv(16, 32, stride=2)
        self.freq_block2 = DepthwiseSeparableConv(32, 64, stride=2)
        self.freq_block3 = DepthwiseSeparableConv(64, 128, stride=2)

        # Fusion: Hợp nhất 2 luồng (128 + 128 = 256 channels)
        self.ca = ChannelAttention(256)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1. Trích xuất không gian
        f_spat = self.spat_stem(x)
        f_spat = self.spat_block1(f_spat)
        f_spat = self.spat_block2(f_spat)
        f_spat = self.spat_block3(f_spat)

        # 2. Trích xuất tần số cao qua SRM
        freq_in = self.srm(x)
        f_freq = self.freq_stem(freq_in)
        f_freq = self.freq_block1(f_freq)
        f_freq = self.freq_block2(f_freq)
        f_freq = self.freq_block3(f_freq)

        # 3. Hòa trộn đặc trưng bằng Attention
        fused = torch.cat([f_spat, f_freq], dim=1) # (B, 256, H/16, W/16)
        fused = self.ca(fused)
        fused = self.fuse_conv(fused)

        # 4. Phân loại
        out = self.pool(fused).view(fused.size(0), -1)
        out = self.dropout(out)
        return self.fc(out)
