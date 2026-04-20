"""
CNN-BiLSTM-CTC model for handwriting recognition.

Architecture (following meijieru/crnn.pytorch, adapted for IAM line images):
  - CNN backbone: VGG-style, 7 conv layers with max-pooling
  - BiLSTM: 2 layers, 256 hidden units each direction
  - CTC: built-in torch.nn.CTCLoss (no warp-ctc dependency)

Input: grayscale line image, H=32, W=variable
Output: probability sequence over alphabet characters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: (T, N, input_size)  → rnn expects (N, T, input_size) with batch_first
        x = x.permute(1, 0, 2)             # (N, T, C)
        recurrent, _ = self.rnn(x)          # (N, T, hidden*2)
        output = self.linear(recurrent)     # (N, T, num_classes)
        output = output.permute(1, 0, 2)    # (T, N, num_classes)
        return output


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for HTR.

    Args:
        img_h: input image height (must be 32)
        num_channels: 1 (gray) or 3 (RGB)
        num_classes: alphabet size + 1 (for CTC blank token)
        hidden_size: BiLSTM hidden size per direction
    """
    def __init__(self, img_h=64, num_channels=1, num_classes=None, hidden_size=256):
        super().__init__()
        assert img_h in (32, 64), f'CRNN supports img_h=32 or 64, got {img_h}'
        assert num_classes is not None

        # --- CNN backbone ---
        # H=32: 32->16->8->4->2->1  (last conv kernel h=2)
        # H=64: 64->32->16->8->4->1 (last conv kernel h=4)
        last_kernel_h = img_h // 16   # 32->2, 64->4
        ks = [3, 3, 3, 3, 3, 3, last_kernel_h]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn_layers = []
        def conv_bn_relu(i, batch_norm=False):
            in_ch  = num_channels if i == 0 else nm[i - 1]
            out_ch = nm[i]
            cnn_layers.append(nn.Conv2d(in_ch, out_ch, ks[i], ss[i], ps[i]))
            if batch_norm:
                cnn_layers.append(nn.BatchNorm2d(out_ch))
            cnn_layers.append(nn.ReLU(inplace=True))

        conv_bn_relu(0)                    # conv1: 64
        cnn_layers.append(nn.MaxPool2d(2, 2))  # H: 32->16, W: /2

        conv_bn_relu(1)                    # conv2: 128
        cnn_layers.append(nn.MaxPool2d(2, 2))  # H: 16->8, W: /2

        conv_bn_relu(2, batch_norm=True)   # conv3: 256
        conv_bn_relu(3)                    # conv4: 256
        cnn_layers.append(nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # H: 8->4, W: preserve

        conv_bn_relu(4, batch_norm=True)   # conv5: 512
        conv_bn_relu(5)                    # conv6: 512
        cnn_layers.append(nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # H: 4->2, W: preserve

        conv_bn_relu(6, batch_norm=True)   # conv7: 512, kernel 2x2, no pad → H: 2->1

        self.cnn = nn.Sequential(*cnn_layers)

        # --- RNN ---
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (N, C, H, W)  H=32, W=variable
        Returns:
            log_probs: (T, N, num_classes)  for CTC loss
        """
        # CNN features
        conv = self.cnn(x)          # (N, 512, 1, W')
        b, c, h, w = conv.size()
        assert h == 1, f'Expected h=1 after CNN, got h={h}'

        # Reshape for RNN: (W', N, C) = (T, N, input_size)
        conv = conv.squeeze(2)      # (N, 512, W')
        conv = conv.permute(2, 0, 1)  # (W', N, 512) = (T, N, 512)

        # BiLSTM
        output = self.rnn(conv)     # (T, N, num_classes)

        # Log softmax for CTC
        log_probs = F.log_softmax(output, dim=2)
        return log_probs


def build_model(alphabet: str, img_h: int = 64, hidden_size: int = 256) -> CRNN:
    """
    Build CRNN model.
    num_classes = len(alphabet) + 1  (extra for CTC blank, index 0)
    """
    num_classes = len(alphabet) + 1
    model = CRNN(
        img_h=img_h,
        num_channels=1,
        num_classes=num_classes,
        hidden_size=hidden_size,
    )
    return model


# IAM alphabet: space + printable ASCII subset found in IAM labels
IAM_ALPHABET = (
    ' !"#&\'()*+,-./0123456789:;?'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abcdefghijklmnopqrstuvwxyz'
)


if __name__ == '__main__':
    model = build_model(IAM_ALPHABET)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,}")
    # Test forward pass
    x = torch.randn(2, 1, 64, 400)
    out = model(x)
    print(f"Input: {x.shape}  Output: {out.shape}")  # (T, N, num_classes)
