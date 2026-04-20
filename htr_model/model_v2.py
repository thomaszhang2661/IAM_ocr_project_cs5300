"""
CRNN v2: VGG_FeatureExtractor + BiGRU encoder.

Based on the reference architecture provided:
  - VGG_FeatureExtractor: same 7-layer CNN backbone
  - Encoder: BiGRU (gru type from config)
  - Decoder: Linear projection to charset_size
  - Height: supports H=32 and H=64 via dynamic last conv kernel

Key difference from model_v1 (model.py):
  - RNN: GRU instead of LSTM
  - height collapse: mean(dim=2) instead of squeeze(2) — tolerant to h>1
  - output permutation matches reference CRNN forward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGFeatureExtractor(nn.Module):
    """
    VGG-style CNN backbone.
    Identical to the reference VGG_FeatureExtractor, with H=64 support:
      ks[6] = img_h // 16  (=2 for H=32, =4 for H=64)
    """

    def __init__(self, input_channel=1, output_channel=512, img_h=64):
        super().__init__()
        assert img_h in (32, 64), f'Supports img_h=32 or 64, got {img_h}'

        last_kernel_h = img_h // 16  # 32->2, 64->4
        ks = [3, 3, 3, 3, 3, 3, last_kernel_h]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, output_channel]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn  = input_channel if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            cnn.add_module(f'relu{i}', nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling0', nn.MaxPool2d(2, 2))
        convRelu(1)
        cnn.add_module('pooling1', nn.MaxPool2d(2, 2))
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling2', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling3', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(6, True)

        self.ConvNet = cnn

    def forward(self, x):
        return self.ConvNet(x)   # (N, C, 1, W')


class Encoder(nn.Module):
    """BiGRU encoder (matches reference Encoder with type='gru')."""

    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=True, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size, hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x):
        # x: (N, T, input_size)
        output, hidden = self.rnn(x)   # output: (N, T, hidden*2)
        return output, hidden


class CRNN_v2(nn.Module):
    """
    CRNN v2: VGGFeatureExtractor + BiGRU + Linear decoder.
    Mirrors the reference CRNN forward (gru path).

    Output: (T, N, num_classes) with log_softmax — ready for CTCLoss.
    """

    def __init__(self, img_h=64, num_channels=1, num_classes=None,
                 encoder_input_size=512, encoder_hidden_size=256,
                 encoder_layers=2, encoder_bidirectional=True, dropout=0.1):
        super().__init__()
        assert num_classes is not None

        self.cnn = VGGFeatureExtractor(
            input_channel=num_channels,
            output_channel=encoder_input_size,
            img_h=img_h,
        )

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = Encoder(
            input_size=encoder_input_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_layers,
            bidirectional=encoder_bidirectional,
            dropout=dropout,
        )

        decoder_in = encoder_hidden_size * 2 if encoder_bidirectional else encoder_hidden_size
        self.decoder = nn.Linear(decoder_in, num_classes)

    def forward(self, x):
        """
        x: (N, C, H, W)
        returns: (T, N, num_classes) with log_softmax
        """
        # CNN
        feat = self.cnn(x)                       # (N, 512, h, W')
        # collapse height by mean (= squeeze when h=1, robust when h>1)
        feat = feat.mean(dim=2)                  # (N, 512, W')
        feat = feat.permute(0, 2, 1)             # (N, W', 512)  [batch_first]
        feat = self.dropout(feat)                # dropout after CNN

        # BiGRU
        output, _ = self.encoder(feat)           # (N, W', hidden*2)
        output = self.decoder(output)            # (N, W', num_classes)

        # Permute to (T, N, C) for CTC
        output = output.permute(1, 0, 2)         # (T, N, num_classes)
        return F.log_softmax(output, dim=2)


# IAM alphabet (shared with model.py)
IAM_ALPHABET = (
    ' !"#&\'()*+,-./0123456789:;?'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abcdefghijklmnopqrstuvwxyz'
)


def build_model_v2(alphabet: str, img_h: int = 64, hidden_size: int = 256, dropout: float = 0.1) -> CRNN_v2:
    num_classes = len(alphabet) + 1   # +1 for CTC blank (index 0)
    return CRNN_v2(
        img_h=img_h,
        num_channels=1,
        num_classes=num_classes,
        encoder_input_size=512,
        encoder_hidden_size=hidden_size,
        encoder_layers=2,
        encoder_bidirectional=True,
        dropout=dropout,
    )


if __name__ == '__main__':
    model = build_model_v2(IAM_ALPHABET)
    total = sum(p.numel() for p in model.parameters())
    print(f'Model v2 parameters: {total:,}')
    x = torch.randn(2, 1, 64, 400)
    out = model(x)
    print(f'Input: {x.shape}  Output: {out.shape}')
