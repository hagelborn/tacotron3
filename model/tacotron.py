from model.decoder import Decoder
from model.encoder import Encoder, SimpleEncoder
import model.hparams as hparams
from model.layers import ConvNorm
import torch.nn as nn
import torch
import torch.nn.functional as F

class Tacotron3(nn.Module):
    def __init__(self,activate_encoder):
        super(Tacotron3, self).__init__()
        if activate_encoder:
            self.encoder = Encoder()
        else:
            self.encoder = SimpleEncoder()
        self.decoder = Decoder(activate_encoder)
        self.postnet = Postnet()


    def forward(self,inputs):
        mel_source, mel_lengths, embedding, mel_target = inputs
        mel_lengths = mel_lengths.data

        encoder_outputs = self.encoder(mel_source, embedding)
        mel_outputs, alignments = self.decoder(encoder_outputs,mel_target,mel_lengths)

        end_padding_ind = get_reverse_mask(mel_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)

        # FIX This sollution is ugleh - is there a better way?
        mel_outputs = mel_outputs.permute(0, 2, 1)
        mel_outputs[end_padding_ind, :] = 0
        mel_outputs = mel_outputs.permute(0, 2, 1)

        mel_outputs_postnet = mel_outputs_postnet.permute(0, 2, 1)
        mel_outputs_postnet[end_padding_ind, :] = 0
        mel_outputs_postnet = mel_outputs_postnet.permute(0, 2, 1)

        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs, mel_outputs_postnet, alignments

    def inference(self,inputs):
        mel_source, mel_lengths, embedding, _ = inputs

        encoder_outputs = self.encoder(mel_source, embedding)
        mel_outputs, _ = self.decoder.inference(encoder_outputs)

        end_padding_ind = get_reverse_mask(mel_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)

        # FIX This sollution is ugleh - is there a better way?
        mel_outputs = mel_outputs.permute(0,2,1)
        mel_outputs[end_padding_ind, :] = 0
        mel_outputs = mel_outputs.permute(0,2,1)

        mel_outputs_postnet = mel_outputs_postnet.permute(0,2,1)
        mel_outputs_postnet[end_padding_ind, :] = 0
        mel_outputs_postnet = mel_outputs_postnet.permute(0,2,1)

        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return mel_outputs_postnet



class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x

def get_reverse_mask(lengths):
    ids = torch.arange(0, hparams.max_len, out=torch.LongTensor(int(hparams.max_len)))
    mask = (ids > lengths.unsqueeze(1)).bool()

    return mask