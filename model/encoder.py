import torch
import torch.nn as nn
import model.hparams as hparams

class SimpleEncoder(nn.Module):
    def __init__(self):
        super(SimpleEncoder, self).__init__()
        self.max_len = hparams.max_len

    def forward(self,mel_input,embedding): # FIX mel_input unnecessary, change data_set for activate_encoder?
        embedding = embedding.unsqueeze(1)
        encoder_output = embedding.repeat(1, self.max_len, 1)
        return encoder_output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_num_layers = hparams.encoder_num_layers
        self.encoder_hidden_dim = hparams.encoder_hidden_dim

        self.n_mel_channels = hparams.n_mel_channels
        self.max_len = hparams.max_len

        self.lstm = nn.LSTM(input_size=self.n_mel_channels,
                            hidden_size=self.encoder_hidden_dim,
                            num_layers=self.encoder_num_layers,
                            batch_first=True,
                            bidirectional=hparams.bidirect
                            )

    def forward(self,mel_input,embedding):
        mel_encoding, _ = self.lstm(mel_input)
        #embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1,self.max_len,1)
        encoder_output = torch.cat((mel_encoding,embedding),dim=2)
        return encoder_output


    def inference(self,mel_input):

        return mel_input