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
        self.speaker_encoder_num_layers = hparams.speaker_encoder_num_layers
        self.speaker_encoder_hidden_dim = hparams.speaker_encoder_hidden_dim
        self.encoder_dropout = hparams.encoder_dropout

        self.latent_dim = hparams.latent_dim

        self.time_encoder_num_layers = hparams.time_encoder_num_layers
        self.time_encoder_hidden_dim = hparams.time_encoder_hidden_dim

        self.n_mel_channels = hparams.n_mel_channels
        self.max_len = hparams.max_len

        self.speaker_encoder = nn.LSTM(input_size=self.n_mel_channels,
                            hidden_size=self.speaker_encoder_hidden_dim,
                            num_layers=self.speaker_encoder_num_layers,
                            batch_first=True,
                            dropout=self.encoder_dropout,
                            bidirectional=hparams.bidirect
                            )
        self.time_encoder = nn.LSTM(input_size=self.n_mel_channels,
                                    hidden_size=self.time_encoder_hidden_dim,
                                    num_layers=self.time_encoder_num_layers,
                                    dropout=self.encoder_dropout,
                                    batch_first=True)

        self.l1 = nn.Linear(self.speaker_encoder_hidden_dim*2,
                            self.latent_dim)
        self.l2 = nn.Linear(self.speaker_encoder_hidden_dim*2,
                            self.latent_dim)

    def reparametrize(self,mu,logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(mu.size())
        z = mu + std * esp
        return z

    def get_embed(self,x):
        mu, logvar = self.l1(x), self.l2(x)
        z = self.reparametrize(mu, logvar)
        return z

    def encode_speaker(self,x):
        _, (hidden, _) = self.speaker_encoder(x)
        y = torch.cat((hidden[-1],hidden[-2]),dim=-1)
        embedding = self.get_embed(y)
        return embedding

    def forward(self,mel_input):
        embedding = self.encode_speaker(mel_input)
        time_encoding, _ = self.time_encoder(mel_input)
        embedding = embedding.unsqueeze(1)
        embedding = embedding.repeat(1,self.max_len,1)

        encoder_output = torch.cat((time_encoding,embedding),dim=2)
        return encoder_output


    def inference(self,mel_input):

        return mel_input
