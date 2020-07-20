import numpy as np
from torch.utils import data
from pathlib import Path
import torch
from data.transform import LogCompression
from itertools import product
import random
import model.hparams as hparams


class Tacotron3Train(data.Dataset):
    """ Dataset for training modified tacotron(or SV2TTS really)
    Retrives a random partial embedding/mel frame for each person every time for stochasticity
    Returns many combinations
    Input data: mel x, embedding y
    Ouput data: mel y
    """
    def __init__(self,transform=LogCompression(),
                 datapath=Path('/home/alex/tacotron3/data'),
                 mode='train'):
        self.transform = transform
        # Change this at later stage
        if hparams.n_mel_channels == 80:
                self.mel_path = datapath.joinpath('80mels')
        elif hparams.n_mel_channels == 256:
            self.mel_path = datapath.joinpath('256mels')

        self.transform = transform
        self.max_len = hparams.max_len

        # Making sure all people exist
        self.people = [p.name for p in self.mel_path.iterdir() if p.is_file() and p.name != '.DS_Store']

        if mode == 'train':
            self.people = self.people[round(0.2*len(self.people)):]
        else:
            self.people = self.people[:round(0.2*len(self.people))]


    def __len__(self):
        return len(self.people)

    def __getitem__(self, item):
        person = self.people[item]

        # Random choice from persons directory
        mel_path = self.mel_path.joinpath(person)


        mel = np.load(mel_path,allow_pickle=True)
        mel = torch.from_numpy(mel)
        mel = self.transform(mel)

        padded_mel = mel.new_zeros(self.max_len,mel.shape[1])
        if mel.shape[0] < self.max_len:
            mel_length = mel.shape[0]
            padded_mel[:mel_length,:] = mel
        else:
            mel_length = self.max_len
            padded_mel = mel[:mel_length,:]

        return (padded_mel, mel_length, padded_mel), padded_mel.transpose(1,0) #FIX - skipped clone() since this should return a new tensor anyway?
