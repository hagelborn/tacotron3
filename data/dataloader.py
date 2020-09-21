import numpy as np
from torch.utils import data
from pathlib import Path
import torch
from data.transform import LogCompression
import model.hparams as hparams
import platform
import random


class Tacotron3Train(data.Dataset):
    """ Dataset for training modified tacotron(or SV2TTS really)
    Retrives a random partial embedding/mel frame for each person every time for stochasticity
    Returns many combinations
    Input data: mel x, embedding y
    Ouput data: mel y
    """
    def __init__(self,transform=LogCompression(),
                 mode='train'):
        self.transform = transform
        if platform.system() == 'Linux':
            datapath = Path('/home/alex/tacotron3/data')
        else:
            datapath= Path('/Users/alexanderhagelborn/PycharmProjects/speaker_decoder/data')

        # Change this at later stage
        if hparams.n_mel_channels == 80:
            self.mel_path = datapath.joinpath('80mels')
        elif hparams.n_mel_channels == 256:
            self.mel_path = datapath.joinpath('256mels')

        self.label_path = datapath.joinpath('binlabels')

        self.transform = transform
        self.max_len = hparams.max_len

        # List of people, shuffled "randomly" with seed
        self.people = [p.name for p in self.mel_path.iterdir() if p.is_file() and p.name != '.DS_Store']
        random.seed(hparams.seed)
        random.shuffle(self.people)

        if mode == 'train':
            self.people = self.people[:round(0.8*len(self.people))]
        else:
            self.people = self.people[round(0.8*len(self.people)):]


    def __len__(self):
        return len(self.people)

    def get_name(self,item):
        return self.people[item]

    def get_person(self,item):
        person = self.people[item]
        mel_path = self.mel_path.joinpath(person)
        label_path = self.label_path.joinpath(person)

        mel = np.load(mel_path, allow_pickle=True)
        mel = torch.from_numpy(mel)
        mel = self.transform(mel)
        target_label = np.load(label_path, allow_pickle=True)
        target_label = torch.from_numpy(target_label).float()

        padded_mel = mel.new_zeros(self.max_len, mel.shape[1])
        if mel.shape[0] < self.max_len:
            mel_length = mel.shape[0]
            padded_mel[:mel_length, :] = mel
        else:
            mel_length = self.max_len
            padded_mel = mel[:mel_length, :]

        return (padded_mel, mel_length, 1), target_label

    def __getitem__(self, item):
        person = self.people[item]

        mel_path = self.mel_path.joinpath(person)
        label_path = self.label_path.joinpath(person)

        target_label = np.load(label_path,allow_pickle=True)
        target_label = torch.from_numpy(target_label).float()

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

        return (padded_mel, mel_length, padded_mel), (padded_mel.transpose(1,0), target_label) #FIX - skipped clone() since this should return a new tensor anyway?
