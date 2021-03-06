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
                 mode='train'):
        self.transform = transform
        # Change this at later stage
        cwd = Path.cwd()
        self.datapath = cwd.joinpath('data')
        if hparams.n_mel_channels == 80:
            self.mel_path = self.datapath.joinpath('80mels')
        elif hparams.n_mel_channels == 256:
            self.mel_path = self.datapath.joinpath('256mels')

        self.emb_path = self.datapath.joinpath('embeddings')
        self.transform = transform
        self.max_len = hparams.max_len

        # Making sure all people exist
        mel_people = set(p.name for p in self.mel_path.iterdir() if p.is_file() and p.name != '.DS_Store')
        emb_people = set(p.name for p in self.emb_path.iterdir() if p.is_file() and p.name != '.DS_Store')
        self.people = [person for person in emb_people if person in mel_people]

        random.seed(hparams.seed)
        random.shuffle(self.people)

        if mode == 'train':
            self.people = self.people[:round(0.8 * len(self.people))]
        else:
            self.people = self.people[round(0.8 * len(self.people)):]


    def __len__(self):
        return len(self.people)

    def __getitem__(self, item):
        person = self.people[item]

        # Random choice from persons directory
        mel_path = self.mel_path.joinpath(person)
        emb_path = self.emb_path.joinpath(person)

        embedding = np.load(emb_path,allow_pickle=True)
        embedding = torch.from_numpy(embedding)

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

        return (padded_mel, mel_length, embedding, padded_mel), padded_mel.transpose(1,0) #FIX - skipped clone() since this should return a new tensor anyway?



class Tacotron3Inference(data.Dataset):
    """ Dataset for training modified tacotron(or SV2TTS really)
    Retrives a random partial embedding/mel frame for each person every time for stochasticity
    Returns many combinations
    Input data: mel x, embedding y
    Ouput data: mel y
    """
    def __init__(self, active_encoder,
                 transform=LogCompression(),
                 datapath=Path('/Users/alexanderhagelborn/PycharmProjects/speaker_decoder/data/sequence_256'),
                 mode='validate',
                 ):
        self.transform = transform
        self.mel_path = datapath.joinpath('melspectrogram_frames')
        self.emb_path = datapath.joinpath('partial_embeddings')
        self.active_encoder = active_encoder

        # Making sure all people exist
        emb_people = set(p.name for p in self.emb_path.iterdir() if p.is_dir())
        mel_people = set(p.name for p in self.mel_path.iterdir() if p.is_dir())
        self.people = sorted(emb_people.intersection(mel_people))

        if mode == 'train':
            self.people = self.people[:round(0.8 * len(self.people))]
        else:
            self.people = self.people[round(0.8 * len(self.people)):]

        self.people_combinations = [combo for combo in product(self.people, self.people) if combo[0] != combo[1]] # Not allowed to combo with themselves

    def get_person(self,item):
        person = self.people[item]
        mel_path = self.mel_path.joinpath(person)
        emb_path = self.emb_path.joinpath(person)

        embeddings = [np.load(x.as_posix()) for x in emb_path.iterdir() if x.is_file() and x.name != '.DS_Store']
        mels = [np.load(x.as_posix()) for x in mel_path.iterdir() if x.is_file() and x.name != '.DS_Store']

        embeddings = [embeddings[0] for x in range(10)]
        mels = [mels[0] for x in range(10)]

        embeddings = [torch.from_numpy(embedding) for embedding in embeddings]
        embeddings = torch.stack(embeddings)

        mels = [torch.from_numpy(mel) for mel in mels]
        mels = torch.stack(mels)
        mels = self.transform(mels)
        if self.active_encoder:
            input_mels = mels
        else:
            input_mels = 0

        return (input_mels, embeddings), mels

    def get_person_name(self,item):
        return self.people[item]

    def nbr_people(self):
        return len(self.people)

    def __len__(self):
        return len(self.people_combinations)

    def __getitem__(self, item):
        xpers, ypers = self.people_combinations[item]

        # Random choice from persons directory
        input_mel_path = self.mel_path.joinpath(xpers)
        input_emb_path = self.emb_path.joinpath(ypers)
        output_mel_path = self.mel_path.joinpath(ypers)

        xmel = random.choice([x for x in input_mel_path.iterdir() if x.is_file() and x.name != '.DS_Store'])
        yemb = random.choice([x for x in input_emb_path.iterdir() if x.is_file() and x.name != '.DS_Store'])
        ymel = random.choice([x for x in output_mel_path.iterdir() if x.is_file() and x.name != '.DS_Store'])

        input_emb = np.load(yemb,allow_pickle=True)
        target_mel = np.load(ymel,allow_pickle=True)

        input_emb = torch.from_numpy(input_emb)
        target_mel = torch.from_numpy(target_mel)

        if self.active_encoder:
            input_mel = np.load(xmel, allow_pickle=True)
            input_mel = torch.from_numpy(input_mel)
            input_mel = self.transform(input_mel)
        else:
            input_mel = 0

        target_mel = self.transform(target_mel)

        return (input_mel, input_emb), target_mel  # FIX - Maybe inference should pair the right embedding with the right mel? thought
