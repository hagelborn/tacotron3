from embedding_analysis import get_labels
import sounddevice as sd
from pathlib import Path
import librosa as l
import pickle

if __name__ == '__main__':

    wav_path = Path('/Users/alexanderhagelborn/PycharmProjects/speaker_decoder/data/trimmed_wav')
    label_path = Path('/Users/alexanderhagelborn/PycharmProjects/speaker_decoder/data/label.pickle')
    label_dict = get_labels()

    label_set = set(label_dict.keys())

    wav_set = set([file.stem for file in wav_path.iterdir() if file.is_file() and file.name != '.DS_Store'])
    no_labels = wav_set.difference(label_set)
    no_labels = no_labels.intersection(wav_set)

    new_labels = dict()

    new_names = []
    for name in no_labels:
        if name[:3] == 'fem':
            new_labels[name] = 'female'
            new_names.append(name)
        elif name[:3] == 'mal':
            new_labels[name] = 'male'
            new_names.append(name)

    new_names = set(new_names)
    no_labels = no_labels.difference(new_names)

    for name in no_labels:
        print(name)
        wavfile = wav_path.joinpath(name + '.wav')
        wav, fs = l.core.load(wavfile)
        sd.play(wav,fs)
        new_labels[name] = input('Enter label of voice: ')

    label_dict.update(new_labels)
    label_path = label_path.as_posix() + '.pickle'
    with open(label_path,'wb') as f:
        pickle.dump(label_dict,f)
