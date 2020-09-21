import torch
from model.tacotron import Tacotron3
from data.dataloader import Tacotron3Train
from torch.utils.data import DataLoader
from data.transform import InverseLogCompression
from librosa.feature.inverse import mel_to_audio
from librosa.output import write_wav
from train import warm_start_model

def save_wav(mel,name):
    mel = mel.detach().numpy()
    fs = 16000
    wav = mel_to_audio(mel,sr=fs,n_fft=2048,hop_length=int(10/1000*fs),win_length=int(25/1000*fs))
    path = 'output/wavs/' + name + '.wav'
    write_wav(path,wav,fs)
    return


if __name__ == '__main__':

    model = Tacotron3()
    warm_start_model('output/gaussclass/checkpoint_4000.pt',model)
    ds = Tacotron3Train()

    dl = iter(DataLoader(ds,batch_size=4))
    inv = InverseLogCompression()

    batch = next(dl)
    inputs, labels = batch

    output = model.inference(inputs)
    for i in range(output.shape[0]):
        logmel = output[i]
        mel = inv(logmel)
        save_wav(mel,str(i))


