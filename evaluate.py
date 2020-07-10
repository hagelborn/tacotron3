
import argparse
import torch
from train import load_model, warm_start_model
from data.dataloader import Tacotron3Inference
from plotting_utils import plot_spectrogram_to_numpy
from torch.utils.data import DataLoader
from librosa.display import specshow
import matplotlib
matplotlib.use('MacOSX')
from matplotlib import pyplot as plt

def plot_mels(specs,fs=16000):
    specshow(specs, x_axis='time',
                             y_axis='mel', sr=fs,
                             fmax=fs / 2)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights_path', type=str, default=None,
                        required=True, help='weights path')
    args = parser.parse_args()

    model = load_model()
    warm_start_model(args.weights_path,model)
    valset = Tacotron3Inference()
    val_loader = DataLoader(valset,batch_size=4)

    model.eval()
    with torch.no_grad():
        for input in val_loader:
            output = model.inference(input)
            for mel_pred in output:
                maxval = torch.max(mel_pred)
                print(maxval)
                #data = plot_spectrogram_to_numpy(mel_pred)
                mel_pred = mel_pred.numpy()
                fig = plt.figure()
                plot_mels(mel_pred)
                plt.show()
            break
