
import argparse
import torch
from train import load_model, warm_start_model
from data.dataloader import Tacotron3Inference, Tacotron3Train
from plotting_utils import plot_spectrogram_to_numpy
from torch.utils.data import DataLoader
from librosa.display import specshow
import matplotlib
matplotlib.use('MacOSX')
from matplotlib import pyplot as plt
import model.hparams as hparams

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
    parser.add_argument('--encoder', dest='activate_encoder', action='store_true')

    parser.add_argument('--no_encoder', dest='activate_encoder', action='store_false')

    parser.set_defaults(activate_encoder=True)
    args = parser.parse_args()

    model = load_model(args.activate_encoder)
    warm_start_model(args.weights_path,model)
    valset = Tacotron3Inference(active_encoder=args.activate_encoder, mode='validate')
    #valset = Tacotron3Train(mode='validate')
    val_loader = DataLoader(valset,batch_size=hparams.val_batch_size)

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input, target = batch
            output = model.inference(input)
            for i in range(hparams.val_batch_size):
                mel_pred = output[i]
                mel_target = target[i].permute(1,0)

                maxval = torch.max(mel_pred)
                minval = torch.min(mel_pred)
                print(minval.item())
                mel_pred = mel_pred.numpy()
                mel_target = mel_target.numpy()

                fig = plt.figure()
                plt.subplot(121)
                plot_mels(mel_pred)
                plt.title('Predicted')
                plt.subplot(122)
                plot_mels(mel_target)
                plt.title('Target')
                plt.show()
            break
