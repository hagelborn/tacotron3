from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import librosa.display
import librosa
import argparse


from data.dataloader import *
from data.transform import *
import torch
from evaluate import plot_mels
from train import load_model, warm_start_model
import model.hparams as hparams


def overlap_add(mel_frames,nmels):
    # Hamming windows used for overlap add
    hamm = torch.hamming_window(160,periodic=False)
    half_hamm = torch.cat((torch.ones(80), hamm[80:]),dim=0)
    hamm.unsqueeze_(1)
    half_hamm.unsqueeze_(1)

    time = (1 + len(mel_frames)) * 80
    recon = torch.zeros((time,nmels))
    ind = torch.arange(160) - 80

    for i in range(len(mel_frames)):
        frame = mel_frames[i]
        if frame.shape[1] != nmels:
            frame = frame.permute(1,0)

        ind += 80
        if i == 0:
            recon[ind,:] += frame * half_hamm
        elif i == len(mel_frames) - 1:
            recon[ind,:] += frame * torch.flip(half_hamm,dims=(0,))
        else:
            recon[ind, :] += frame * hamm

    return recon

def save_wav(mel, name, interp=False):
    if interp:
        wav_pred_name = Path.cwd().as_posix() + '/output/wavs/' + name + 'interp' + '.wav'
    else:
        wav_pred_name = Path.cwd().as_posix() + '/output/wavs/' + name + 'stitch' + '.wav'
    sr = hparams.sampling_rate
    wav_pred = librosa.feature.inverse.mel_to_audio(np.transpose(mel, axes=(1, 0)), sr=sr,
                                                    win_length=int(25 * sr / 1000),
                                                    hop_length=int(10 * sr / 1000))
    librosa.output.write_wav(wav_pred_name, wav_pred, sr=sr)
    return


if __name__ == '__main__':

    #############################   Argparser   ###############################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--checkpoint_path',type=str,required=True)
    parser.add_argument('-p','--plot', dest='plot', action='store_true')
    parser.add_argument('-s','--save_audio',dest='save', action='store_true')
    parser.add_argument('--encoder', dest='activate_encoder', action='store_true')
    parser.add_argument('--no_encoder', dest='activate_encoder', action='store_false')
    parser.add_argument('-n', '--n_samples', required=False, default=3, type=int)

    parser.set_defaults(activate_encoder=True)
    parser.set_defaults(plot=False)
    parser.set_defaults(save=False)

    args = parser.parse_args()


    #############################    Start of script   ######################################################
    model = load_model(args.activate_encoder)
    warm_start_model(args.checkpoint_path,model)
    model.eval()

    ds = Tacotron3Inference(active_encoder=args.activate_encoder,mode='validate')
    np.random.seed(1234)
    nbrs = [np.random.randint(0, ds.nbr_people() - 1) for x in range(args.n_samples)]

    inv = InverseLogCompression()

    with torch.no_grad():
        for nbr in nbrs:
            inputs, target = ds.get_person(nbr)
            person = ds.get_person_name(nbr)
            print(person)

            out = model.inference(inputs)

            recon = overlap_add(out, hparams.n_mel_channels)
            target = overlap_add(target, hparams.n_mel_channels)

            if args.save:
                recon = inv(recon)
                target = inv(target)
                m = torch.max(recon)
                m2 = torch.min(recon)

                mel = recon.detach().numpy()
                targetmel = target.detach().numpy()
                save_wav(mel, person + '_predict')
                save_wav(targetmel, person + '_true')

            if args.plot:
                logmel = recon.transpose(1, 0).detach().numpy()
                # plt.figure()
                # plot_mels(logmel)
                # plt.title('Reconstructed')
                # plt.show()
                fig = plt.figure()
                plt.subplot(121)
                plot_mels(logmel)
                plt.title('Predicted')
                plt.subplot(122)
                plot_mels(logtarget)
                plt.title('Target')
                plt.show()