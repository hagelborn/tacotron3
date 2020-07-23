import torch
from train import *
from model.tacotron import Tacotron3
from sklearn.decomposition.pca import PCA
import matplotlib
matplotlib.use('MACOSX')
from matplotlib import pyplot as plt

import pickle
import platform
from pathlib import Path

def get_labels():
    if platform.system() == 'Linux':
        datapath = Path('/home/alex/tacotron3/data')
    else:
        datapath = Path('/Users/alexanderhagelborn/PycharmProjects/speaker_decoder/data/label.pickle')

    with open(datapath,'rb') as f:
        label_dict = pickle.load(f)

    return label_dict

def get_binary_labels(label_dict):
    bin_labels = {}
    for key in label_dict:
        if label_dict[key] == 'male':
            bin_labels[key] = 1
        elif label_dict[key] == 'female':
            bin_labels[key] = 0
        else:
            print('lol someone has a label thats misspelled')
    return bin_labels





if __name__ == '__main__':
    pca = PCA(2)
    label_dict = get_labels()

    model = Tacotron3()
    checkpoint_path = 'checkpoints/3900.pt'
    warm_start_model(checkpoint_path,model)
    dataset = Tacotron3Train()
    nbr_items = len(dataset)


    model.eval()
    embedding_list = []
    names = []
    with torch.no_grad():
        for i in range(nbr_items):
            input, _ = dataset[i]
            embeddings = model.get_embeddings(input)
            embedding_list.append(embeddings)
            names.append(dataset.get_name(i))

    embeddings = torch.Tensor(hparams.batch_size*len(embedding_list), 100)
    torch.cat(embedding_list, out=embeddings)

    embeddings = embeddings.detach().numpy()

    # Labels
    label_dict = get_labels()
    bin_labels = get_binary_labels(label_dict)
    labels = [bin_labels[name[:-4]] for name in names]

    reduced = pca.fit_transform(embeddings)
    plt.figure()
    plt.scatter(reduced[:,0],reduced[:,1], c=labels)
    plt.show()
