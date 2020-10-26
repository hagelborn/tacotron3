import torch
from train import *
from model.tacotron import Tacotron3
from sklearn.decomposition.pca import PCA
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import pickle
import platform
from pathlib import Path
from numpy.linalg import svd
from numpy.linalg import matrix_rank

def get_labels():
    if platform.system() == 'Linux':
        datapath = Path('~/data/label.pickle')
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

    model = Tacotron3()
    checkpoint_path = 'output/gaussclass/checkpoint_4000.pt'
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
    print('Matrix rank', matrix_rank(embeddings))

    # Labels
    #label_dict = get_labels()
    #bin_labels = get_binary_labels(label_dict)
    #labels = [bin_labels[name[:-4]] for name in names]

    # PCA
    reduced = pca.fit_transform(embeddings)
    plt.figure()
    plt.scatter(reduced[:,0],reduced[:,1]) #), c=labels)

    # SVD vectors
    s = svd(embeddings, compute_uv=False)
    print(len(s))
    plt.figure()
    plt.stem(s)
    plt.title('SVD vectors of embedding space - 64 dims')
    plt.show()