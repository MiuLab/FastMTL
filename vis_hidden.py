import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib.lines import Line2D
 

def vis_hidden(X, y, save_file, y_label):
    save_png = save_file+".png"
    save_npy = save_file+".npy"
    '''t-SNE'''
    tsne = TSNE(n_components=2, random_state=501, verbose=1)
    print(X.shape)
    X_tsne = tsne.fit_transform(X)
     
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
      
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min) 
    
    store_data = {"X":X_norm, "Y": y}
    np.save(save_npy, store_data)


    params = {'legend.fontsize': 30, 'legend.handlelength': 2}
    plt.rcParams.update(params)
    fig, ax = plt.subplots(figsize=(32,32))
    custom_lines = []
    yrange = [i for i in range(min(y),max(y)+1)]
    for i in yrange:
        custom_lines.append(Line2D([0], [0], color=plt.cm.Set1(i), lw=4))
    ax.legend(custom_lines, y_label)
    print("------------ plot figure -----------")
    for i in tqdm(range(X_norm.shape[0])):
        plt.plot(X_norm[i, 0], X_norm[i, 1], 'o', color=plt.cm.Set1(y[i]))
        plt.xticks([])
        plt.yticks([])
    plt.savefig(save_png)
