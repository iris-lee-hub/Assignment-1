import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from skimage.transform import resize, rescale

def load_files(file_indices):
    size = len(file_indices)
    X = np.zeros((size, 2048, 2048, 51))
    y = np.zeros((size, 2048, 2048, 2))
    cell_types = []
    for index, file in enumerate(file_indices):
        point = np.load(f"keren/Point{file:02}.npz", allow_pickle = True)
        X[index] = point['X']
        y[index] = point['y']
        cell_types.append(point["cell_types"])
    return (X, y, cell_types)
        
def normalize(X):
    for channel in range(X.shape[-1]):
        X[:,:,:,channel] = X[:,:,:,channel]-np.min(X[:,:,:,channel])
        if np.max(X[:,:,:,channel])!= 0:
            X[:,:,:,channel] = X[:,:,:,channel]/np.max(X[:,:,:,channel])
    return X

def cropped_views(X, y, total_cells, cell_types_list, exclude_type = None, maximum = 4000):
    # Cropped views of each cell
    training_set = np.zeros((total_cells,500, 51))
    training_label = np.zeros(total_cells)

    # for indexing
    count = 0
    index = 0
    
    # iterate over all the cells in all the batches
    for batch in range(X.shape[0]):
        n_cells = int(np.max(y[batch,:,:,0]))
        for cell in range(1,n_cells):
            # only save data if we are not excluding this cell type
            num = cell_types_list[batch].item()[cell] 
            if num != 0 and num != 1 and num != 17 and (count < maximum or num != exclude_type):
                # get the signal for all the pixels in each cell
                cropped_view = X[batch][np.where(y[batch,:,:,0]== cell)]
                training_set[index] = resize(cropped_view, (500, 51), anti_aliasing = True)
                training_label[index] = num - 2
                index +=1
                if num == exclude_type:
                    count +=1
    return (training_set, training_label)

def cropped_views_testing(X, y, total_cells):
    # Cropped views of each cell
    training_set = np.zeros((total_cells,500, 51))

    # for indexing
    index = 0
    
    # iterate over all the cells in all the batches
    for batch in range(X.shape[0]):
        n_cells = int(np.max(y[batch,:,:,0]))
        for cell in range(1,n_cells):
            # only save data if we are not excluding this cell type
            
            # get the signal for all the pixels in each cell
            cropped_view = X[batch][np.where(y[batch,:,:,0]== cell)]
            training_set[index] = resize(cropped_view, (500, 51), anti_aliasing = True)
            
            index +=1

    return (training_set)

def expression_panel(cell_type_Expression, meta_cell_types_list, meta_channels):
    
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(20)


    ax = fig.add_subplot(111)
    cax = ax.matshow(cell_type_Expression, interpolation = "nearest")
    ax.yaxis.set_major_locator(mticker.FixedLocator(np.arange(len(meta_cell_types_list))))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter(meta_cell_types_list))
    ax.set_yticklabels(meta_cell_types_list);
    ax.xaxis.set_major_locator(mticker.FixedLocator(np.arange(len(meta_channels))))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(meta_channels))
    ax.set_xticklabels(meta_channels, rotation = 'vertical')
    