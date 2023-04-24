import numpy as np
import yaml
from pathlib import Path
import wrangle_data as wd
import LinearClassifier as lc
from sklearn.preprocessing import MultiLabelBinarizer
import torch


def classify(X, y):
    data_of_interest = [2, 9,10,11,13,14,15,16,19,24,25,26,27,28,29,34,35,37,44,45,47,49,50]
    X =X[..., data_of_interest]
    X = wd.normalize(X)
    
    total_cells = 0
    for i in range(X.shape[0]):
        total_cells += int(np.max(y[i,:,:,0]))

    training_set, training_label = wd.cropped_views(X, y, total_cells)

    X = np.mean(training_set, axis = 1)
    mlb = MultiLabelBinarizer()
    label_one_hot = mlb.fit_transform([[label] for label in training_label])

    model = lc.LinearClassifier(num_features = X.shape[1], num_classes = 15)
    model.w = torch.load("weight_tensor.pt")

    x = torch.tensor(X, dtype = torch.float32)
    y_pred = model.predict(x).detach().numpy()
    y_pred = np.argmax(y_pred, axis = 1)
    
    return dict(map(lambda i,j: (i,j),np.unique(y), y))




