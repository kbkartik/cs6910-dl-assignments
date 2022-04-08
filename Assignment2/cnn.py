import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

SEED = 123
device = torch.device("cuda")

class CNN(nn.Module):

    def __init__(self, img_dims, HYPERPARAMS):
        super(CNN, self).__init__()
        
        N_LAYERS = 5
        N_CLASSES = 10
        self.img_dims = img_dims
        rand_image = torch.rand(img_dims, device=device)
        n_conv_filters = lambda x: int(HYPERPARAMS['conv_filter_org'] * x)

        _, C_in, _, _ = img_dims
        in_channels = C_in
        out_channels = HYPERPARAMS['n_filters']
        seq_model = nn.Sequential()

        for i in range(N_LAYERS):
            seq_model.add_module("Conv2d_"+str(i+1), nn.Conv2d(in_channels, out_channels, HYPERPARAMS['conv_filter_size'], padding='same'))
            if HYPERPARAMS['batch_norm']:
                seq_model.add_module("BN_"+str(i+1), nn.BatchNorm2d(out_channels))
            seq_model.add_module(HYPERPARAMS['activation']+'_'+str(i+1), nn.ReLU() if HYPERPARAMS['activation'] == 'relu' else nn.Tanh())
            seq_model.add_module("Maxpool2d_"+str(i+1), nn.MaxPool2d(2))
            in_channels = out_channels
            out_channels = n_conv_filters(out_channels)
            if i == N_LAYERS - 1:
                seq_model.add_module("Flatten", nn.Flatten())

        """
        self.cnn = nn.Sequential(*seq_model).to(device)
        self.cnn.eval()
        with torch.no_grad():
            in_features = self.cnn(rand_image).shape[1]

        self.fc1 = nn.Linear(in_features, HYPERPARAMS['n_mlp_neurons'], device=device)
        self.dropout = nn.Dropout(HYPERPARAMS['dropout'])
        self.head = nn.Linear(HYPERPARAMS['n_mlp_neurons'], N_CLASSES, device=device)
        """
        """
        seq_model = seq_model.to(device)
        seq_model.eval()
        with torch.no_grad():
            in_features = seq_model(rand_image).shape[1]

        seq_model.add_module('fc1', nn.Linear(in_features, HYPERPARAMS['n_mlp_neurons'], device=device))
        seq_model.add_module('Dropout', nn.Dropout(HYPERPARAMS['dropout']))
        seq_model.add_module('fcReLU', nn.ReLU())
        seq_model.add_module('head', nn.Linear(HYPERPARAMS['n_mlp_neurons'], N_CLASSES, device=device))
        
        self.seq_model = seq_model
        """
        self.cnn = seq_model.to(device)
        self.cnn.eval()
        with torch.no_grad():
            in_features = self.cnn(rand_image).shape[1]

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc1', nn.Linear(in_features, HYPERPARAMS['n_mlp_neurons'], device=device))
        self.classifier.add_module('Dropout', nn.Dropout(HYPERPARAMS['dropout']))
        self.classifier.add_module('fcReLU', nn.ReLU())
        self.classifier.add_module('head', nn.Linear(HYPERPARAMS['n_mlp_neurons'], N_CLASSES, device=device))
        self.classifier = self.classifier.to(device)
    
    def forward(self, x):
        """
        x = self.cnn(x)
        x = F.relu(self.dropout(self.fc1(x)))
        pred_probs = F.softmax(self.head(x), dim=1)
        """
        x = self.cnn(x)
        x = self.classifier(x)
        pred_probs = F.softmax(x, dim=1)
        return pred_probs