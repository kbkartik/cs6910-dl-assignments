import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch.utils.data as data
from torchvision import utils as torch_utils
from torchvision import models

import wandb
import numpy as np
import random
from PIL import Image

SEED = 123
device = torch.device("cuda")

class Utils:

    def __init__(self, train_path, test_path, batch_size, img_dims, train_transform=False, partb=False):
        self.initialize_dataloader(train_path, test_path, batch_size, img_dims, train_transform, partb)
    
    def initialize_dataloader(self, train_path, test_path, batch_size, img_dims, train_transform, partb):
        target_data_transform = T.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        train_val_data = ImageFolder(root=train_path, transform=T.Compose([]), target_transform=target_data_transform)

        train_size = int(0.9 * len(train_val_data))
        val_size = len(train_val_data) - train_size
        self.train_data, self.val_data = data.random_split(train_val_data, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

        self.train_data.dataset.transform.transforms += [T.RandomResizedCrop(size=img_dims) if train_transform else T.Resize(img_dims)]

        if partb:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            #self.dataset_means, self.dataset_stds = self.get_dataset_means_std(img_dims, self.train_data)
            self.dataset_means = torch.Tensor([0.4706, 0.4596, 0.3895])
            self.dataset_stds = torch.Tensor([0.1964, 0.1891, 0.1871])
            normalize = T.Normalize(mean=self.dataset_means, std=self.dataset_stds)

        self.train_data.dataset.transform.transforms += [T.ToTensor(), normalize]
        self.val_data.dataset.transform = T.Compose([T.Resize(img_dims), T.ToTensor(), normalize])

        self.trainloader = data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.valloader = data.DataLoader(self.val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        self.test_data = ImageFolder(root=test_path, transform=T.Compose([T.Resize(img_dims), T.ToTensor(), normalize]), target_transform=target_data_transform)
        self.testloader = data.DataLoader(self.test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def get_dataset_means_std(self, img_dims, dataset):

        means = []
        stds = []

        nimages = 0
        mean = 0.0
        var = 0.0
        self.trainloader.dataset.dataset.transform.transforms += [T.Resize(img_dims), T.ToTensor()]
        
        for _, data in enumerate(self.trainloader):
            batch, _ = data
            batch = batch.view(batch.size(0), batch.size(1), -1)

            nimages += batch.size(0)
            # Compute mean and std here
            mean += batch.mean(2).sum(0) 
            var += batch.var(2).sum(0)

        # Final step
        mean /= nimages
        var /= nimages
        std = torch.sqrt(var)
        self.trainloader.dataset.dataset.transform.transforms = []
        return mean, std

    def get_pretrained_models(self, model_name, num_classes, feature_extract):

        PRETRAINED_MODEL = {'RN50': [models.resnet50, (224, 224)],
                    'IV3' : [models.inception_v3, (299, 299)],
                    'MV3S' : [models.mobilenet_v3_small, (224, 224)]
                    }

        model, _ = PRETRAINED_MODEL[model_name]
        model = model(pretrained=True, progress=False)
        
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        if model_name == 'RN50':
            model.fc = nn.Linear(2048, num_classes)
        
        elif model_name == 'IV3':
            model.AuxLogits.fc = nn.Linear(768, num_classes)
            model.fc = nn.Linear(2048, num_classes)
            
        elif model_name == 'MV3S':
            model.classifier[3] = nn.Linear(1024, num_classes)
        
        model = model.to(device)
        return model

    def visualize_filters_cnn(self, model, layer=0, ch=0, allkernels=False, nrow=8, padding=1):
        model.eval()
        with torch.no_grad():
            tensor = model.cnn[layer].weight.data.clone()
            n, c, h, w = tensor.shape

            if n == 64:
                cols = int(np.sqrt(n))
            else:
                print("configure filter map plot manually")

            if allkernels: tensor = tensor.view(n*c, -1, h, w)
            elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

            grid = torch_utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)

            grid_img = wandb.Image(grid)
            wandb.log({"first_layer_filters": grid_img})

    def plot_predictions(self, imgs, targets, predictions):
        targets = targets.cpu()
        predictions = predictions.cpu()
        column_names = ["image", "truth", "guess"]
        predictions_table = wandb.Table(columns=column_names)

        # Unnormalizing image tensor
        self.invTransf = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], 
                                                std = 1/self.dataset_stds), 
                                    T.Normalize(mean = -1*self.dataset_means, 
                                                std = [ 1., 1., 1. ]),
                                    T.ToPILImage()])
        cls_names = {y:x for x, y in self.test_data.class_to_idx.items()}
        i = 0
        for i in range(10):
            row = [wandb.Image(self.invTransf(imgs[i])), cls_names[targets[i].item()], predictions[i]]
            predictions_table.add_data(*row)
        wandb.run.log({"Predictions_Table": predictions_table})

    def get_rand_image(self, dataloader_type='train'):
        dataloader_dict = {'train': next(iter(self.trainloader)),
                           'val': next(iter(self.valloader)),
                           'test': next(iter(self.testloader))}
        
        data = dataloader_dict[dataloader_type]
        rand_int = torch.randint(data[0].shape[0], size=(1,)).item()
        
        return (data[0][rand_int], data[1][rand_int])