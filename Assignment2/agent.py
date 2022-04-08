import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import wandb

import numpy as np
import random

SEED = 123
device = torch.device("cuda")

class Agent:

    def __init__(self, model, utils_agent, num_epochs, is_inception=False):
        self.model = model
        self.utils_agent = utils_agent
        self.num_epochs = num_epochs
        self.is_inception = is_inception

        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        self.optimizer = optim.Adam(params_to_update, lr=0.0001)
        
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self,):
        self.model.train()
        for ep in range(self.num_epochs):
            running_train_loss = 0.0
            for i, batch_data in enumerate(self.utils_agent.trainloader):
                images, targets = batch_data
                images = images.to(device)
                targets = targets.to(device)
                
                self.optimizer.zero_grad()
                if self.is_inception:
                    outputs, aux_outputs = self.model(images)
                    loss1 = self.loss_fn(outputs, targets)
                    loss2 = self.loss_fn(aux_outputs, targets)
                    loss = loss1 + 0.4*loss2
                else:
                    pred_probs = self.model(images)
                    loss = self.loss_fn(pred_probs, targets)

                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item()
                if (i+1) % len(self.utils_agent.trainloader) == 0:
                    val_loss, val_acc = self.evaluate()
                    wandb.log({'train_loss': running_train_loss, 'val_loss': val_loss})
                    running_train_loss = 0
        
        return val_acc

    def evaluate(self, test_data=False):

        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            n_correct_preds = 0
            dataloader = self.utils_agent.valloader if not test_data else self.utils_agent.testloader

            for indices, batch_data in enumerate(dataloader):
                images, targets = batch_data
                images = images.to(device)
                targets = targets.to(device)

                pred_probs = self.model(images)
                pred_labels = pred_probs.argmax(1)
                loss = self.loss_fn(pred_probs, targets)

                total_loss += loss.item()
                targets_labels = targets.argmax(1)
                n_correct_preds += torch.sum(pred_labels == targets_labels)
            
            accuracy = n_correct_preds/len(dataloader.dataset)*100

        if test_data:
            self.utils_agent.plot_predictions(images, targets_labels, pred_probs)

        self.model.train()

        return total_loss, accuracy