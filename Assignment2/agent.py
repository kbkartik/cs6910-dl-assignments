class Agent:

    def __init__(self, model, utils_agent, num_epochs):
        self.model = model
        self.utils_agent = utils_agent
        self.num_epochs = num_epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0002)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self,):
        
        for ep in range(self.num_epochs):

            running_train_loss = 0.0
            for i, data in enumerate(self.utils_agent.trainloader):
                images, targets = data
                images = images.to(device)
                targets = targets.to(device)
                
                self.optimizer.zero_grad()
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

            for j, data in enumerate(dataloader):
                images, targets = data
                images = images.to(device)
                targets = targets.to(device)

                pred_probs = self.model(images)
                pred_labels = pred_probs.argmax(1)
                loss = self.loss_fn(pred_probs, targets)

                total_loss += loss.item()
                n_correct_preds += torch.sum(pred_labels == targets)
            
            accuracy = n_correct_preds/len(dataloader.dataset)*100

        return total_loss, accuracy