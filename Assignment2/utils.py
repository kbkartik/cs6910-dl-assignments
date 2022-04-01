class Utils:

    def __init__(self, train_path, test_path, batch_size, data_transform=None):
        self.train_path = train_path
        self.val_path = test_path
        self.data_transform = data_transform
        self.batch_size = batch_size
        self.initialize_dataloader()
    
    def initialize_dataloader(self,):

        train_val_data = ImageFolder(root=self.train_path, transform=self.data_transform)

        train_size = int(0.9 * len(train_val_data))
        val_size = len(train_val_data) - train_size

        train_data, val_data = data.random_split(train_val_data, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
        
        train_data.target_transform = T.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        #val_data.target_transform = T.toTensor()

        self.trainloader = data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.valloader = data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        test_data = ImageFolder(root=test_path, transform=self.data_transform,)
        self.testloader = data.DataLoader(test_data, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)