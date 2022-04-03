class Utils:

    def __init__(self, train_path, test_path, batch_size, img_dim, train_transform=False, partb=False):
        self.initialize_dataloader(train_path, test_path, batch_size, img_dim, train_transform)
    
    def initialize_dataloader(self, train_path, test_path, batch_size, img_dim, train_transform):
        target_data_transform = T.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        train_val_data = ImageFolder(root=train_path, transform=T.Compose([]), target_transform=target_data_transform)

        train_size = int(0.9 * len(train_val_data))
        val_size = len(train_val_data) - train_size
        train_data, val_data = data.random_split(train_val_data, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

        train_data.dataset.transform.transforms += [T.RandomResizedCrop(size=img_dim) if train_transform else T.Resize(img_dim)]
        #train_data.dataset.target_transform = T.Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

        if partb:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            means, std = self.get_dataset_means_std(train_data)
            normalize = T.Normalize(mean=means, std=std)

        train_data.dataset.transform.transforms += [T.ToTensor(), normalize]
        
        val_data.dataset.transform = T.Compose([T.Resize(img_dim), T.ToTensor(), normalize])

        self.trainloader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.valloader = data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        test_data = ImageFolder(root=test_path, transform=T.Compose([T.Resize(img_dim), T.ToTensor(), normalize]), target_transform=target_data_transform)
        self.testloader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    def get_dataset_means_std(self, dataset):

        means = []
        stds = []
        
        for img, _ in dataset:
            means.append(torch.mean(img))
            stds.append(torch.std(img))

        mean = torch.mean(torch.tensor(means))
        std = torch.mean(torch.tensor(stds))

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