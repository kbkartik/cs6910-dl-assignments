class CNN(nn.Module):

    def __init__(self, img_dims, HYPERPARAMS):
        super(CNN, self).__init__()
        
        N_LAYERS = 5
        N_CLASSES = 10
        rand_image = torch.rand(img_dims, device=device)
        n_conv_filters = lambda x: int(HYPERPARAMS['conv_filter_org'] * x)

        _, C_in, _, _ = img_dims
        in_channels = C_in
        out_channels = HYPERPARAMS['n_filters']
        seq_model = []

        for i in range(N_LAYERS):
            seq_model.append(nn.Conv2d(in_channels, out_channels, HYPERPARAMS['conv_filter_size'], padding='same'))
            if HYPERPARAMS['batch_norm']:
                seq_model.append(nn.BatchNorm2d(out_channels))
            seq_model.append(nn.ReLU() if HYPERPARAMS['activation'] == 'relu' else nn.Tanh())
            seq_model.append(nn.MaxPool2d(2))
            in_channels = out_channels
            out_channels = n_conv_filters(out_channels)
            if i == N_LAYERS - 1:
                seq_model.append(nn.Flatten())

        self.cnn = nn.Sequential(*seq_model).to(device)
        
        self.cnn.eval()
        with torch.no_grad():
            in_features = self.cnn(rand_image).shape[1]

        self.fc1 = nn.Linear(in_features, HYPERPARAMS['n_mlp_neurons'], device=device)
        self.head = nn.Linear(HYPERPARAMS['n_mlp_neurons'], N_CLASSES, device=device)
        self.dropout = nn.Dropout(HYPERPARAMS['dropout'])
    
    def forward(self, x):
        x = self.cnn(x)
        x = F.relu(self.dropout(self.fc1(x)))
        pred_probs = F.softmax(self.head(x), dim=1)
        return pred_probs