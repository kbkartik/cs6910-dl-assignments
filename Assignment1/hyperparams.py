bayes_sweep_config = {
    "name" : "fmnist-bayes-50",
    "method" : "bayes",
    "metric" : {
        "name" : "val_acc",
        "goal" : "maximize",
    },
    "parameters" : {
        "activation" : {
            "values" : ['sigmoid', 'tanh', 'relu']
        },
        "batch_size" : {
            "values" : [16, 32, 64]
        },
        "epochs" : {
            "values" : [5, 10]
        },
        "hidden_layer_size": {
            "values" : [32, 64, 128]
        },
        "lamda" : {
            "values" : [0, 0.0005, 0.5]
        },
        "loss_fn" : {
            "values" : 'CCE'
        },
        "lr" : {
            "values" : [0.0001, 0.001]
        },
        "n_hidden_layers" : {
            "values" : [3, 4, 5]
        },
        "optimizer": {
            "values": ['sgd', 'sgdm', 'NAG', 'rmsprop', 'adam', 'nadam']
        },
        "weight_init_type" : {
            "values" : ['random', 'xavier']
        },
    }
}

random_sweep_config = {
    "name" : "fmnist-random-50",
    "method" : "random",
    "metric" : {
        "name" : "val_acc",
        "goal" : "maximize",
    },
    "parameters" : {
        "activation" : {
            "values" : ['sigmoid', 'tanh', 'relu']
        },
        "batch_size" : {
            "values" : [16, 32, 64]
        },
        "epochs" : {
            "values" : [5, 10]
        },
        "hidden_layer_size": {
            "values" : [32, 64, 128]
        },
        "lamda" : {
            "values" : [0, 0.0005, 0.5]
        },
        "loss_fn" : {
            "values" : 'CCE'
        },
        "lr" : {
            "values" : [0.0001, 0.001]
        },
        "n_hidden_layers" : {
            "values" : [3, 4, 5]
        },
        "optimizer": {
            "values": ['sgd', 'sgdm', 'NAG', 'rmsprop', 'adam', 'nadam']
        },
        "weight_init_type" : {
            "values" : ['random', 'xavier']
        },
    }
}

default_config = {
    "activation" : 'sigmoid',
    "batch_size" : 16,
    "epochs" : 5,
    "hidden_layer_size": 32,
    "lamda": 0.0005,
    "loss_fn" : 'CCE',
    "lr" : 0.0001,
    "n_hidden_layers" : 3,
    "optimizer": "adam",
    "weight_init_type" : 'random',
}