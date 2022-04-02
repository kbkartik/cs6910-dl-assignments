# bayes config for wandb
bayes_config = {
    "name" : "PartA_instance1",
    "method" : "bayes",
    "metric" : {
        "name" : "val_acc",
        "goal" : "maximize",
    },
    "parameters" : {
        "n_filters" : {
            "values" : [32, 64]
        },
        "conv_filter_org" : {
            "values" : [0.5, 1, 2]
        },
        "batch_norm" : {
            "values" : [True, False]
        },
        "conv_filter_size": {
            "values" : [2, 3, 4]
        },
        "n_mlp_neurons": {
            "values" : [128, 256]
        },
        "dropout" : {
            "values" : [0.4, 0.6]
        },
        "activation" : {
            "values" : ['relu', 'tanh']
        },
        "data_aug" : {
            "values" : [True, False]
        },
    }
}

# default config for wandb
default_config = {
    "n_filters" : 32,
    "conv_filter_org" : 1,
    "batch_norm" : True,
    "conv_filter_size": 3,
    "n_mlp_neurons": 256,
    "dropout" : 0.2,
    "activation" : 'relu',
    "data_aug" : True,
}