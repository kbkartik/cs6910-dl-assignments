# bayes config for wandb
bayes_config_instance1 = {
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

# default config for wandb for instance 1
default_config_instance1 = {
    "n_filters" : 32,
    "conv_filter_org" : 1,
    "batch_norm" : True,
    "conv_filter_size": 3,
    "n_mlp_neurons": 256,
    "dropout" : 0.2,
    "activation" : 'relu',
    "data_aug" : True,
}

# bayes config for wandb
bayes_config_instance2 = {
    "name" : "PartA_instance2",
    "method" : "bayes",
    "metric" : {
        "name" : "val_acc",
        "goal" : "maximize",
    },
    "parameters" : {
        "n_filters" : {
            "values" : [64]
        },
        "conv_filter_org" : {
            "values" : [2]
        },
        "batch_norm" : {
            "values" : [True]
        },
        "conv_filter_size": {
            "values" : [3, 5]
        },
        "n_mlp_neurons": {
            "values" : [128, 256]
        },
        "dropout" : {
            "values" : [0, 0.4, 0.6]
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
default_config_instance2 = {
    "n_filters" : 64,
    "conv_filter_org" : 2,
    "batch_norm" : True,
    "conv_filter_size": 5,
    "n_mlp_neurons": 128,
    "dropout" : 0,
    "activation" : 'relu',
    "data_aug" : True,
}

# Part A best config
part_A_best_config = {
    "name" : "PartA_best",
    "method" : "grid",
    "parameters" : {
        "n_filters" : {
            "values" : [64]
        },
        "conv_filter_org" : {
            "values" : [2]
        },
        "batch_norm" : {
            "values" : [True]
        },
        "conv_filter_size": {
            "values" : [3]
        },
        "n_mlp_neurons": {
            "values" : [128]
        },
        "dropout" : {
            "values" : [0]
        },
        "activation" : {
            "values" : ['relu']
        },
        "data_aug" : {
            "values" : [False]
        },
    }
}

# default best config PartA
default_best_config_part_a = {
    "n_filters" : 64,
    "conv_filter_org" : 2,
    "batch_norm" : True,
    "conv_filter_size": 3,
    "n_mlp_neurons": 128,
    "dropout" : 0,
    "activation" : 'relu',
    "data_aug" : False,
}

# partb grid config for wandb
partb_config = {
    "name" : "PartB",
    "method" : "grid",
    "metric" : {
        "name" : "val_acc",
        "goal" : "maximize",
    },
    "parameters" : {
        "model_name" : {
            "values" : [['RN50', (224, 224)], 
                        ['IV3', (299, 299)], 
                        ['MV3S', (224, 224)]]
        },
        "strategy" : {
            "values" : ['finetuning', 'feature_extraction']
        },
        "data_aug" : {
            "values" : [True, False]
        },
    }
}

# part B default config for wandb
partb_default_config = {
    "model_name" : ['RN50', (224, 224)],
    "strategy" : 'feature_extraction',
    "data_aug" : False,
}