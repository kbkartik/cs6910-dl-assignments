# Default Q2 config
default_q2_config = {
    'cell_type': "LSTM",
    'emb_dim': 64,
    'attention': False,
    'n_enc_layers': 3,
    'dropout': 0.0,
    'n_dec_layers': 1,
    'hid_st_dim': 128,
    'beam_width': 3,
    'optimizer': "Adam",
    'epochs': 30,
    'batch_size': 128,
}

bayes_config_q2_instance = {
    "name" : "A3_Q2_instance1",
    "method" : "bayes",
    "metric" : {
        "name" : "val_acc",
        "goal" : "maximize",
    },
    "parameters" : {
        "emb_dim" : {
            "values" : [32, 64, 256]
        },
        "cell_type" : {
            "values" : ['RNN', 'LSTM', 'GRU']
        },
        "n_enc_layers" : {
            "values" : [1, 2, 3]
        },
        "n_dec_layers" : {
            "values" : [1, 2, 3]
        },
        "hid_st_dim": {
            "values" : [32, 64, 256]
        },
        "dropout" : {
            "values" : [0, 0.2, 0.4, 0.6]
        },
        "attention" : {
            "values" : [False]
        },
        "batch_size": {
            "values" : [32, 64]
        },
        "beam_width": {
            "values": [3, 5, 7]
        },
        "optimizer": {
            "values": ["Adam", "Nadam", "Momentum"]
        },
        'epochs': {
            'values': [5, 10, 20]
        }
    }
}

default_q5_config = {
    'cell_type': "LSTM",
    'emb_dim': 128,
    'attention': True,
    'n_enc_layers': 3,
    'dropout': 0.0,
    'n_dec_layers': 1,
    'hid_st_dim': 128,
    'beam_width': 3,
    'optim': "Adam",
    'epochs': 5,
    'batch_size': 32,
}

bayes_config_q5_instance = {
    "name" : "A3_Q5_instance1",
    "method" : "bayes",
    "metric" : {
        "name" : "val_acc",
        "goal" : "maximize",
    },
    "parameters" : {
        "emb_dim" : {
            "values" : [64, 256]
        },
        "cell_type" : {
            "values" : ['RNN', 'LSTM', 'GRU']
        },
        "n_enc_layers" : {
            "values" : [1, 2, 3]
        },
        "n_dec_layers" : {
            "values" : [1, 2, 3]
        },
        "hid_st_dim": {
            "values" : [64, 256]
        },
        "dropout" : {
            "values" : [0, 0.2, 0.4, 0.6]
        },
        "attention" : {
            "values" : [True]
        },
        "batch_size": {
            "values" : [32, 64]
        },
        "beam_width": {
            "values": [3, 5, 7]
        },
        "optim": {
            "values": ["Adam", "Nadam", "SGDM"]
        },
        'epochs': {
            'values': [5, 10, 20]
        }
    }
}


best_q2_config = {
    'cell_type': "GRU",
    'emb_dim': 256,
    'attention': False,
    'n_enc_layers': 3,
    'dropout': 0.6,
    'n_dec_layers': 2,
    'hid_st_dim': 256,
    'beam_width': 7,
    'optim': "Adam",
    'epochs': 20,
    'batch_size': 64,
}

best_q5_config = {
    'cell_type': "LSTM",
    'emb_dim': 256,
    'attention': True,
    'n_enc_layers': 1,
    'dropout': 0.6,
    'n_dec_layers': 3,
    'hid_st_dim': 256,
    'beam_width': 7,
    'optim': "Nadam",
    'epochs': 20,
    'batch_size': 64,
}