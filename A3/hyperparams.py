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

bayes_config_instance1 = {
    "name" : "Q2_instance1",
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