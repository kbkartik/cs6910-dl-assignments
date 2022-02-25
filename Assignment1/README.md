## Assignment 1

The notebook A1.ipynb contains the main code to create and train a MLP. Also, wandb sweeps are run in the same notebook. Hyperparameter configs are fetched from hyperparams.py and passed to wandb.sweep in A1.ipynb train() method. Similarly, other sub-modules such as optimizers, activations, loss function are written in their respective python files from where they can be called.