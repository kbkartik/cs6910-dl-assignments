from seq2seq import Seq2Seq
from transliteration import Transliteration
from hyperparams import default_q2_config, bayes_config_instance1
import wandb

wandb.init(config=default_q2_config, entity="kbdl", project="test")
config = wandb.config
HYPERPARAMS = config._as_dict()

# Set the run name
wandb.run.name = config["cell_type"] + "_emb_" + str(config["emb_dim"]) + "_n_enc_layers_" + str(config["n_enc_layers"])
wandb.run.name += "_hidden_dim_" str(config["hid_st_dim"]) + "_n_dec_layers_" + str(config["n_dec_layers"])
wandb.run.name += "_dout_" + str(config["dropout"])
wandb.run.name += "_bw_" + str(config["beam_width"]) + "_optim_" + config["optimizer"]
wandb.run.name += "_ep_" + str(config["epochs"]) + "_batch_" + str(config["batch_size"])
wandb.run.name += "_att_" if config["attention"] else ""

# Loading the datasets
transliteration = Transliteration(HYPERPARAMS, tgt_lang='hi')

# Training and evaluating the model
model = Seq2Seq(HYPERPARAMS)
model.train(transliteration.train_ds, transliteration.val_ds)
model.evaluate(transliteration.test_ds, "test", write_to_file=True, make_plots=True)