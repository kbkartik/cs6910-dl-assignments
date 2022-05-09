import tensorflow as tf
from tensorflow.keras.optimizers import Nadam, Adam, SGD
from tensorflow_addons.seq2seq import GreedyEmbeddingSampler, BeamSearchDecoder, BasicDecoder, tile_batch

from utils import Masked_CrossEntropy, _plot_predictions, _plot_attention_maps, _plot_connectivity

import numpy as np
import pandas as pd
import wandb
import os

class Seq2Seq:
    def __init__(self, dataset_configs, HYPERPARAMS):
        self.dataset_configs = dataset_configs
        self.HYPERPARAMS = HYPERPARAMS
        self.encoder = Encoder(dataset_configs, HYPERPARAMS)
        self.decoder = Decoder(dataset_configs, HYPERPARAMS)

        # Setting the optimizer
        if HYPERPARAMS['optim'] == "SGDM":
            self.optimizer = SGD(momentum=0.9, nesterov=True)
        elif HYPERPARAMS['optim'] == "Adam":
            self.optimizer = Adam()
        else:
            self.optimizer = Nadam()

        # Greedy Decoder
        self.greedy_dec = BasicDecoder(self.decoder.rnn_cell, GreedyEmbeddingSampler(), output_layer=self.decoder.fc, maximum_iterations=dataset_configs['max_tgt_len'])

        # Beam Search
        self.beam_decoder = BeamSearchDecoder(self.decoder.rnn_cell, beam_width=HYPERPARAMS["beam_width"], output_layer=self.decoder.fc, maximum_iterations=dataset_configs['max_tgt_len'])

    @tf.function
    def train_step(self, inp, targ):

        loss = 0

        # Using gradient tape to track gradients for updates
        with tf.GradientTape() as tape:

            # Passing inputs through the encoder network
            enc_output, states = self.encoder(inp)

            # Removing the end word token
            dec_input = targ[:, :-1]

            # Removing the start word token
            real = targ[:, 1:]

            # Sharing encoder state / outputs with the decoder network
            if self.HYPERPARAMS["attention"]:
                self.decoder.attention_mechanism.setup_memory(enc_output)
                decoder_initial_state = self.decoder.build_initial_state(self.HYPERPARAMS["batch_size"], states, tf.float32)
            else:
                decoder_initial_state = self.decoder.get_initial_state(states)

            # Passing the inputs and the encoder state throgh
            pred = self.decoder(dec_input, decoder_initial_state)
            logits = pred.rnn_output
            loss = Masked_CrossEntropy(real, logits)

        # Accumulating gradients and updating weights
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    @tf.function
    def val_step(self, inp, targ):

        # Passing inputs through the encoder network
        enc_output, states = self.encoder(inp)

        # Removing the end word token
        dec_input = targ[:, :-1]

        # Removing the start word token
        real = targ[:, 1:]

        # Sharing encoder state / outputs with the decoder network
        if self.HYPERPARAMS["attention"]:
            self.decoder.attention_mechanism.setup_memory(enc_output)
            decoder_initial_state = self.decoder.build_initial_state(self.HYPERPARAMS["batch_size"], states, tf.float32)
        else:
            decoder_initial_state = self.decoder.get_initial_state(states)

        # Passing the inputs and the encoder state throgh
        pred = self.decoder(dec_input, decoder_initial_state)
        logits = pred.rnn_output
        loss = Masked_CrossEntropy(real, logits)

        return loss

    def train(self, train_dataset, val_dataset, eval_on_val_ds=True):

        steps_per_epoch = (self.dataset_configs["n_train_egs"] // self.HYPERPARAMS["batch_size"])
        patience = 5
        wait = 0
        best = 0

        for epoch in range(self.HYPERPARAMS["epochs"]):
            train_batch_loss = 0
            val_batch_loss = 0
            for batch, ((t_inp, t_targ), (v_inp, v_targ)) in enumerate(zip(train_dataset.take(steps_per_epoch), val_dataset[0].take(steps_per_epoch))):

                # Compute and log train and val loss for each batch
                train_batch_loss += self.train_step(t_inp, t_targ).numpy()
                val_batch_loss += self.val_step(v_inp, v_targ).numpy()

            wandb.log({"train_loss": float(train_batch_loss/steps_per_epoch), "val_loss": float(val_batch_loss/steps_per_epoch)})

            # The early stopping strategy: stop the training if `val_loss` does not
            # decrease over a certain number of epochs.
            wait += 1
            if val_batch_loss > best:
                best = val_batch_loss
                wait = 0
            if wait >= patience:
                break
        
        # Compute validation accuracy on the entire validation set every epoch
        if eval_on_val_ds:
            val_accuracy = self.evaluate(val_dataset[1], "val", write_to_file=False, make_plots=False)
            wandb.log({"val_acc": float(val_accuracy*100)})

    def evaluate(self, inputs, dataset_type, write_to_file=False, make_plots=False):

        inference_batch_size = inputs.shape[0]

        # Pass the inputs through the encoder network
        enc_out, states = self.encoder(inputs)

        # Tile the encoder outputs and hidden states for beam search
        if self.HYPERPARAMS["attention"]:
            enc_out = tile_batch(enc_out, multiplier=self.HYPERPARAMS["beam_width"])
            self.decoder.attention_mechanism.setup_memory(enc_out)
            hidden_state = tile_batch(states, multiplier=self.HYPERPARAMS["beam_width"])
            hidden_state = self.decoder.build_initial_state(self.HYPERPARAMS["beam_width"] * inference_batch_size, hidden_state, tf.float32,)
        else:
            if self.HYPERPARAMS["cell_type"] != "LSTM":
                states = states[0]

            states = self.decoder.get_initial_state(states)
            hidden_state = tile_batch(states, multiplier=self.HYPERPARAMS["beam_width"])
            enc_out = tile_batch(enc_out, multiplier=self.HYPERPARAMS["beam_width"])

            if self.HYPERPARAMS["n_dec_layers"] == 1:
                hidden_state = (hidden_state,)
            else:
                if self.HYPERPARAMS["cell_type"] == "LSTM":
                    hidden_state = tuple([hidden_state[2 * i], hidden_state[2 * i + 1]] for i in range(int(len(hidden_state) / 2)))
                else:
                    hidden_state = tuple(hidden_state)

        # Start and end tokens for beam search
        start_tokens = tf.fill([inference_batch_size], self.dataset_configs["tgt_lang_tokenizer"].word_index["\t"],)
        end_token = self.dataset_configs["tgt_lang_tokenizer"].word_index["\n"]

        decoder_embedding_matrix = self.decoder.embedding.variables[0]

        # Running beam search
        outputs, final_state, sequence_lengths = self.beam_decoder(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=hidden_state,)

        # Getting the output in desired shape
        final_outputs = tf.transpose(outputs.predicted_ids, perm=(0, 2, 1))
        beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0, 2, 1))
        
        # Getting numpy arrays from tensors
        final_outputs, beam_scores = final_outputs.numpy(), beam_scores.numpy()
        
        # Getting the inputs and allowed labels
        labels = self.dataset_configs["updated_"+dataset_type+"_tgts"]
        texts = list(labels.keys())

        # Evaluation loop
        predictions = {
            "Latin Script": [],
            "Predicted Native Script": [],
            "isAccurate": [],
        }
        for i in range(inference_batch_size):
            # Text and allowed labels
            text = texts[i]
            allowed_labels = labels[text]

            # Converting numpy array back to text format
            output = self.dataset_configs["tgt_lang_tokenizer"].sequences_to_texts(final_outputs[i])

            # Using the end token to end words; removing spaces
            output = [a[: a.index("\n")].replace(" ", "") if "\n" in a else a.replace(" ", "") for a in output]

            # Saving input, predicted text and whether the model was accurate
            predictions["Latin Script"].append(text)
            predictions["Predicted Native Script"].append(output[0])
            predictions["isAccurate"].append(output[0] in allowed_labels)

        # Converting to pandas dataframe
        predictions = pd.DataFrame.from_dict(predictions)

        # Computing accuracy
        accuracy = predictions["isAccurate"].mean()
        
        # Log test accuracy
        if dataset_type == "test":
            wandb.log({"test_accuracy": accuracy})

        # Write the pandas dataframe into a file
        if write_to_file:
            path = os.path.join("predictions", wandb.run.name)

            if not os.path.exists(path):
                os.makedirs(path)

            if self.HYPERPARAMS["attention"]:
                predictions[["Latin Script", "Predicted Native Script", "isAccurate"]].to_csv(path + "/predictions_attention.csv", index=False)
            else:
                predictions[["Latin Script", "Predicted Native Script", "isAccurate"]].to_csv(path + "/predictions_vanilla.csv", index=False)
        
        # Plot correct and incorrect predictions
        if make_plots:
            _plot_predictions(predictions)
            
            if self.HYPERPARAMS["attention"]:
                _plot_attention_maps(self.encoder, self.decoder, self.greedy_dec, self.HYPERPARAMS, inputs, dataset_type)
            
            _plot_connectivity(inputs, dataset_type, self.dataset_configs, self.HYPERPARAMS, self.encoder, self.decoder, self.greedy_dec)
        
        return accuracy