from tensorflow.keras.layers import Embedding, LSTM, LSTMCell, GRU, GRUCell, SimpleRNN, SimpleRNNCell, StackedRNNCells, Dense
from tensorflow_addons.seq2seq import BahdanauAttention, AttentionWrapper, BasicDecoder
from tensorflow_addons.seq2seq.sampler import TrainingSampler
from tensorflow.keras import Model

import tensorflow as tf
import numpy as np
import wandb


class Encoder(Model):
    def __init__(self, dataset_configs, HYPERPARAMS):
        super(Encoder, self).__init__()

        self.dataset_configs = dataset_configs

        # Layer to convert inputs into embeddings
        self.embedding = Embedding(dataset_configs["src_vocab_size"], HYPERPARAMS["emb_dim"])

        # Using either LSTM, GRU or Simple RNN as the recurrent layer
        if HYPERPARAMS["cell_type"] == "LSTM":
            rec_func = LSTM
        elif HYPERPARAMS["cell_type"] == "GRU":
            rec_func = GRU
        else:
            rec_func = SimpleRNN

        # Stacking the recurrent layers
        self.recurrent_layers = {}
        for i in range(HYPERPARAMS["n_enc_layers"]):
            self.recurrent_layers[i] = rec_func(HYPERPARAMS['hid_st_dim'], dropout=HYPERPARAMS["dropout"],
                                                recurrent_dropout=HYPERPARAMS["dropout"], return_sequences=True,
                                                return_state=True)

    def call(self, x):

        # Passing inputs through embedding layer
        word_embedding = self.embedding(x)

        x = word_embedding
        # Passing the embeddings through all recurrent layers
        for i in range(HYPERPARAMS["n_enc_layers"]):
            x, *states = self.recurrent_layers[i](x)

        return x, states


class Decoder(Model):
    def __init__(self, config, dataset_configs, HYPERPARAMS):
        super(Decoder, self).__init__()

        self.config = config

        # Layer to convert inputs into embeddings
        self.embedding = Embedding(dataset_configs["tgt_vocab_size"], HYPERPARAMS["emb_dim"])

        # Output layer
        self.fc = Dense(dataset_configs["tgt_vocab_size"])

        # Using either LSTMCell, GRU or Simple RNN as the recurrent unit
        if HYPERPARAMS["cell_type"] == "LSTM":
            rec_func = LSTMCell
        elif HYPERPARAMS["cell_type"] == "GRU":
            rec_func = GRUCell
        else:
            rec_func = SimpleRNNCell

        # Decoder RNN - single cell to sample every time step
        rnn_cells = [
            rec_func(HYPERPARAMS['hid_st_dim'])
            for _ in range(HYPERPARAMS["n_dec_layers"])
        ]
        if HYPERPARAMS["attention"]:
            rnn_cell = StackedRNNCells(rnn_cells)
            self.attention_mechanism = BahdanauAttention(
                units=HYPERPARAMS['hid_st_dim'],
                memory=None,
                memory_sequence_length=HYPERPARAMS["batch_size"]
                * [dataset_configs["max_tgt_len"]],
            )
            self.rnn_cell = AttentionWrapper(
                rnn_cell,
                self.attention_mechanism,
                attention_layer_size=HYPERPARAMS['hid_st_dim'],
                alignment_history=True,
            )
        else:
            self.rnn_cell = StackedRNNCells(rnn_cells)

        # Sampler to sample from softmax output at every time step
        self.sampler = TrainingSampler()

        # Decoder that helps get the output of the network (all time steps)
        self.decoder = BasicDecoder(
            self.rnn_cell, sampler=self.sampler, output_layer=self.fc
        )

    def build_initial_state(self, batch_size, encoder_state, Dtype):
        """
        This function is to be used when attention is used as we need an attention wrapper state.
        """

        # Building the initial state for the decoder based on number of layer and cell type
        if self.HYPERPARAMS['n_dec_layers'] == 1:
            if self.HYPERPARAMS["cell_type"] == "LSTM":
                encoder_state = (encoder_state,)
            else:
                encoder_state = tuple(encoder_state)
        else:
            if self.HYPERPARAMS["cell_type"] == "LSTM":
                # If there are multiple decoder layers, feed encoder state as initial state to all layers
                states = []
                for i in range(self.HYPERPARAMS['n_dec_layers']):
                    states += [encoder_state]

                encoder_state = tuple(states)
            else:
                # If there are multiple decoder layers, feed encoder state as initial state to all layers
                states = []
                for i in range(self.HYPERPARAMS['n_dec_layers']):
                    states += encoder_state

                encoder_state = tuple(states)

        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_size, dtype=Dtype
        )

        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

        return decoder_initial_state

    def get_initial_state(self, encoder_states):
        """
        This function is to be used when attention is not used.
        """

        # Building the initial state for the decoder based on number of layer and cell type
        if self.HYPERPARAMS['n_dec_layers'] == 1:
            states = encoder_states
        else:
            # If there are multiple decoder layers, feed encoder state as initial state to all layers
            if self.HYPERPARAMS["cell_type"] == "LSTM":
                states = []
                for i in range(self.HYPERPARAMS['n_dec_layers']):
                    states += encoder_states
            else:
                states = [
                    encoder_states
                    for i in range(self.HYPERPARAMS['n_dec_layers'])
                ]

        return states

    def call(self, inputs, initial_state):

        # Passing inputs through embedding layer
        x = self.embedding(inputs)

        # Passing the embeddings through the decoder object to get the outputs
        outputs, *states = self.decoder(
            x,
            initial_state=initial_state,
            sequence_length=self.HYPERPARAMS["batch_size"]
            * [self.dataset_configs["max_length_output"] - 1],
        )

        return outputs