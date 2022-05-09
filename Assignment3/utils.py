from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import wandb

def Masked_CrossEntropy(real, pred):

    cross_entropy = SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask * loss
    loss = tf.reduce_mean(loss)

    return loss

def _plot_predictions(predictions):
    
    # Seperate correctly predicted examples from incorrectly predicted examples
    correct_predictions = predictions[predictions["isAccurate"] == True][["Latin Script", "Predicted Native Script"]].sample(20, replace=True)
    incorrect_predictions = predictions[predictions["isAccurate"] == False][["Latin Script", "Predicted Native Script"]].sample(20, replace=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot a table with correct predictions
    _plot_table_with_predictions(correct_predictions, "Correct Predictions", axes[0])
    
    # Plot a table with incorrect predictions
    _plot_table_with_predictions(incorrect_predictions, "Incorrect Predictions", axes[1])
    
    fig.tight_layout()
    wandb.log({"Correct & Incorrect Predictions": wandb.Image(fig)})
    plt.close(fig)

def _plot_table_with_predictions(predictions, dataset_type, ax):
    
    # Getting the font properties to write in English and Devanagri
    eng_font_prop = fm.FontProperties(fname="/content/Nirmala.ttf")
    lang_font_prop = fm.FontProperties(fname="/content/NotoSansDevanagari-SemiBold.ttf")
    header_color = "#40466e"
    row_colors = ["#f1f1f2", "w"]
    
    # Plotting a table with the data from the dataframe
    table = ax.table(cellText=predictions.values, colWidths=[0.5] * len(predictions.columns), colLabels=predictions.columns[:2], cellLoc="center", rowLoc="center", loc="center",)
    ax.axis("off")
    ax.set_title(dataset_type)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    
    for k, cell in table._cells.items():
        cell.set_edgecolor("w")
        
        # Setting font properties depedning on language
        if k[0] == 1:
            cell.get_text().set_fontproperties(eng_font_prop) #change here 
        else:
            if k[1] == 0:
                cell.get_text().set_fontproperties(eng_font_prop) #change here 
            else:
                cell.get_text().set_fontproperties(lang_font_prop) #change here

        # Table coloring and formating
        if k[0] == 0:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor("#40466e")
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

def _plot_attention_maps(encoder, decoder, greedy_dec, HYPERPARAMS, inputs, dataset_type):
    
    # Sample 9 inputs for plots
    idxs = tf.range(tf.shape(inputs)[0])
    ridxs = tf.random.shuffle(idxs)[:9]
    rand_inputs = tf.gather(inputs, ridxs)

    inference_batch_size = rand_inputs.shape[0]

    # Pass the inputs through the encoder network
    enc_out, states = encoder(rand_inputs)

    # Tile the encoder outputs and hidden states for beam search
    if HYPERPARAMS["attention"]:
        decoder.attention_mechanism.setup_memory(enc_out)
        hidden_state = decoder.build_initial_state(inference_batch_size, states, tf.float32,)
    else:
        if HYPERPARAMS["cell_type"] != "LSTM":
            states = states[0]

        hidden_state = decoder.get_initial_state(states)

        if HYPERPARAMS["n_dec_layers"] == 1:
            hidden_state = (hidden_state,)
        else:
            if HYPERPARAMS["cell_type"] == "LSTM":
                hidden_state = tuple([hidden_state[2 * i], hidden_state[2 * i + 1]] for i in range(int(len(hidden_state) / 2)))
            else:
                hidden_state = tuple(hidden_state)

    # Start and end tokens for beam search
    start_tokens = tf.fill([inference_batch_size], dataset_configs["tgt_lang_tokenizer"].word_index["\t"],)
    end_token = dataset_configs["tgt_lang_tokenizer"].word_index["\n"]

    decoder_embedding_matrix = decoder.embedding.variables[0]

    # Running beam search
    outputs, final_state, sequence_lengths = greedy_dec(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=hidden_state,)

    # Collecting attention weights
    alignments = tf.transpose(final_state.alignment_history.stack(), [1, 2, 0])

    alignments = alignments.numpy()

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    # Getting the inputs and allowed labels
    labels = dataset_configs["updated_"+dataset_type+"_tgts"]
    texts = list(labels.keys())

    outputs = dataset_configs["tgt_lang_tokenizer"].sequences_to_texts(outputs.sample_id.numpy())

    # Getting the font properties to write in English and Devanagri
    eng_font_prop = fm.FontProperties(fname="/content/Nirmala.ttf")
    lang_font_prop = fm.FontProperties(fname="/content/NotoSansDevanagari-SemiBold.ttf")

    rand_idx = ridxs.numpy()
    for i in range(3):
        for j in range(3):
            idx = rand_idx[i * 3 + j]

            text = list(texts[idx])
            pred_output = outputs[i * 3 + j]
            pred_output = list(pred_output[: pred_output.index("\n")].replace(" ", "") if "\n" in pred_output else pred_output.replace(" ", ""))

            # Slicing to get the relevant attention values
            attention_map = alignments[i * 3 + j][8:8+len(text), :len(pred_output)]
            
            # Heatmap
            axes[i][j].pcolor(attention_map.T, cmap=plt.cm.Blues, alpha=0.9)

            xticks = range(0, len(text))
            axes[i][j].set_xticks(xticks, minor=False)
            axes[i][j].set_xticklabels("")

            xticks1 = [k + 0.5 for k in xticks]
            axes[i][j].set_xticks(xticks1, minor=True)
            axes[i][j].set_xticklabels(text, minor=True, fontproperties=eng_font_prop) #change here

            yticks = range(0, len(pred_output))
            axes[i][j].set_yticks(yticks, minor=False)
            axes[i][j].set_yticklabels("")

            yticks1 = [k + 0.5 for k in yticks]
            axes[i][j].set_yticks(yticks1, minor=True)
            axes[i][j].set_yticklabels(pred_output, minor=True, fontproperties=lang_font_prop) #change here

            axes[i][j].grid(True)
            axes[i][j].set_title(''.join(text), fontproperties=eng_font_prop)

    fig.tight_layout()
    wandb.log({"Attention Heatmaps": wandb.Image(fig)})
    plt.close(fig)

def _plot_connectivity(inputs, dataset_type, dataset_configs, HYPERPARAMS, encoder, decoder, greedy_dec):

        # Sample 9 inputs for plots
        idxs = tf.range(tf.shape(inputs)[0])
        ridxs = tf.random.shuffle(idxs)[:9]
        rand_inputs = tf.gather(inputs, ridxs)

        inference_batch_size = rand_inputs.shape[0]

        with tf.GradientTape() as tape:

            # Pass the inputs through the encoder network
            enc_out, states = encoder(rand_inputs)

            # Tile the encoder outputs and hidden states for beam search
            if HYPERPARAMS["attention"]:
                decoder.attention_mechanism.setup_memory(enc_out)
                hidden_state = decoder.build_initial_state(inference_batch_size, states, tf.float32,)
            else:
                if HYPERPARAMS["cell_type"] != "LSTM":
                    states = states[0]

                hidden_state = decoder.get_initial_state(states)

                if HYPERPARAMS["n_dec_layers"] == 1:
                    hidden_state = (hidden_state,)
                else:
                    if HYPERPARAMS["cell_type"] == "LSTM":
                        hidden_state = tuple([hidden_state[2 * i], hidden_state[2 * i + 1]] for i in range(int(len(hidden_state) / 2)))
                    else:
                        hidden_state = tuple(hidden_state)

            # Start and end tokens for beam search
            start_tokens = tf.fill([inference_batch_size], dataset_configs["tgt_lang_tokenizer"].word_index["\t"],)
            end_token = dataset_configs["tgt_lang_tokenizer"].word_index["\n"]

            decoder_embedding_matrix = decoder.embedding.variables[0]

            # Running beam search
            outputs, final_state, sequence_lengths = greedy_dec(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=hidden_state,)

        # Getting the jacobian for each example 
        partial_grads = tape.batch_jacobian(outputs.rnn_output, encoder.encoded_inputs)
        encoder_embedding_matrix = encoder.embedding.variables[0]

        # Computing connectivity (as given in the accompanying blog)
        full_grads = tf.matmul(partial_grads, tf.transpose(encoder_embedding_matrix))
        connectivity = tf.reduce_sum(full_grads ** 2, axis=[2, 4])
        connectivity = tf.transpose(connectivity, [0, 2, 1])
        connectivity = connectivity.numpy()

        fig, axes = plt.subplots(3, 3, figsize=(9, 9))

        # Getting the inputs and allowed labels
        labels = dataset_configs["updated_"+dataset_type+"_tgts"]
        texts = list(labels.keys())

        outputs = dataset_configs["tgt_lang_tokenizer"].sequences_to_texts(outputs.sample_id.numpy())

        # Getting the font properties to write in English and Devanagri
        eng_font_prop = fm.FontProperties(fname="/content/Nirmala.ttf")
        lang_font_prop = fm.FontProperties(fname="/content/NotoSansDevanagari-SemiBold.ttf")

        rand_idx = ridxs.numpy()
        for i in range(3):
            for j in range(3):
                idx = rand_idx[i * 3 + j]

                text = list(texts[idx])
                pred_output = outputs[i * 3 + j]
                pred_output = list(pred_output[: pred_output.index("\n")].replace(" ", "") if "\n" in pred_output else pred_output.replace(" ", ""))

                # Slicing to get plots for the word and normalizing the result
                connectivity_map = connectivity[i * 3 + j][:len(text), :len(pred_output)]
                connectivity_map = connectivity_map / np.sum(connectivity_map, axis=0)

                # Heatmap
                axes[i][j].pcolor(connectivity_map.T, cmap=plt.cm.Blues, alpha=0.9)
 
                xticks = range(0, len(text))
                axes[i][j].set_xticks(xticks, minor=False)
                axes[i][j].set_xticklabels("")

                xticks1 = [k + 0.5 for k in xticks]
                axes[i][j].set_xticks(xticks1, minor=True)
                axes[i][j].set_xticklabels(text, minor=True, fontproperties=eng_font_prop) #change here

                yticks = range(0, len(pred_output))
                axes[i][j].set_yticks(yticks, minor=False)
                axes[i][j].set_yticklabels("")

                yticks1 = [k + 0.5 for k in yticks]
                axes[i][j].set_yticks(yticks1, minor=True)
                axes[i][j].set_yticklabels(pred_output, minor=True, fontproperties=lang_font_prop) #change here

                axes[i][j].grid(True)
                axes[i][j].set_title(''.join(text), fontproperties=eng_font_prop)

        fig.tight_layout()
        wandb.log({"Connectivity": wandb.Image(fig)})
        plt.close(fig)