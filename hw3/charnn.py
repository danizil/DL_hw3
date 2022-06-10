import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    import numpy as np
    sorted_text = np.unique(sorted(text))
    char_to_idx = {char: idx for idx, char in enumerate(sorted_text)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    # for i, char in enumerate(sorted_text):
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    n_removed = 0
    text_clean = text
    for char in chars_to_remove:
        n_removed += text_clean.count(char)
        text_clean = text_clean.replace(char, '')
    
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    import numpy as np
    hot_inds = np.vectorize(char_to_idx.get)(list(text))
    result = torch.zeros(len(text), len(char_to_idx), dtype=torch.int8)    
    result[range(len(text)), hot_inds] = 1
    # result = torch.tensor(result, dtype=torch.int8)    

    # for i, char in enumerate(text):
    #     result[i, char_to_idx[char]] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    import numpy as np
    # np_embedded_text = embedded_text.detach.numpy()
    embedded_inds = torch.argmax(embedded_text, dim = 1)
    result_arr = np.vectorize(idx_to_char.get)(embedded_inds)
    result = ''
    result = result.join(result_arr)
    
    #naive implementation to test vectorized implementation
    # result = ''
    # for r in range(embedded_text.shape[0]):
    #     result += idx_to_char[embedded_text[r].argmax().item()]
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create an embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    train_onehot = chars_to_onehot(text[:-1], char_to_idx)
    samples_tup = torch.split(train_onehot, seq_len, dim=0)
    target_onehot_inds = torch.argmax(chars_to_onehot(text[1:], char_to_idx), dim=1)
    labels_tup = torch.split(target_onehot_inds, seq_len, dim=0)
    
    if len(samples_tup[-1]) != len(samples_tup[0]):
        samples_tup = samples_tup[:-1]
        labels_tup = labels_tup[:-1]

    samples = torch.stack(samples_tup, dim=0)
    labels = torch.stack(labels_tup, dim=0)
    
    
    # THIS NEXT PIECE OF CODE WAS A LONG ASS JOURNY BECAUSE I DIDN'T UNDERSTAND THE ASSIGNMENT AND I'M KEEPING
    # IT FOR IF I NEED TO MAKE OVERLAPPING SEQUENCES
    # import numpy as np
    # from numpy.lib.stride_tricks import sliding_window_view

    # # turn the text into big 1hot array
    # text_np = chars_to_onehot(text, char_to_idx).detach().numpy()
    # inds_np = np.argmax(text_np, axis = 1)
    # # make into overlapping sub-sequences of seq_len
    # samples_np = sliding_window_view(text_np, seq_len, axis=0).transpose(0,2,1)
    # inds_np = sliding_window_view(inds_np, seq_len, axis=0)
    # # convert to tensors
    # # all_samples = torch.from_numpy(np.copy(samples_np[:-1, :, :]))
    # # all_inds = torch.from_numpy(np.copy(inds_np[1:,:]), axis = 2)
    # all_samples = samples_np
    # all_inds = inds_np

    # # samples are the first N-1 sequences, labels are the last
    # samples = all_samples[:-1, :, :]
    # labels = all_inds[1:, :]
    # samples = torch.from_numpy(np.copy(samples)).to(device)
    # labels = torch.from_numpy(np.copy(labels)).to(device)

    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    y = y/temperature
    result = torch.softmax(y, dim=dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    #  1. Feed the start_sequence into the model.
    #===============================================
    with torch.no_grad():
        start_sequence = start_sequence
        start_sequence_onehot = chars_to_onehot(start_sequence, char_to_idx).to(device)
        start_sequence_onehot = start_sequence_onehot.unsqueeze(0)

        out = model(start_sequence_onehot)
        out_seq_scores = out[0].squeeze(0)
        out_hidden_state = out[1]

        last_char_scores = out_seq_scores[-1]

        #  2. Sample a new char from the output distribution of the last output
        #     char. Convert output to probabilities first.
        last_char_probs = hot_softmax(last_char_scores, dim=0, temperature=T)
        last_char_ind = torch.multinomial(last_char_probs, 1)
        last_char_onehot = torch.zeros(last_char_probs.shape, device=device)
        last_char_onehot[last_char_ind] = 1
        out_text += idx_to_char[last_char_ind.item()]

        #  3. Feed the new char into the model, rinse and repeat.
        for _ in range(n_chars-len(start_sequence) - 1):
            out = model(last_char_onehot.unsqueeze(0).unsqueeze(0), out_hidden_state)
            out_char_scores = out[0].squeeze(0).squeeze(0)
            out_hidden_state = out[1]

            last_char_probs = hot_softmax(out_char_scores, dim=0, temperature=T)
            last_char_ind = torch.multinomial(last_char_probs, 1)
            last_char_onehot = torch.zeros(last_char_probs.shape, device=device)
            last_char_onehot[last_char_ind] = 1

            out_text += idx_to_char[last_char_ind.item()]


    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset  # dataset is both the source and the target
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: =====
        # XXXXXXX Attempt 1: cut the end of the dataset so that all of the batches fit
        bs = self.batch_size
        idx = [0]*(len(self.dataset) - len(self.dataset) % bs)  # 30 in the example
        nb = len(self.dataset) // bs # num_batches: 3 in the example
        
        for i in range(bs):
            idx[i::bs] = range(i*nb, (i+1)*nb)
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []

        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        
        # initialize first layer from input dimension to hidden dimension
        #=================================================================
        # z_1st_layer = [nn.Linear(in_dim + h_dim, h_dim, bias=True), nn.Sigmoid()]
        # self.add_module('z_0', z_1st_layer[0])
        # self.add_module('z_sig_0', z_1st_layer[1])
        z_1st_layer = [nn.Linear(in_dim, h_dim, bias=True), nn.Linear(h_dim, h_dim, bias=False), nn.Sigmoid()]
        self.add_module('z_xh_0', z_1st_layer[0])
        self.add_module('z_hh_0', z_1st_layer[1])
        self.add_module('z_sig_0', z_1st_layer[2])

        r_1st_layer = [nn.Linear(in_dim, h_dim, bias=True), nn.Linear(h_dim, h_dim, bias=False), nn.Sigmoid()]
        self.add_module('r_xh_0', r_1st_layer[0])
        self.add_module('r_hh_0', r_1st_layer[1])
        self.add_module('r_sig_0', r_1st_layer[2])
        # TODO: pitfall - do all parameters update? did i add module correctly?
        g_1st_layer = [nn.Linear(in_dim, h_dim, bias=True), nn.Linear(h_dim, h_dim, bias=False), nn.Tanh()]
        self.add_module('g_xh_0',g_1st_layer[0])
        self.add_module('g_hh_0',g_1st_layer[1])
        self.add_module('g_tanh_0',g_1st_layer[2])

        # TODO: test with dropout = 0
        dropout_1st = nn.Dropout(dropout)
        # TODO: do i really need to self.add_module?
        self.add_module('dropout_0',dropout_1st)
        first_layer_params = [z_1st_layer, r_1st_layer, g_1st_layer, dropout_1st]
        self.layer_params.append(first_layer_params)
        #===============================
        # END initialize first layer END

        # construct the rest of the layers
        #=================================
        for i in range(1, n_layers):
            z_layer = [nn.Linear(h_dim, h_dim, bias=True), nn.Linear(h_dim, h_dim, bias=False), nn.Sigmoid()]
            self.add_module(f'z_xh_{i}', z_layer[0])
            self.add_module(f'z_hh_{i}', z_layer[1])
            self.add_module(f'z_sig_{i}', z_layer[2])
            r_layer = [nn.Linear(h_dim, h_dim, bias=True), nn.Linear(h_dim, h_dim, bias=False), nn.Sigmoid()]
            self.add_module(f'r_xh_{i}', r_layer[0])
            self.add_module(f'r_hh_{i}', r_layer[1])
            self.add_module(f'r_sig_{i}', r_layer[2])
            g_layer = [nn.Linear(h_dim, h_dim, bias=True), nn.Linear(h_dim, h_dim, bias=False), nn.Tanh()]
            self.add_module(f'g_xh_{i}', g_layer[0])
            self.add_module(f'g_hh_{i}', g_layer[1])
            self.add_module(f'g_tanh_{i}', g_layer[2])
            dropout_layer = nn.Dropout(dropout)
            self.add_module(f'dropout_{i}', dropout_layer)
            layer_params = [z_layer, r_layer, g_layer, dropout_layer]
            self.layer_params.append(layer_params)
        #==================================
        # END construct the rest of the layers END

        # construct the final linear layer
        #==================================
        out_layer = nn.Linear(h_dim, out_dim, bias=True)
        self.add_module('out_layer', out_layer)
        self.layer_params.append(out_layer)
        #==================================
        # END construct the final linear layer END
        
        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for k in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, k, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======

        # initialize hidden state to hidden state, layer output to the expected shape
        #==============================
        
        layer_output = torch.zeros(batch_size, seq_len, self.out_dim, device=input.device) # (B, S, O)
        for t in range(seq_len):
            x_t = layer_input[:, t, :].float()  # (B, I)
            # for each time step do:
            #=======================
            for k in range(self.n_layers):
                # hidden state for layer k for all batches
                #=========================================
                # h_k = h[:, k, :] # (B, H)
                h_k = layer_states[k] # (B, H)
                # TODO: test that the concatenation is in the right order
                
                # chose the layers for ease of reading 
                # ====================================
                z_layers = self.layer_params[k][0]
                r_layers = self.layer_params[k][1]
                g_layers = self.layer_params[k][2]
                dropout_layer = self.layer_params[k][3]
                # TODO: change to call the layers by name 
                
                # find z and r from simple concatenation
                #=========================================
                z_output = z_layers[2](z_layers[0](x_t) + z_layers[1](h_k)) # (B, H)
                r_output = r_layers[2](r_layers[0](x_t) + r_layers[1](h_k)) # (B, H)

                # find g and h using r and z
                #============================
                rh_k = r_output * h_k # (B, H)
                g_layer = g_layers[2](g_layers[0](x_t) + g_layers[1](rh_k)) # (B, H)
                h_k = (1 - z_output) * h_k + z_output * g_layer # (B, H)
                # next layer - h + dropout. next timestep - no dropout
                x_t = dropout_layer(h_k)
                # h[:, k, :] = h_k
                layer_states[k] = h_k
            
            # finally, apply the final W to get the output
            # ==============================================
            layer_output[:, t, :] = self.layer_params[-1](x_t) # (B, O)
        hidden_state = torch.stack(layer_states, dim=1)

                


        # ========================
        return layer_output, hidden_state
