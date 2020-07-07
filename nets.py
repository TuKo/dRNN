# Definition of networks to be used in experiments
from torch import nn
import torch.nn.functional as F
import torch


class SingleLayerLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None, dropout=None, bias=True, bidi=False,
                 embedding_size=0):
        super(SingleLayerLSTMNet, self).__init__()
        in_size = input_size
        self.embedding = None
        self.embedding_size = embedding_size
        if embedding_size > 0:
            self.embedding = nn.Embedding(input_size, embedding_size)
            in_size = embedding_size
        if output_size is None:
            output_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bidi = bidi
        if dropout is None:
            self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True, bias=bias, bidirectional=bidi)
        else:
            if bidi:
                raise ValueError('biLSTM with dropout is not implemented')
            self.lstm = CustomizedLSTM(in_size, hidden_size, batch_first=True,  weights_dropout=dropout)
        if bidi:
            self.linear = nn.Linear(2*hidden_size, output_size)
        else:
            self.linear = nn.Linear(hidden_size, output_size)

    # def forward(self, input_seq, input_lengths, hidden=None):
    def forward(self, input_seq, hidden=None, redrop=True):
        # Pack padded batch of sequences for RNN module
        # packed = nn.utils.rnn.pack_padded_sequence(input_seq, input_lengths)
        # packed = nn.utils.rnn.pack_padded_sequence(input_seq, input_seq.shape[1])

        # Forward pass on LSTM
        # outputs, hidden = self.lstm(packed, hidden)
        # print(input_seq.shape)

        if self.embedding_size > 0:
            in_seq = self.embedding(input_seq)
        else:
            in_seq = input_seq

        if self.dropout:
            outputs, hidden = self.lstm(in_seq, hidden, redrop=redrop)
        else:
            outputs, hidden = self.lstm(in_seq, hidden)

        # Unpack padding
        # outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # Return output and hidden state
        char_space = self.linear(outputs)
        # char_scores = F.log_softmax(char_space, dim=2)
        # return char_scores, hidden
        return char_space, hidden

    def init_hidden(self, device='cpu'):
        return torch.zeros(1, 1, 2*self.hidden_size, dtype=torch.float, device=device)


class MultiLayerLSTMNet(nn.Module):
    def __init__(self, layers, input_size, hidden_size, output_size=None, dropout=None, bias=True, bidi=False,
                 embedding_size=0):
        super(MultiLayerLSTMNet, self).__init__()
        if layers < 2:
            raise ValueError('Two or more layers required')
        in_size = input_size
        self.embedding = None
        self.embedding_size = embedding_size
        if embedding_size > 0:
            self.embedding = nn.Embedding(input_size, embedding_size)
            in_size = embedding_size
        if output_size is None:
            output_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers=layers, batch_first=True, bias=bias, bidirectional=bidi)
        if bidi:
            self.linear = nn.Linear(2*hidden_size, output_size)
        else:
            self.linear = nn.Linear(hidden_size, output_size)

    # def forward(self, input_seq, input_lengths, hidden=None):
    def forward(self, input_seq, hidden=None, redrop=True):
        if self.embedding_size > 0:
            in_seq = self.embedding(input_seq)
        else:
            in_seq = input_seq

        # Forward pass on LSTM
        outputs, hidden = self.lstm(in_seq, hidden)

        # Return output and hidden state
        char_space = self.linear(outputs)
        return char_space, hidden


class CustomizedLSTM(nn.Module):
    """
    Notes: in order to have the dropout implemented on the weights, we need to
    """
    def __init__(self, *args, weights_dropout=0., input_dropout=0., output_dropout=0., batch_first=True, **kwargs):
        super(CustomizedLSTM, self).__init__()
        self.lstm = nn.LSTM(*args, **kwargs, batch_first=batch_first)
        self.weights_dropout = weights_dropout
        self.input_dropout = input_dropout
        self.output_dropout = output_dropout
        # self.input_drop = ...
        # self.output_drop = ...

        self._init_weights()

        # Get the weights that we should apply weights dropout to and move them to avoid the
        self._new_weights_hh = []
        self._replace_weights_name()

    def _init_weights(self):
        # TODO: Shall we initialize the lstm weights?
        # TODO: How should we initialize the linear layer
        return

    def _replace_weights_name(self):
        param_list = []
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                param_list.append(name)
        for name in param_list:
            # move to the new name
            param = getattr(self.lstm, name)
            self.lstm.register_parameter(name + '_drop', torch.nn.Parameter(param.data))
            self._new_weights_hh.append(name)

    # def init_hidden(self, device='cpu'):
    #     return torch.zeros(1, 1, 2*self.hidden_size, dtype=torch.float, device=device)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for name in self._new_weights_hh:
                param = getattr(self.lstm, name)
        return self

    def _weights_dropout(self):
        for name in self._new_weights_hh:
            param = getattr(self.lstm, name + '_drop')
            drop_param = torch.nn.functional.dropout(param,
                                                     p=self.weights_dropout,
                                                     training=self.training)
            object.__setattr__(self.lstm, name, drop_param)

    def forward(self, input_seq, hidden=None, redrop=True):
        # TODO: apply dropout for input?
        # Forward pass on LSTM
        if redrop:
            self._weights_dropout()
        output, hidden = self.lstm(input_seq, hidden)
        # TODO: apply dropout for output?
        # Return output and hidden state
        return output, hidden


class POSNet(nn.Module):
    def __init__(self, num_chars, char_embedding_dim, char_hidden_size,  words_hidden_size, num_words,
                 word_embedding_dim, num_pos_tags, word_embedding=None, word_delay=0,
                 dropout=None, bias=True, bidi_char=False, bidi_sentence=False, device="cpu"):
        super(POSNet, self).__init__()
        self.device = device
        if bidi_sentence:
            self.words_hidden_size = words_hidden_size * 2
        else:
            self.words_hidden_size = words_hidden_size
        if bidi_char:
            self.char_hidden_size = char_hidden_size * 2
        else:
            self.char_hidden_size = char_hidden_size
        self.pos_tags = num_pos_tags
        self.dropout = dropout
        self.bidi_char = bidi_char
        self.bidi_sentence = bidi_sentence
        self.char_pad_value = 3
        self.word_pad_value = 3
        self.word_delay = word_delay

        if word_embedding is None:
            self.word_emb = nn.Embedding(num_words, word_embedding_dim)
        else:
            self.word_emb = nn.Embedding.from_pretrained(word_embedding)

        self.char_embedding_dim = char_embedding_dim
        self.word_embedding_dim = word_embedding_dim

        self.char_emb = nn.Embedding(num_chars, char_embedding_dim)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_size,
                                 num_layers=1, batch_first=True, bias=bias, bidirectional=bidi_char)
        self.word_lstm = nn.LSTM(word_embedding_dim + self.char_hidden_size, words_hidden_size,
                                 num_layers=1, batch_first=True, bias=bias, bidirectional=bidi_sentence)
        self.linear = nn.Linear(self.words_hidden_size, num_pos_tags)

    def forward(self, sentence, sentence_length, words, words_length):
        # Get the word and process it with char-level lstm
        max_sentence_length = sentence.shape[1]
        max_word_length = words.shape[2]
        max_word_sentence_length = words.shape[1]
        batch_size = sentence.shape[0]

        emb_char = self.char_emb(words)
        emb_char_packed = torch.nn.utils.rnn.pack_padded_sequence(emb_char.view(-1, max_word_length, self.char_embedding_dim),
                                                                  words_length.view(-1),
                                                                  enforce_sorted=False,
                                                                  batch_first=True)
        char_output_packed, _ = self.char_lstm(emb_char_packed)
        char_output, _ = torch.nn.utils.rnn.pad_packed_sequence(char_output_packed,
                                                                batch_first=True,
                                                                padding_value=0,
                                                                total_length=max_word_length)

        last_char_output = char_output[torch.arange(max_word_sentence_length*batch_size), words_length.view(-1)-1, :].\
            view(-1, max_word_sentence_length, self.char_hidden_size)

        char_output = char_output.view(-1, max_word_sentence_length, max_word_length, self.char_hidden_size)[:, :, 0, :]

        # Get the embedding of the word
        emb_output = self.word_emb(sentence)

        # Concatenate the encoded char-level lstm outputs with the embeddings
        if self.bidi_char:
            if self.word_delay > 0:
                # add missing delay words to char output
                last_char_output = torch.cat([last_char_output,
                                              torch.full((batch_size, self.word_delay, self.char_hidden_size),
                                                         self.word_pad_value,
                                                         device=self.device)],
                                             dim=1)
                char_output = torch.cat([char_output,
                                         torch.full((batch_size, self.word_delay, self.char_hidden_size),
                                                    self.word_pad_value,
                                                    device=self.device)],
                                        dim=1)
            emb_both = torch.cat([char_output[:, :, self.char_hidden_size//2:],
                                  last_char_output[:, :, :self.char_hidden_size//2],
                                  emb_output], dim=2)
        else:
            if self.word_delay > 0:
                # add missing delay words to char output
                last_char_output = torch.cat([last_char_output,
                                              torch.full((batch_size, self.word_delay, self.char_hidden_size),
                                                         self.word_pad_value,
                                                         device=self.device)],
                                             dim=1)
            emb_both = torch.cat([last_char_output, emb_output], dim=2)
        emb_both_packed = torch.nn.utils.rnn.pack_padded_sequence(emb_both,
                                                                  sentence_length,
                                                                  enforce_sorted=False,
                                                                  batch_first=True)
        # Get the word-level lstm outputs
        tag_outputs_packed, _ = self.word_lstm(emb_both_packed)
        tag_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(tag_outputs_packed,
                                                                batch_first=True,
                                                                padding_value=0,
                                                                total_length=max_sentence_length)
        # Get the PoS tags
        pos_space = self.linear(tag_outputs)
        return pos_space
