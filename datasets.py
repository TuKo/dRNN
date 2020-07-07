from typing import Counter, Dict

import conllu
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class Text8(Dataset):
    """ text8 dataset """
    """ 
    Metric is bits/character = entropy over test set / number of characters
    this is log2 of ppx
    """
    def __init__(self, dataset, root_dir="./data/", length=180, transform=None, device="cpu", alphabet=None,
                 output_shift=True, delay=0):
        """ Ctor

        :param dataset: selects the part of the dataset {'train','valid','test'}
        :param root_dir: points to the root folder where the "text8" file is [default: './data/']
        :param length: length of each sequence
        :param transform: used to apply transforms to the data
        """
        super(Text8, self).__init__()
        self.dataset = dataset
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        self.output_shift = output_shift
        self.delay = delay

        with open(self.root_dir + '/text8', 'r') as f:
            data = f.read()
        f.close()

        if alphabet is None:
            self.alphabet = {c: i for i, c in enumerate(set(data))}
        else:
            self.alphabet = alphabet.copy()

        # NOTE: we extract an additional character for the output strings.
        if dataset is 'train':
            self.data = torch.tensor(self._str2int(data[:90000000+1]), dtype=torch.long, device=self.device)
        elif dataset is 'valid':
            self.data = torch.tensor(self._str2int(data[90000000:95000000+1]), dtype=torch.long, device=self.device)
        elif dataset is 'test':
            self.data = torch.tensor(self._str2int(data[95000000:]), dtype=torch.long, device=self.device)
        else:
            raise ValueError('Wrong dataset type %s' % (set))

        self.n_chars = len(self.alphabet)
        self.length = length
        self.samples = (len(self.data)-self.delay-1) // length

        print(self.alphabet, len(self.alphabet))

    def __len__(self):
        """ Returns the number of samples in the dataset """
        return self.samples

    def __getitem__(self, item):
        """ Returns the sample #item from the dataset """
        # Compute position for this sample
        pos = item * self.length

        input = self.data[pos:pos + self.length + self.delay].clone().detach()

        # NLL Loss requires ID for the labels (instead of one-hot)
        if self.output_shift:
            output = self.data[pos+1:pos+self.length+1].clone().detach()
        else:
            # input and output are the same, for MLM experiments
            output = self.data[pos:pos + self.length].clone().detach()

        sample = {
            'input': input,
            'output': output
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _str2int(self, text):
        return [self.alphabet[c] for c in text]

    def get_alphabet(self):
        return self.alphabet.copy()


class ReverseData(Dataset):
    """ Reverse sequences dataset """
    """ 
    Metric is cross entropy 
    """
    def __init__(self, dataset, input_classes, sequence_length, train_size,
                 root_dir="./data/reverse/", transform=None, device="cpu"):
        """ Dataset Ctor

        :param dataset: selects the part of the dataset {'train','valid','test'}
        :param root_dir: points to the root folder where the "text8" file is [default: './data/']
        :param transform: used to apply transforms to the data
        """
        super(ReverseData, self).__init__()
        self.dataset = dataset
        self.input_classes = input_classes
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        self.subfilename = 'reverse_%d_i%d_s%d_%s' % (train_size, input_classes, sequence_length, dataset)
        f = np.load(self.root_dir + '/' + self.subfilename + '.npz')
        data = f['data']
        targets = f['targets']
        self.data = torch.tensor(data, dtype=torch.float, device=self.device)
        self.targets = torch.tensor(targets, dtype=torch.long, device=self.device)
        self.input_size = 1
        self.length = data.shape[1]
        self.samples = data.shape[0]

    def __len__(self):
        """ Returns the number of samples in the dataset """
        return self.samples

    def __getitem__(self, item):
        """ Returns the sample #item from the dataset """

        # Add one hot encoding for input
        input = self.data[item]

        # NLL Loss requires ID for the labels (instead of one-hot)
        output = self.targets[item]

        sample = {
            'input': input,
            'output': output
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class SineData(Dataset):
    """ Sin(scale*[conv(x, filter)]) sequences dataset """
    """ 
    Metric to use is MSE 
    """
    def __init__(self, dataset, scale, causality, acausality, sequence_length, train_size,
                 root_dir="./data/sin/", transform=None, device="cpu"):
        """ Dataset Ctor

        :param dataset: selects the part of the dataset {'train','valid','test'}
        :param root_dir: points to the root folder where the "text8" file is [default: './data/']
        :param transform: used to apply transforms to the data
        """
        super(SineData, self).__init__()
        self.dataset = dataset
        self.scale = scale
        self.causality = causality
        self.acausality = acausality
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        self.subfilename = 'sin_%d_s%.2f_%d_%d_s%d_%s' % (train_size, scale, causality, acausality, sequence_length,
                                                          dataset)
        f = np.load(self.root_dir + '/' + self.subfilename + '.npz')
        data = f['data']
        targets = f['targets']
        length = f['seq_length']
        self.data = torch.tensor(data, dtype=torch.float, device=self.device).view((-1, length, 1))
        self.targets = torch.tensor(targets, dtype=torch.float, device=self.device).view((-1, length, 1))
        self.input_size = 1
        self.length = data.shape[1]
        self.samples = data.shape[0]

    def __len__(self):
        """ Returns the number of samples in the dataset """
        return self.samples

    def __getitem__(self, item):
        """ Returns the sample #item from the dataset """

        # Add one hot encoding for input
        input = self.data[item]

        # NLL Loss requires ID for the labels (instead of one-hot)
        output = self.targets[item]

        sample = {
            'input': input,
            'output': output
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class DelayTransform(object):
    """ Transforms input and output for a specific delay d """
    def __init__(self, delay=0, device='cpu'):
        self.delay = delay
        self.device = device

    def __call__(self, sample):
        if self.delay == 0:
            return sample

        in_seq, out_seq = sample['input'], sample['output']
        if len(in_seq.shape) == 1:
            seq_length, = in_seq.shape
            input_seq = torch.zeros((seq_length+self.delay,), dtype=torch.float, device=self.device)
            input_seq[:seq_length] = in_seq
        else:
            seq_length, seq_elems = in_seq.shape
            input_seq = torch.zeros((seq_length+self.delay, seq_elems), dtype=torch.float, device=self.device)
            input_seq[:seq_length, :] = in_seq

        return {'input': input_seq,
                'output': out_seq}


class UD(Dataset):
    """ Universal Dependencies dataset """
    """ 
    Metric is accuracy
    """
    def __init__(self, dataset, language="en", root_dir="./data/",
                 transform=None, device="cpu", alphabet_size: int = 100,
                 max_word_length: int = 75):
        super(UD, self).__init__()
        self.dataset = dataset
        self.language = language
        self.max_word_length = max_word_length
        root_dir = Path(root_dir)
        self.root_dir = root_dir
        self.transform = transform
        self.device = device

        # NOTE: we extract an additional character for the output strings.
        if dataset is 'train':
            pass
        elif dataset is 'valid':
            dataset = "dev"
        elif dataset is 'test':
            pass
        else:
            raise ValueError('Wrong dataset type %s' % (set))

        if language == "en":
            self._data_dir = "UD_English-EWT"
            self._conllu_file = f"en_ewt-ud-{dataset}.conllu"
        elif language == "fr":
            self._data_dir = "UD_French-GSD"
            self._conllu_file = f"fr_gsd-ud-{dataset}.conllu"
        elif language == "de":
            self._data_dir = "UD_German-GSD"
            self._conllu_file = f"de_gsd-ud-{dataset}.conllu"
        elif language == "ja":
            self._data_dir = "UD_Japanese-GSD"
            self._conllu_file = f"ja_gsd-ud-{dataset}.conllu"
        else:
            raise ValueError(f"Unknown language {language}")
        with open(root_dir / self._data_dir / self._conllu_file) as fin:
            self._data = list(conllu.parse_incr(fin))

        character_frequencies: Counter[str] = Counter()
        self.word2index: Dict[str, int] = {}
        vocabulary = np.load(root_dir / f"polyglot-{language}.npz")
        self.words = vocabulary["words"]
        self.embeddings = vocabulary["embeddings"]
        for i, word in enumerate(self.words):
            self.word2index[word] = i
            for character in word:
                character_frequencies[character] += 1

        # 4 non-words in vocabulary
        most_common = character_frequencies.most_common(alphabet_size - 4)
        self.alphabet = list(self.words[:4]) + [i[0] for i in most_common]
        self.char2index: Dict[str, int] = {}
        for i, char in enumerate(self.alphabet):
            self.char2index[char] = i

        sentence_lengths = [len(s)+2 for s in self._data]
        self.max_sentence_length = np.max(sentence_lengths)
        self.sentence_lengths = torch.tensor(sentence_lengths,
                                             dtype=torch.long,
                                             device=self.device)

        with open(root_dir / "cpos.ud") as fin:
            self.pos_tags = [line.strip() for line in fin]
        self.pos_tags.append("_")
        self.pos2index: Dict[str, int] = {}
        for i, pos in enumerate(self.pos_tags):
            self.pos2index[pos] = i

        self.samples = len(self._data)

        self.sentences = torch.zeros((self.samples, self.max_sentence_length),
                                     dtype=torch.long, device=self.device)
        self.targets = torch.zeros((self.samples, self.max_sentence_length),
                                   dtype=torch.long, device=self.device)
        self.word_lengths = torch.ones(
            (self.samples, self.max_sentence_length),
            dtype=torch.long, device=self.device)
        # self.characters = torch.zeros(
        #     (self.samples, self.max_sentence_length, self.max_word_length),
        #     dtype=torch.long, device=self.device)
        self.characters = torch.ones(
            (self.samples, self.max_sentence_length, self.max_word_length),
            dtype=torch.long, device=self.device) * 3

        self._word_count = 0
        for i, sentence in enumerate(self._data):
            # 1 means start
            self.sentences[i, 0] = 1
            self.characters[i, 0, 0] = 1
            # 2 means end
            self.characters[i, 0, 1] = 2
            self.word_lengths[i, 0] = 2
            self._word_count += len(sentence)
            for j, word_data in enumerate(sentence, 1):
                word = word_data["form"]
                # 0 means unknown
                word_idx = self.word2index.get(word, 0)
                self.sentences[i, j] = word_idx
                self.targets[i, j] = self.pos2index[word_data["upostag"]]
                word_length = min(len(word) + 2, 75)
                self.word_lengths[i, j] = word_length
                self.characters[i, j, 0] = 1
                for k in range(1, word_length-1):
                    character = word[k-1]
                    character_idx = self.char2index.get(character, 0)
                    self.characters[i, j, k] = character_idx
                self.characters[i, j, k+1] = 2
            self.sentences[i, j+1] = 2
            self.characters[i, j+1, 0] = 1
            self.characters[i, j+1, 1] = 2
            self.word_lengths[i, j+1] = 2

    def __len__(self):
        """ Returns the number of samples in the dataset """
        return len(self._data)

    def __getitem__(self, item):
        """ Returns the sample #item from the dataset """
        sen_length = self.sentence_lengths[item].clone().detach()
        input_sen = self.sentences[item, :].clone().detach()
        words_length = self.word_lengths[item, :].clone().detach()
        input_chars = self.characters[item, :, :].clone().detach()

        # NLL Loss requires ID for the labels (instead of one-hot)
        output = self.targets[item, :].clone().detach()

        sample = {
            'input_sentence': input_sen,
            'input_chars': input_chars,
            'sen_length': sen_length,
            'words_length': words_length,
            'output': output
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_num_chars(self):
        return len(self.alphabet)

    def get_num_words(self):
        return len(self.words)

    def get_num_pos_tags(self):
        return len(self.pos_tags)

    def get_embeddings(self):
        return torch.tensor(self.embeddings, dtype=torch.float,
                            device=self.device)

    def count_words_in_treebank(self):
        """Count words in treebank."""
        return self._word_count


class POSDelayTransform(object):
    """ Transforms input and output for a specific delay d """
    def __init__(self, char_delay=0, word_delay=0, char_pad_id=3, word_pad_id=3, device='cpu'):
        self.char_delay = char_delay
        self.word_delay = word_delay
        self.char_pad_id = char_pad_id
        self.word_pad_id = word_pad_id
        self.device = device

    def __call__(self, sample):
        if self.word_delay == 0 and self.char_delay == 0:
            return sample

        in_sen = sample['input_sentence']
        sen_length = sample['sen_length'].clone().detach()
        length = in_sen.shape[0]

        if self.word_delay > 0:
            input_sen = torch.zeros((length+self.word_delay, ), dtype=torch.long, device=self.device)
            input_sen[:length] = in_sen
            input_sen[length:length+self.word_delay] = self.word_pad_id
            sen_length += self.word_delay
        else:
            input_sen = in_sen

        in_char = sample['input_chars']
        words_length = sample['words_length'].clone().detach()
        if self.char_delay > 0:
            input_char = torch.cat([in_char, torch.zeros((length, self.char_delay),
                                                         dtype=torch.long,
                                                         device=self.device)], dim=1)
            words_length += self.char_delay
        else:
            input_char = in_char

        return {
            'input_sentence': input_sen,
            'input_chars': input_char,
            'sen_length': sen_length,
            'words_length': words_length,
            'output': sample['output']
        }
