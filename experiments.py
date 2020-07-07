from datasets import *
from nets import *
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time


# Definition of experiments
class Experiment(object):
    def __init__(self, name=''):
        self.name_ = name

    @property
    def name(self):
        return self.name_

    @name.setter
    def name(self, value):
        self.name_ = value


class ReverseExperiment(Experiment):
    def __init__(self, dataset_setup, model_layers, delay: int, bidi=False,
                 batch_size=128, lr=1e-3, lr_schedule=None, max_epochs=50, rnn_units=100,
                 weight_decay=0, dropout=None, patience=0, data_dir='./data/reverse/',
                 name='', checkpoint_dir=None, device="cpu"):
        super(ReverseExperiment, self).__init__(name)
        self.delay = delay
        self.model_layers = model_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = device
        self.rnn_units = rnn_units
        self.dropout = dropout
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.weight_decay = weight_decay
        self.patience = patience
        self.bidi = bidi
        self.seq_length = None
        self.model = None
        self.dataset_setup = dataset_setup
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.total_params = 0
        self.early = {'wait': 0, 'best_loss': 1e15, 'min_delta': 1e-3}
        if self.bidi:
            self.delay = 0
            self.model_layers = 1
        elif self.delay > 0:
            self.model_layers = 1

    def preload_data(self, set_type):
        dataset = ReverseData(set_type,
                              self.dataset_setup['input_classes'],
                              self.dataset_setup['sequence_length'],
                              self.dataset_setup['train_size'],
                              root_dir=self.data_dir,
                              device=self.device,
                              transform=DelayTransform(self.delay, device=self.device))
        if self.seq_length is None:
            self.seq_length = dataset.length
        return dataset

    def model_setup(self):
        # Setup a network based on experiment setup
        model = None
        if self.model_layers > 1:
            # This should be a stacked-LSTM
            model = MultiLayerLSTMNet(self.model_layers,
                                      self.dataset_setup['input_classes'],
                                      self.rnn_units,
                                      self.dataset_setup['input_classes'],
                                      dropout=self.dropout).to(self.device)
        else:
            model = SingleLayerLSTMNet(self.dataset_setup['input_classes'],
                                       self.rnn_units,
                                       self.dataset_setup['input_classes'],
                                       bidi=self.bidi,
                                       dropout=self.dropout).to(self.device)
        loss_function = nn.NLLLoss().to(self.device)
        if self.weight_decay == 0:
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return model, loss_function, optimizer

    def save_model(self, directory):
        torch.save({
            'model': self.model.state_dict(),
            'delay': self.delay,
            'model_layers': self.model_layers,
            'bidi': self.bidi,
            'seq_length': self.seq_length,
        }, directory + '/model_weights.pt')

    def save_checkpoint(self, epoch, optimizer, loss, results):
        if self.checkpoint_dir is None:
            return
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'delay': self.delay,
            'model_layers': self.model_layers,
            'bidi': self.bidi,
            'seq_length': self.seq_length,
            'results': results,
            'wait': self.early['wait'],
        }, self.checkpoint_dir + '/checkpoint_epoch_' + str(epoch) + '.tar')

    def check_early_stop(self, current_loss):
        # Early stopping disabled?
        if self.patience <= 0:
            return False

        if current_loss - self.early['best_loss'] < - self.early['min_delta']:
            self.early['best_loss'] = current_loss
            self.early['wait'] = 1
        else:
            if self.early['wait'] > self.patience:
                return True
            self.early['wait'] += 1
        return False

    def run(self):
        dataset = self.preload_data("train")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        dataset_val = self.preload_data("valid")
        dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=0)

        model, loss_function, optimizer = self.model_setup()
        self.model = model
        self.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('#Parameters', self.total_params)

        # Add the learning rate scheduler
        if self.lr_schedule:
           scheduler = self.schedule = ReduceLROnPlateau(optimizer,
                                                         patience=3,
                                                         factor=self.lr_schedule,
                                                         threshold=1e-2,
                                                         threshold_mode='rel',
                                                         eps=1e-5,
                                                         verbose=True)

        results = {'loss': [],
                   'val_loss': [],
                   'acc_val': [],
                   'acc_test': 1e10,
                   }
        clip = 1.0
        for epoch in range(self.max_epochs):
            # Training
            print('Starting Epoch', epoch)
            train_loss = []
            for i_batch, sample_batched in enumerate(dataloader):
                source_orig, targets = sample_batched['input'], sample_batched['output']
                # NOTE: We don't need to reset the initial hidden state because the default is to use zero for c0 and h0
                model.zero_grad()
                output, _ = model(source_orig)
                if self.delay == 0:
                    char_scores = output
                else:
                    char_scores = output[:, self.delay:, :]
                total_loss = loss_function(F.log_softmax(char_scores, dim=2).permute([0, 2, 1]), targets)
                total_loss.backward()
                _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                train_loss.append(total_loss.item())
                print('Batch', i_batch, 'Loss:', total_loss.item(), 'mean loss', sum(train_loss) / (i_batch+1))

            # Validation
            model.eval()
            total_acc, total_items, total_val_loss = self.eval_model(dataloader_val, loss_function)
            results['acc_val'].append(total_acc / total_items)
            results['val_loss'].append(total_val_loss)
            results['loss'].append(train_loss)
            print('Validation ACC: ', total_acc / total_items, ' (out of total_items:', total_items, ')')
            print('Validation loss: ', total_val_loss)
            model.train()

            # Save a checkpoint for reference
            self.save_checkpoint(epoch, optimizer, loss_function, results)

            # Check LR scheduler
            if self.lr_schedule:
                scheduler.step(total_val_loss)

            # Early stopping?
            if self.check_early_stop(total_val_loss):
                print('Early stopping at epoch %d...' % (epoch))
                break

        # Test the trained model
        model.eval()
        dataset_test = self.preload_data("test")
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=0)
        total_acc, total_items, _ = self.eval_model(dataloader_test, loss_function)
        print('Test ACC: ', total_acc / total_items, ' (out of total_items:', total_items, ')')
        results['acc_test'] = total_acc / total_items
        return results

    def eval_model(self, dataloader, loss_function):
        total_acc = 0.0
        total_items = 0.0
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                source, targets = sample_batched['input'], sample_batched['output']
                batch_size = source.size(0)
                # Predict for this batch
                output, _ = self.model(source)
                # Compute Accuracy for the batch
                # NOTE: Softmax missing at the output of the network
                if self.delay == 0:
                    char_scores = output
                else:
                    char_scores = output[:, self.delay:, :]
                # use categorical sampling to predict the output of the network. This is in case the network cannot
                # predict a value with a higher chance.
                pred_cat = torch.distributions.categorical.Categorical(logits=char_scores)
                predictions = pred_cat.sample()
                acc = (predictions == targets).sum().item()
                val_loss = loss_function(F.log_softmax(char_scores, dim=2).permute([0, 2, 1]), targets)
                total_loss += val_loss.item()
                total_acc += acc
                total_items += batch_size * self.seq_length
                total_batches += 1
        return total_acc, total_items, total_loss / total_batches


class SineExperiment(Experiment):
    def __init__(self, dataset_setup, model_layers, delay: int, bidi=False,
                 batch_size=128, lr=1e-3, lr_schedule=None, max_epochs=50, rnn_units=100,
                 weight_decay=0, dropout=None, patience=0, data_dir='./data/sin/',
                 name='', checkpoint_dir=None, device="cpu"):
        super(SineExperiment, self).__init__(name)
        self.delay = delay
        self.model_layers = model_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = device
        self.rnn_units = rnn_units
        self.dropout = dropout
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.weight_decay = weight_decay
        self.patience = patience
        self.bidi = bidi
        self.seq_length = None
        self.model = None
        self.dataset_setup = dataset_setup
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.total_params = 0
        self.early = {'wait': 0, 'best_loss': 1e15, 'min_delta': 1e-2}
        if self.bidi:
            self.delay = 0
            self.model_layers = 1
        elif self.delay > 0:
            self.model_layers = 1

    def preload_data(self, set_type):
        dataset = SineData(set_type,
                           self.dataset_setup['scale'],
                           self.dataset_setup['causality'],
                           self.dataset_setup['acausality'],
                           self.dataset_setup['sequence_length'],
                           self.dataset_setup['train_size'],
                           root_dir=self.data_dir,
                           device=self.device,
                           transform=DelayTransform(self.delay, device=self.device))
        if self.seq_length is None:
            self.seq_length = dataset.length
        return dataset

    def model_setup(self):
        model = None
        if self.model_layers > 1:
            # This should be a stacked-LSTM
            model = MultiLayerLSTMNet(self.model_layers,
                                      1,
                                      self.rnn_units,
                                      1,
                                      bidi=self.bidi,
                                      dropout=self.dropout).to(self.device)
        else:
            model = SingleLayerLSTMNet(1,
                                       self.rnn_units,
                                       1,
                                       bidi=self.bidi,
                                       dropout=self.dropout).to(self.device)
        loss_function = nn.MSELoss().to(self.device)
        if self.weight_decay == 0:
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return model, loss_function, optimizer

    def save_model(self, directory):
        torch.save({
            'model': self.model.state_dict(),
            'delay': self.delay,
            'model_layers': self.model_layers,
            'bidi': self.bidi,
        }, directory + '/model_weights.pt')

    def save_checkpoint(self, epoch, optimizer, loss, results):
        if self.checkpoint_dir is None:
            return
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'delay': self.delay,
            'model_layers': self.model_layers,
            'bidi': self.bidi,
            'seq_length': self.seq_length,
            'results': results,
            'wait': self.early['wait'],
        }, self.checkpoint_dir + '/checkpoint_epoch_' + str(epoch) + '.tar')

    def check_early_stop(self, current_loss):
        # Early stopping disabled?
        if self.patience <= 0:
            return False

        if current_loss - self.early['best_loss'] < - self.early['min_delta']:
            self.early['best_loss'] = current_loss
            self.early['wait'] = 1
        else:
            if self.early['wait'] > self.patience:
                return True
            self.early['wait'] += 1
        return False

    def run(self):
        dataset = self.preload_data("train")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        dataset_val = self.preload_data("valid")
        dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=0)

        model, loss_function, optimizer = self.model_setup()
        self.model = model
        self.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('#Parameters', self.total_params)

        # Add the learning rate scheduler
        if self.lr_schedule:
           scheduler = self.schedule = ReduceLROnPlateau(optimizer,
                                                         patience=3,
                                                         factor=self.lr_schedule,
                                                         threshold=1e-2,
                                                         threshold_mode='rel',
                                                         eps=1e-5,
                                                         verbose=True)

        results = {'loss': [],
                   'val_loss': [],
                   'mse_val': [],
                   'mse_test': 1e10,
                   }
        clip = 1.0
        for epoch in range(self.max_epochs):
            # Training
            print('Starting Epoch', epoch)
            train_loss = []
            for i_batch, sample_batched in enumerate(dataloader):
                source_orig, targets = sample_batched['input'], sample_batched['output']
                # NOTE: We don't need to reset the initial hidden state because the default is to use zero for c0 and h0
                model.zero_grad()
                output, _ = model(source_orig)
                if self.delay == 0:
                    filtered_output = output
                else:
                    filtered_output = output[:, self.delay:, :]
                total_loss = loss_function(filtered_output, targets)
                total_loss.backward()
                _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                train_loss.append(total_loss.item())
                print('Batch', i_batch, 'Loss:', total_loss.item(), 'mean loss', sum(train_loss) / (i_batch+1))

            # Validation
            model.eval()
            total_mse, total_items, total_val_loss = self.eval_model(dataloader_val, loss_function)
            results['mse_val'].append(total_mse / total_items)
            results['val_loss'].append(total_val_loss)
            results['loss'].append(train_loss)
            print('Validation MSE: ', total_mse / total_items, ' (out of total_items:', total_items, ')')
            print('Validation loss: ', total_val_loss)
            model.train()

            # Save a checkpoint for reference
            self.save_checkpoint(epoch, optimizer, loss_function, results)

            # Check LR scheduler
            if self.lr_schedule:
                scheduler.step(total_val_loss)

            # Early stopping?
            if self.check_early_stop(total_val_loss):
                print('Early stopping at epoch %d...' % (epoch))
                break

            # check for total convergence
            if total_val_loss < 1e-5:
                print('Automatic stopping due to MSE error is zero')
                break

        # Test the trained model
        model.eval()
        dataset_test = self.preload_data("test")
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=0)
        total_mse, total_items, _ = self.eval_model(dataloader_test, loss_function)
        print('Test MSE: ', total_mse / total_items, ' (out of total_items:', total_items, ')')
        results['mse_test'] = total_mse / total_items
        return results

    def eval_model(self, dataloader, loss_function):
        total_acc = 0.0
        total_items = 0.0
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                source, targets = sample_batched['input'], sample_batched['output']
                batch_size = source.size(0)
                # Predict for this batch
                output, _ = self.model(source)
                # Compute Accuracy for the batch
                # NOTE: Softmax missing at the output of the network
                if self.delay == 0:
                    filtered_output = output
                else:
                    filtered_output = output[:, self.delay:, :]
                acc = ((filtered_output - targets)**2).sum().item()
                val_loss = loss_function(filtered_output, targets)
                total_loss += val_loss.item()
                total_acc += acc
                total_items += batch_size * self.seq_length
                total_batches += 1
        return total_acc, total_items, total_loss / total_batches


class POSExperiment(Experiment):
    def __init__(self, language, char_delay, char_units, char_embeddings, word_delay, word_units, word_embeddings,
                 pretrained_word_embeddings=False,
                 model_layers=1, bidi_char=False, bidi_sentence=False,
                 batch_size=128, lr=1e-3, lr_schedule=None, max_epochs=50,
                 weight_decay=0, dropout=None, patience=0, data_dir='./data/',
                 name='', checkpoint_dir=None, device="cpu"):
        super(POSExperiment, self).__init__(name)
        self.char_delay = char_delay
        self.word_delay = word_delay
        self.char_units = char_units
        self.word_units = word_units
        self.char_embeddings_dim = char_embeddings
        self.word_embeddings_dim = word_embeddings
        self.pretrained_word_embeddings = pretrained_word_embeddings

        self.embeddings = None
        self.model = None
        self.model_layers = model_layers

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.dropout = dropout
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.weight_decay = weight_decay
        self.patience = patience
        self.bidi_char = bidi_char
        self.bidi_sentence = bidi_sentence

        self.language = language
        self.device = device
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.total_params = 0
        self.early = {'wait': 0, 'best_loss': 1e15, 'min_delta': 1e-2}
        if self.bidi_char:
            self.char_delay = 0
        if self.bidi_sentence:
            self.word_delay = 0
            self.model_layers = 1
        elif self.word_delay > 0:
            self.model_layers = 1

    def preload_data(self, set_type):
        dataset = UD(set_type,
                     self.language,
                     root_dir=self.data_dir,
                     device=self.device,
                     transform=POSDelayTransform(self.char_delay, self.word_delay, device=self.device))
        self.num_chars = dataset.get_num_chars()
        self.num_words = dataset.get_num_words()
        self.num_pos_tags = dataset.get_num_pos_tags()
        self.embeddings = self.load_embedding(dataset)
        return dataset

    def load_embedding(self, dataset):
        if self.pretrained_word_embeddings:
            embeddings = dataset.get_embeddings()
            self.word_embeddings_dim = embeddings.size(1)
        else:
            embeddings = None
        return embeddings

    def model_setup(self):
        model = POSNet(self.num_chars, self.char_embeddings_dim, self.char_units,
                       self.word_units, self.num_words, self.word_embeddings_dim,
                       self.num_pos_tags, word_embedding=self.embeddings, word_delay=self.word_delay,
                       bidi_char=self.bidi_char, bidi_sentence=self.bidi_sentence,
                       device=self.device).to(self.device)
        loss_function = nn.NLLLoss().to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        return model, loss_function, optimizer

    def save_model(self, directory):
        torch.save({
            'model': self.model.state_dict(),
            'char_delay': self.char_delay,
            'word_delay': self.word_delay,
            'model_layers': self.model_layers,
            'bidi_char': self.bidi_char,
            'bidi_sentence': self.bidi_sentence,
        }, directory + '/model_weights.pt')

    def save_checkpoint(self, epoch, optimizer, loss, results):
        if self.checkpoint_dir is None:
            return
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'char_delay': self.char_delay,
            'word_delay': self.word_delay,
            'model_layers': self.model_layers,
            'bidi_char': self.bidi_char,
            'bidi_sentence': self.bidi_sentence,
            'results': results,
            'wait': self.early['wait'],
        }, self.checkpoint_dir + '/checkpoint_epoch_' + str(epoch) + '.tar')

    def check_early_stop(self, current_loss):
        # Early stopping disabled?
        if self.patience <= 0:
            return False

        if current_loss - self.early['best_loss'] < - self.early['min_delta']:
            self.early['best_loss'] = current_loss
            self.early['wait'] = 1
        else:
            if self.early['wait'] > self.patience:
                return True
            self.early['wait'] += 1
        return False

    def run(self):
        dataset = self.preload_data("train")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        dataset_val = self.preload_data("valid")
        dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=0)

        model, loss_function, optimizer = self.model_setup()
        self.model = model
        self.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('#Parameters', self.total_params)

        # Add the learning rate scheduler
        if self.lr_schedule:
           scheduler = self.schedule = ReduceLROnPlateau(optimizer,
                                                         patience=3,
                                                         factor=self.lr_schedule,
                                                         threshold=1e-2,
                                                         threshold_mode='rel',
                                                         eps=1e-5,
                                                         verbose=True)

        results = {'loss': [],
                   'val_loss': [],
                   'acc_val': [],
                   'acc_test': 1e10,
                   }
        clip = 1.0
        for epoch in range(self.max_epochs):
            # Training
            print('Starting Epoch', epoch)
            train_loss = []
            for i_batch, sample_batched in enumerate(dataloader):
                sentence = sample_batched['input_sentence']
                words = sample_batched['input_chars']
                targets = sample_batched['output']
                sentence_length = sample_batched['sen_length']
                words_length = sample_batched['words_length']
                batch_size = sentence_length.size(0)
                model.zero_grad()
                output = model(sentence, sentence_length, words, words_length)
                output = F.log_softmax(output, dim=2).permute([0, 2, 1])
                # Need to do softmax based on the length of each sentence.
                hidden_dim = output.size(1)

                sen_length = sentence_length[0] - self.word_delay - 1
                total_loss = loss_function(output[0, :, self.word_delay+1:sentence_length[0]-1].view(1, hidden_dim, -1),
                                           targets[0, 1:sen_length].view(1, -1))
                for s in range(1, batch_size):
                    sen_length = sentence_length[s] - self.word_delay - 1
                    total_loss += loss_function(output[s, :, self.word_delay+1:sentence_length[s]-1].view(1, hidden_dim, -1),
                                                targets[s, 1:sen_length].view(1, -1))

                total_loss.backward()
                _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                train_loss.append(total_loss.item())
                print('Batch', i_batch, 'Loss:', total_loss.item(), 'mean loss', sum(train_loss) / (i_batch+1))

            # Validation
            model.eval()
            total_acc, total_items, total_val_loss = self.eval_model(dataloader_val, loss_function)
            results['acc_val'].append(total_acc / total_items)
            results['val_loss'].append(total_val_loss)
            results['loss'].append(train_loss)
            print('Validation ACC: ', total_acc / total_items, ' (out of total_items:', total_items, ')')
            print('Validation loss: ', total_val_loss)
            model.train()

            # Save a checkpoint for reference
            self.save_checkpoint(epoch, optimizer, loss_function, results)

            # Check LR scheduler
            if self.lr_schedule:
                scheduler.step(total_val_loss)

            # Early stopping?
            if self.check_early_stop(total_val_loss):
                print('Early stopping at epoch %d...' % (epoch))
                break

        # Test the trained model
        model.eval()
        dataset_test = self.preload_data("test")
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=0)
        total_acc, total_items, _ = self.eval_model(dataloader_test, loss_function)
        print('Test ACC: ', total_acc / total_items, ' (out of total_items:', total_items, ')')
        results['acc_test'] = total_acc / total_items
        return results

    def eval_model(self, dataloader, loss_function):
        total_acc = 0.0
        total_items = 0.0
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                sentence = sample_batched['input_sentence']
                words = sample_batched['input_chars']
                targets = sample_batched['output']
                sentence_length = sample_batched['sen_length']
                words_length = sample_batched['words_length']
                batch_size = sentence.size(0)
                # Predict for this batch
                output = self.model(sentence, sentence_length, words, words_length)
                # Compute Accuracy for the batch
                # NOTE: Softmax missing at the output of the network
                filtered_output = output

                output = F.log_softmax(filtered_output, dim=2).permute([0, 2, 1])
                hidden_dim = output.size(1)
                _, predictions = torch.max(filtered_output, 2)
                acc = 0.0
                for s in range(batch_size):
                    sen_length = sentence_length[s] - self.word_delay - 1
                    acc += (predictions[s, self.word_delay+1:sentence_length[s]-1] == targets[s, 1:sen_length]).sum().item()
                    total_loss += loss_function(output[s, :, self.word_delay+1:sentence_length[s]-1].view(1, hidden_dim, -1),
                                                targets[s, 1:sen_length].view(1, -1)).item()
                total_acc += acc
                total_items += (sentence_length - self.word_delay - 2).sum().item()
                total_batches += 1
        return total_acc, total_items, total_loss / total_batches


class MLMExperiment(Experiment):
    def __init__(self, dataset_setup, layers, delay: int, bidi=False, seq_length=180,
                 batch_size=32, lr=1e-3, lr_schedule=None, max_epochs=50, rnn_units=100, embedding_size=10,
                 weight_decay=0, dropout=None, patience=0, data_dir='./data/',
                 name='', checkpoint_dir=None, device="cpu"):
        super(MLMExperiment, self).__init__(name)
        self.delay = delay
        self.model_layers = layers
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.device = device
        self.rnn_units = rnn_units
        self.dropout = dropout
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.weight_decay = weight_decay
        self.patience = patience
        self.bidi = bidi
        self.seq_length = seq_length
        self.alphabet = None
        self.model = None
        self.dataset_setup = dataset_setup
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.total_params = 0
        self.early = {'wait': 0, 'best_loss': 1e15, 'min_delta': 1e-2}
        if self.bidi or self.model_layers > 1:
            self.delay = 0
        if self.delay > 0:
            self.model_layers = 1
        # percentage of masked elements
        self.p = 0.2

    def preload_data(self, set_type):
        dataset = Text8(set_type,
                        root_dir=self.data_dir,
                        length=self.seq_length,
                        alphabet=self.alphabet,
                        device=self.device,
                        output_shift=False,
                        delay=self.delay,
                        transform=None)
        return dataset

    def model_setup(self):
        model = None
        print('Creating Model delay=', self.delay, ', layers=', self.model_layers, ', units=', self.rnn_units,
              ', bidi=', self.bidi, ', embedding=', self.embedding_size, 'device=', self.device)
        if self.model_layers > 1:
            # This should be a stacked-LSTM
            model = MultiLayerLSTMNet(self.model_layers,
                                      len(self.alphabet)+1,  # +1 for the mask
                                      self.rnn_units,
                                      output_size=len(self.alphabet),
                                      bidi=self.bidi,
                                      embedding_size=self.embedding_size,
                                      dropout=self.dropout).to(self.device)
        else:
            model = SingleLayerLSTMNet(len(self.alphabet)+1,  # +1 for the mask
                                       self.rnn_units,
                                       output_size=len(self.alphabet),
                                       bidi=self.bidi,
                                       embedding_size=self.embedding_size,
                                       dropout=self.dropout).to(self.device)
        loss_function = nn.NLLLoss().to(self.device)
        if self.weight_decay == 0:
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return model, loss_function, optimizer

    def save_model(self, directory):
        torch.save({
            'model': self.model.state_dict(),
            'delay': self.delay,
            'model_layers': self.model_layers,
            'embedding_size': self.embedding_size,
            'rnn_units': self.rnn_units,
            'bidi': self.bidi,
            'alphabet': self.alphabet,
            'dropout': self.dropout,
        }, directory + '/model_weights.pt')

    def load_model(self, file_name):
        state = torch.load(file_name, map_location=lambda storage, loc: storage)
        self.delay = state['delay'] 
        self.model_layers = state['model_layers'] 
        self.embedding_size = state['embedding_size']
        self.bidi = state['bidi']
        self.alphabet = state['alphabet']
        self.mask_id = len(self.alphabet)
        self.rnn_units = state['rnn_units']
        self.model, loss_function, _ = self.model_setup() 
        self.model.load_state_dict(state['model'])
        return loss_function

    def save_checkpoint(self, epoch, optimizer, loss, results):
        if self.checkpoint_dir is None:
            return
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'delay': self.delay,
            'model_layers': self.model_layers,
            'embedding_size': self.embedding_size,
            'alphabet': self.alphabet,
            'bidi': self.bidi,
            'dropout': self.dropout,
            'seq_length': self.seq_length,
            'results': results,
            'wait': self.early['wait'],
        }, self.checkpoint_dir + '/checkpoint_epoch_' + str(epoch) + '.tar')

    def check_early_stop(self, current_loss):
        # Early stopping disabled?
        if self.patience <= 0:
            return False

        if current_loss - self.early['best_loss'] < - self.early['min_delta']:
            self.early['best_loss'] = current_loss
            self.early['wait'] = 1
        else:
            if self.early['wait'] > self.patience:
                return True
            self.early['wait'] += 1
        return False

    def run(self):
        dataset = self.preload_data("train")
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        if self.alphabet is None:
            self.alphabet = dataset.get_alphabet()
        self.mask_id = len(self.alphabet)
        dataset_val = self.preload_data("valid")
        dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=0)

        model, loss_function, optimizer = self.model_setup()
        self.model = model
        self.total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('#Parameters', self.total_params)

        # Add the learning rate scheduler
        if self.lr_schedule:
           scheduler = self.schedule = ReduceLROnPlateau(optimizer,
                                                         patience=3,
                                                         factor=self.lr_schedule,
                                                         threshold=1e-2,
                                                         threshold_mode='rel',
                                                         eps=1e-5,
                                                         verbose=True)

        results = {'loss': [],
                   'val_loss': [],
                   'bpc_val': [],
                   'bpc_test': 1e10,
                   'val_time': [],
                   'test_time': 0.0,
                   }
        clip = 1.0
        for epoch in range(self.max_epochs):
            # Training
            print('Starting Epoch', epoch)
            train_loss = []
            for i_batch, sample_batched in enumerate(dataloader):
                source_orig, targets = sample_batched['input'], sample_batched['output']
                # Create mask for this batch
                batch_size = source_orig.shape[0]
                seq_length = source_orig.shape[1]
                mask = torch.empty((batch_size, seq_length)).uniform_() <= self.p
                source_orig[mask] = self.mask_id
                targets[1-mask[:, :targets.shape[1]]] = self.mask_id
                # NOTE: We don't need to reset the initial hidden state because the default is to use zero for c0 and h0
                model.zero_grad()
                # We need source_orig to be masked and the masked inputs to be used for the output (loss fun) only
                output, _ = model(source_orig)
                if self.delay == 0:
                    filtered_output = output
                else:
                    filtered_output = output[:, self.delay:, :]
                total_loss = F.nll_loss(F.log_softmax(filtered_output, dim=2).permute([0, 2, 1]), targets, ignore_index=self.mask_id)
                total_loss.backward()
                _ = nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                train_loss.append(total_loss.item())
                print('Batch', i_batch, 'Loss:', total_loss.item(), 'mean loss', sum(train_loss) / (i_batch+1))

            # Validation
            model.eval()
            total_bpc, total_items, total_val_loss, total_times, total_batches = self.eval_model(dataloader_val, loss_function)
            results['bpc_val'].append(total_bpc / total_items)
            results['val_loss'].append(total_val_loss)
            results['val_time'].append(sum(total_times) / float(total_batches))
            results['loss'].append(train_loss)
            print('Validation BPC: ', total_bpc / total_items, 'bits/char (out of total_items:', total_items, ')')
            print('Validation loss: ', total_val_loss)
            mu = sum(total_times) / total_batches
            print('Avg. runtime p/sequence: ', mu, ' +/- ', np.sqrt(np.sum(np.array(total_times) ** 2 - mu ** 2) / (total_batches - 1)))
            print('Max. runtime: ', max(total_times))
            print('Total runtime: ', sum(total_times))
            model.train()

            # Save a checkpoint for reference
            self.save_checkpoint(epoch, optimizer, loss_function, results)

            # Check LR scheduler
            if self.lr_schedule:
                scheduler.step(total_val_loss)

            # Early stopping?
            if self.check_early_stop(total_val_loss):
                print('Early stopping at epoch %d...' % (epoch))
                break

            # check for total convergence
            if total_val_loss < 1e-4:
                print('Automatic stopping due to MSE error is zero')
                break

        # Test the trained model
        model.eval()
        dataset_test = self.preload_data("test")
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=0)
        total_bpc, total_items, _, total_times, total_batches = self.eval_model(dataloader_test, loss_function)
        results['bpc_test'] = total_bpc / total_items
        results['test_time'] = sum(total_times) / float(total_batches)

        print('Test BPC: ', total_bpc / total_items, 'bits/char (out of total_items:', total_items, ')')
        mu = sum(total_times) / total_batches
        print('Avg. runtime p/sequence: ', mu, ' +/- ',
              np.sqrt(np.sum(np.array(total_times) ** 2 - mu ** 2) / (total_batches - 1)))
        print('Max. runtime: ', max(total_times))
        print('Total runtime: ', sum(total_times))
        return results

    def run_time_measurement(self, loss_function, repetitions=5):
        dataset = self.preload_data("train")
        dataloader_test = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        overall_times = np.zeros((repetitions,))
        model_bpc = np.zeros((repetitions,))
        model_loss = np.zeros((repetitions,))
        all_nseq = np.zeros((repetitions,))
        for j in range(repetitions+1):
            # first repetition is not recorded to warm up the device (burn-in time)
            out_d = self.eval_model(dataloader_test, loss_function)
            if j > 0:
                i = j - 1
                model_bpc[i], all_nseq[i], model_loss[i], total_times, n_batches = out_d
                overall_times[i] = np.mean(np.array(total_times))
                if i == 0:
                    all_batch_times = np.zeros((repetitions, n_batches))
                all_batch_times[i, ] = np.array(total_times)
        print('model type:,bilstm{},y{}'.format(int(self.bidi), self.model_layers))
        print('delay and units:,d{},c{}'.format(self.delay, self.rnn_units))
        print('avg runtime per batch:,', np.mean(overall_times), ',', np.std(overall_times))
        output = {'test_bpc': model_bpc,
                  'test_loss': model_loss,
                  'total_time': overall_times,
                  'batch_times': all_batch_times,
                  'n_sequences': all_nseq,
                  'delay': self.delay,
                  'model_layers': self.model_layers,
                  'embedding_size': self.embedding_size,
                  'bidi': self.bidi,
                  'n_units': self.rnn_units,}
        return output 

    def eval_model(self, dataloader, loss_function):
        total_bpc = 0.0
        total_items = 0.0
        total_loss = 0.0
        total_batches = 0
        total_times = []
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                source, targets = sample_batched['input'], sample_batched['output']
                batch_size = source.shape[0]
                seq_length = source.shape[1]
                mask = torch.empty((batch_size, seq_length)).uniform_() <= self.p
                source[mask] = self.mask_id
                targets[1 - mask[:, :targets.shape[1]]] = self.mask_id
                # Predict for this batch
                start_time = time.time()
                char_scores, _ = self.model(source)
                end_time = time.time() - start_time
                # Compute BPC for the batch
                # NOTE: Softmax missing at the output of the network
                if self.delay == 0:
                    bpc, batch_size = self.bits_per_character(F.softmax(char_scores, 2), targets)
                    batch_size = float(torch.sum(mask).item())
                    val_loss = F.nll_loss(F.log_softmax(char_scores, dim=2).permute([0, 2, 1]), targets, ignore_index=self.mask_id)
                else:
                    bpc, batch_size = self.bits_per_character(F.softmax(char_scores[:, self.delay:, :], 2), targets)
                    batch_size = float(torch.sum(mask).item())
                    val_loss = F.nll_loss(F.log_softmax(char_scores, dim=2).permute([0, 2, 1])[:, :, self.delay:], targets, ignore_index=self.mask_id)
                total_loss += val_loss.item()
                total_bpc += bpc
                total_items += batch_size
                total_batches += 1
                total_times.append(end_time)
        return total_bpc, total_items, total_loss / total_batches, total_times, total_batches

    def bits_per_character(self, predictions, targets, divide_result=False):
        """ Compute BPC for a tensor of softmax outputs vs expected target values"""
        elements = predictions.shape[0]
        eps = 1e-8
        log2_scores = torch.log2(predictions + eps)
        # create a mask of 1 and 0s for ground truth.
        batch_size, seq_length, len_alfa = log2_scores.shape
        mask_bpc = torch.zeros((batch_size, seq_length, len_alfa+1), device=log2_scores.device)

        for elem_batch in range(elements):
            mask_bpc[elem_batch, torch.arange(self.seq_length), targets[elem_batch, :]] = -1.0

        # compute bpc for each element in the batch
        bpc = torch.sum(torch.sum(torch.sum(torch.mul(log2_scores, mask_bpc[:,:,:-1]), 2), 1)).item()
        if divide_result:
            return bpc / elements
        else:
            return bpc, elements
