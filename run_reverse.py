from experiments import ReverseExperiment
import torch
import argparse
from pathlib import Path
import numpy as np


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def main(delay, model_layers, batch_size, lr, rnn_units, max_epochs, patience, lr_schedule, weight_decay, wdropout,
         bidi, seq_length, train_size, input_classes, experiment_name, data_dir, output_dir, model_dir):
    # 1. Check if the directories exist, otherwise create them
    if not data_dir.exists():
        raise ValueError('Data directory does not exist.')
    output_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    exp_number = 0
    experiment_dir = Path(output_dir, experiment_name + '_' + str(exp_number))
    while experiment_dir.exists():
        exp_number += 1
        experiment_dir = Path(output_dir, experiment_name + '_' + str(exp_number))
    experiment_dir.mkdir(exist_ok=True)
    model_dir = Path(model_dir, experiment_name + '_' + str(exp_number))
    model_dir.mkdir(exist_ok=True)

    print(experiment_dir)
    print(model_dir)

    # 2. Setup the experiment
    dataset_setup = {'input_classes': input_classes,
                     'sequence_length': seq_length,
                     'train_size': train_size,
                     }

    my_experiment = ReverseExperiment(dataset_setup,
                                      model_layers,
                                      delay,
                                      bidi=bidi,
                                      batch_size=batch_size,
                                      rnn_units=rnn_units,
                                      lr=lr,
                                      lr_schedule=lr_schedule,
                                      weight_decay=weight_decay,
                                      dropout=wdropout,
                                      name=experiment_name,
                                      patience=patience,
                                      data_dir=str(data_dir),
                                      checkpoint_dir=str(model_dir),
                                      max_epochs=max_epochs,
                                      device=device)

    # 3. Train the network
    results = my_experiment.run()

    # 4. Save the results
    np.savez(str(experiment_dir) + '/results.npz',
             loss=results['loss'],
             val_loss=results['val_loss'],
             bpc_valid=results['acc_val'],
             bpc_test=results['acc_test'],
             delay=delay,
             model_layers=model_layers,
             dataset_setup=dataset_setup,
             batch_size=batch_size,
             rnn_units=rnn_units,
             max_epochs=max_epochs,
             dropout=wdropout,
             bidi=bidi,
             model_dir=str(model_dir),
             experiment_name=experiment_name)

    # 5. Save the final model
    my_experiment.save_model(str(model_dir))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--experiment_name", type=str, default="reverse", help="Name of experiment")
    parser.add_argument("-c", "--cells", type=int, default=100, help="Number of cells in the RNN Layer")
    parser.add_argument("-q", "--model_layers", type=int, default=1, help="Number of layers in the model to train")
    parser.add_argument("-d", "--delay", type=int, default=0, help="Delay parameter")
    parser.add_argument("-z", "--bidi", action='store_true', help="Use bidirectional LSTM")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="Maximum number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-s", "--lr_schedule", type=float, default=None, help="lr schedule factor")
    parser.add_argument("-w", "--weight_decay", type=float, default=0, help="L2 Weight Decay")
    parser.add_argument("--seq_length", type=int, default=20, help="Sequence length of reverse dataset")
    parser.add_argument("--input_classes", type=int, default=4, help="Input values of the data in reverse dataset")
    parser.add_argument("--train_size", type=int, default=10000, help="Size of train set of reverse dataset")
    parser.add_argument("--weights_dropout", type=float, default=None, help="Dropout value")
    parser.add_argument("-p", "--patience", type=int, default=10, help="Early Stopping patience")
    parser.add_argument("-o", "--output_dir", type=str, default='./results/', help="Directory to store results")
    parser.add_argument("-i", "--data_dir", type=str, default='./data/reverse/', help="Directory with the datasets")
    parser.add_argument("-m", "--model_dir", type=str, default='./results/models/', help="Directory to store the Models")

    args = parser.parse_args()
    print(args)
    delay = args.delay
    rnn_units = args.cells
    model_layers = args.model_layers
    bidi = args.bidi
    batch_size = args.batch_size
    max_epochs = args.epochs
    lr = args.learning_rate
    lr_schedule = args.lr_schedule
    weight_decay = args.weight_decay
    wdropout = args.weights_dropout
    seq_length = args.seq_length
    input_classes = args.input_classes
    train_size = args.train_size
    patience = args.patience
    experiment_name = args.experiment_name
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    print('Torch is using', device)
    main(delay, model_layers, batch_size, lr, rnn_units, max_epochs, patience, lr_schedule, weight_decay, wdropout,
         bidi, seq_length, train_size, input_classes,
         experiment_name, data_dir, output_dir, model_dir)


