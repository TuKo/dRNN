from experiments import MLMExperiment
import torch
import argparse
from pathlib import Path
import numpy as np


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def main(delay, batch_size, lr, rnn_units, layers, max_epochs, patience, lr_schedule, weight_decay, wdropout, bidi,
         experiment_name, data_dir, output_dir, model_dir, seq_length, embedding_size,
         checkpoint_path=None,
         ):
    # 1. Check if the directories exist, otherwise create them
    if not data_dir.exists():
        raise ValueError('Data directory does not exist.')
    if checkpoint_path is None:
        output_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)
        exp_number = 0
        rnd_number = np.random.randint(0, 8192)
        experiment_dir = Path(output_dir, experiment_name + '_' + str(exp_number))
        while experiment_dir.exists():
            exp_number += 1
            rnd_number = np.random.randint(0, 8192)
            experiment_dir = Path(output_dir, experiment_name + '_' + str(exp_number))
        experiment_dir = Path(output_dir, experiment_name + '_' + str(exp_number) + '_' + str(rnd_number))
        experiment_dir.mkdir(exist_ok=True)
        model_dir = Path(model_dir, experiment_name + '_' + str(exp_number) + '_' + str(rnd_number))
        model_dir.mkdir(exist_ok=True)

    print(experiment_dir)
    print(model_dir)

    # 2. Setup the experiment
    dataset_setup = {}
    my_experiment = MLMExperiment(dataset_setup,
                                  layers,
                                  delay,
                                  bidi=bidi,
                                  data_dir=str(data_dir),
                                  batch_size=batch_size,
                                  rnn_units=rnn_units,
                                  seq_length=seq_length,
                                  embedding_size=embedding_size,
                                  lr=lr,
                                  lr_schedule=lr_schedule,
                                  weight_decay=weight_decay,
                                  dropout=wdropout,
                                  name=experiment_name,
                                  patience=patience,
                                  checkpoint_dir=str(model_dir),
                                  max_epochs=max_epochs,
                                  device=device)

    # 3. Train the network
    results = my_experiment.run()

    # 4. Save the results
    np.savez(str(experiment_dir) + '/results.npz',
             loss=results['loss'],
             val_loss=results['val_loss'],
             bpc_valid=results['bpc_val'],
             bpc_test=results['bpc_test'],
             delay=delay,
             batch_size=batch_size,
             bidi=bidi,
             layers=layers,
             seq_length=seq_length,
             embedding_size=embedding_size,
             rnn_units=rnn_units,
             max_epochs=max_epochs,
             model_dir=str(model_dir),
             experiment_name=experiment_name)

    # 5. Save the final model
    my_experiment.save_model(str(model_dir))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--experiment_name", type=str, default="text8_mlm", help="Name of experiment")
    parser.add_argument("-c", "--cells", type=int, default=1024, help="Number of cells in the RNN Layer")
    parser.add_argument("-d", "--delay", type=int, default=0, help="Delay parameter")
    parser.add_argument("-y", "--layers", type=int, default=1, help="Layers")
    parser.add_argument("-z", "--bidi", action='store_true', help="Use bidirectional LSTM")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Maximum number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-s", "--lr_schedule", type=float, default=None, help="lr schedule factor")
    parser.add_argument("-w", "--weight_decay", type=float, default=0, help="L2 Weight Decay")
    parser.add_argument("--seq_length", type=int, default=180, help="Sequence length")
    parser.add_argument("--embedding_size", type=int, default=10, help="Dimension for embedding")
    parser.add_argument("--weights_dropout", type=float, default=None, help="Dropout value")
    parser.add_argument("-p", "--patience", type=int, default=0, help="Early Stopping patience")
    parser.add_argument("-o", "--output_dir", type=str, default='./results/mlm/', help="Directory to store results")
    parser.add_argument("-i", "--data_dir", type=str, default='./data/', help="Directory with the datasets")
    parser.add_argument("-m", "--model_dir", type=str, default='./results/mlm/models/', help="Directory to store the Models")
    parser.add_argument("--checkpoint", help="Start from checkpoint.")

    args = parser.parse_args()
    print(args)
    delay = args.delay

    rnn_units = args.cells
    layers = args.layers
    bidi = args.bidi
    embedding_size = args.embedding_size
    seq_length = args.seq_length
    batch_size = args.batch_size
    max_epochs = args.epochs
    lr = args.learning_rate
    lr_schedule = args.lr_schedule
    weight_decay = args.weight_decay
    wdropout = args.weights_dropout
    patience = args.patience
    experiment_name = args.experiment_name
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    print('Torch is using', device)
    main(delay, batch_size, lr, rnn_units, layers, max_epochs, patience, lr_schedule, weight_decay, wdropout, bidi,
         experiment_name, data_dir, output_dir, model_dir, seq_length, embedding_size,
         args.checkpoint)

