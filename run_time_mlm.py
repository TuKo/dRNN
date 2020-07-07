from experiments import MLMExperiment
import torch
import argparse
from pathlib import Path
import numpy as np
import os


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def main(delay, batch_size, rnn_units, layers, bidi,
         experiment_name, data_dir, output_dir, model_dir, seq_length, embedding_size, model_dirnames):

    dataset_setup = {}
    my_experiment = MLMExperiment(dataset_setup,
                                  layers,
                                  delay,
                                  data_dir=str(data_dir),
                                  bidi=bidi,
                                  batch_size=batch_size,
                                  rnn_units=rnn_units,
                                  seq_length=seq_length,
                                  embedding_size=embedding_size,
                                  name=experiment_name,
                                  checkpoint_dir=str(model_dir),
                                  device=device)
    # Loop over relevant models
    with open(model_dirnames, 'r') as f:
        exp_folders = f.readlines()
    err_list = []
    for fi, fdir in enumerate(exp_folders):
        model_dir = str(output_dir) + '/models/' + str(fdir).rstrip()
        # Load the saved model
        # contains: model, delay, model_layers, embedding_size, bidi, alphabet, rnn_units
        print('Loading model ', model_dir)
        try:
            loss_fcn = my_experiment.load_model(model_dir + '/model_weights.pt')
        except:
            print('Error with loading. Skipping...')
            err_list.append(str(fdir))
            continue
        time_data = my_experiment.run_time_measurement(loss_fcn, repetitions=5)
        save_file = model_dir + '/run_times.npz'
        np.savez(save_file, time_data=time_data)
    print(err_list) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--experiment_name", type=str, default="text8_mlm", help="Name of experiment")
    parser.add_argument("-c", "--cells", type=int, default=10, help="Number of cells in the RNN Layer")
    parser.add_argument("-d", "--delay", type=int, default=1, help="Delay parameter")
    parser.add_argument("-y", "--layers", type=int, default=1, help="Layers")
    parser.add_argument("-z", "--bidi", action='store_true', help="Use bidirectional LSTM")
    parser.add_argument("--embedding_size", type=int, default=10, help="Dimension for embedding")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=180, help="Sequence length")
    parser.add_argument("-o", "--output_dir", type=str, default='./results/mlm/', help="Directory to store results")
    parser.add_argument("-i", "--data_dir", type=str, default='./data/', help="Directory with the datasets")
    parser.add_argument("-m", "--model_dir", type=str, default='./results/mlm/models/', help="Directory to store the Models")
    parser.add_argument("--model_names", type=str, default='./results/mlm/models.txt', help="Text file with list of model names")

    args = parser.parse_args()
    print(args)

    delay = args.delay
    rnn_units = args.cells
    layers = args.layers
    bidi = args.bidi
    embedding_size = args.embedding_size

    seq_length = args.seq_length
    batch_size = args.batch_size
    experiment_name = args.experiment_name
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    model_names = args.model_names

    print('Torch is using', device)
    main(delay, batch_size, rnn_units, layers, bidi,
         experiment_name, data_dir, output_dir, model_dir, seq_length, embedding_size, model_names)
