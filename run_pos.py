from experiments import POSExperiment
import torch
import argparse
from pathlib import Path
import numpy as np


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def main(char_delay, char_units, char_embeddings, word_delay, word_units, word_embeddings, word_embeddings_file,
         bidi_char, bidi_words,
         batch_size, lr, max_epochs, lr_schedule,
         language, experiment_name, data_dir, output_dir, model_dir,
         wdropout=None, patience=0):

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
    my_experiment = POSExperiment(language,
                                  char_delay,
                                  char_units,
                                  char_embeddings,
                                  word_delay,
                                  word_units,
                                  word_embeddings,
                                  pretrained_word_embeddings=word_embeddings_file,
                                  bidi_char=bidi_char,
                                  bidi_sentence=bidi_words,
                                  batch_size=batch_size,
                                  lr=lr,
                                  lr_schedule=lr_schedule,
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
             acc_valid=results['acc_val'],
             acc_test=results['acc_test'],
             bidi_char=bidi_char,
             bidi_sentence=bidi_words,
             language=language,
             char_delay=char_delay,
             char_units=char_units,
             char_embeddings=char_embeddings,
             word_delay=word_delay,
             word_units=word_units,
             word_embeddings=word_embeddings,
             word_embeddings_file=word_embeddings_file,
             batch_size=batch_size,
             max_epochs=max_epochs,
             model_dir=str(model_dir),
             experiment_name=experiment_name)

    # 5. Save the final model
    my_experiment.save_model(str(model_dir))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--experiment_name", type=str, default="pos", help="Name of experiment")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Maximum number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-s", "--lr_schedule", type=float, default=None, help="lr schedule factor")

    parser.add_argument("-c", "--char_units", type=int, default=200, help="Number of cells in the Characters Layer")
    parser.add_argument("-d", "--char_delay", type=int, default=0, help="Delay parameter Characters Layer")
    parser.add_argument("--char_embeddings", type=int, default=100, help="Dimensionality of Character embeddings")
    parser.add_argument("-w", "--word_units", type=int, default=300, help="Number of cells in the Words Layer")
    parser.add_argument("-q", "--word_delay", type=int, default=0, help="Delay parameter Words Layer")
    parser.add_argument("--word_embeddings", type=int, default=64, help="Dimensionality of Character embeddings")
    parser.add_argument("--word_embeddings_file", action='store_true', help="Pre-trained word embeddings")
    parser.add_argument("-z", "--bidi_char", action='store_true', help="Use bidirectional LSTM for characters")
    parser.add_argument("-y", "--bidi_words", action='store_true', help="Use bidirectional LSTM for word-level")

    parser.add_argument("--language", type=str, default="en", help="Language")

    parser.add_argument("--output_dir", type=str, default='./results/', help="Directory to store results")
    parser.add_argument("--data_dir", type=str, default='./data/', help="Directory with the datasets")
    parser.add_argument("--model_dir", type=str, default='./results/models/', help="Directory to store the Models")

    args = parser.parse_args()
    print(args)
    char_delay = args.char_delay
    char_units = args.char_units
    char_embeddings = args.char_embeddings
    word_delay = args.word_delay
    word_units = args.word_units
    word_embeddings = args.word_embeddings
    word_embeddings_file = args.word_embeddings_file

    bidi_char = args.bidi_char
    bidi_words = args.bidi_words

    batch_size = args.batch_size
    max_epochs = args.epochs
    lr = args.learning_rate
    lr_schedule = args.lr_schedule

    language = args.language
    experiment_name = args.experiment_name
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    print('Torch is using', device)
    main(char_delay, char_units, char_embeddings, word_delay, word_units, word_embeddings, word_embeddings_file,
         bidi_char, bidi_words,
         batch_size, lr, max_epochs, lr_schedule,
         language, experiment_name, data_dir, output_dir, model_dir)


