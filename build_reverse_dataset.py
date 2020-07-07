import numpy as np
from pathlib import Path
import argparse


def main(sizes, input_classes, seq_length=50, data_dir='./data/reverse/'):
    datasets = ['train', 'valid', 'test']
    assert len(sizes) == 3, 'Three dataset sizes are required [#train, #validation, #test]'
    # Create directory if it doesn't exists
    d = Path(data_dir)
    d.mkdir(exist_ok=True)
    subfilename = 'reverse_%d_i%d_s%d' % (sizes[0], input_classes, seq_length)
    print(subfilename)
    if seq_length < 30 or input_classes < 12:
        total_elements = input_classes ** seq_length
        if total_elements < np.sum(sizes):
            raise ValueError('Cannot create dataset. There are not enough samples.')

    # For each dataset do:
    for i, dataset in enumerate(datasets):
        print(dataset)
        input_data = np.random.randint(0, input_classes, size=(sizes[i], seq_length))
        data = np.zeros((sizes[i], seq_length, input_classes))
        print(input_data[0, :])
        for j in range(sizes[i]):
            data[j * np.ones(seq_length, dtype=np.long), np.arange(seq_length), input_data[j, :]] = 1.0
        targets = np.flip(input_data, 1)
        # Save the datasets to file
        np.savez(data_dir + '/' + subfilename + '_' + dataset +'.npz',
                 data=data,
                 targets=targets,
                 input_words=input_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input_classes", type=int, default=4, help="Input classes to simulate in the data (=V)")
    parser.add_argument('-s', "--sequence_length", type=int, default=20, help="Length of sequences in dataset")
    parser.add_argument('-1', "--train_samples", type=int, default=10000, help="# samples on train dataset")
    parser.add_argument('-2', "--val_samples", type=int, default=2000, help="# samples on validation dataset")
    parser.add_argument('-3', "--test_samples", type=int, default=2000, help="# samples on test dataset")
    parser.add_argument("-o","--output_dir", type=str, default='./data/reverse/', help="Output directory")
    args = parser.parse_args()
    print(args)

    sizes = [args.train_samples, args.val_samples, args.test_samples]
    input_classes = args.input_classes
    seq_length = args.sequence_length
    output_dir = args.output_dir
    main(sizes, input_classes, seq_length=seq_length, data_dir=output_dir)
