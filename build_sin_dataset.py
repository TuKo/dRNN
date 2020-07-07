import numpy as np
from pathlib import Path
import argparse


def main(sizes, scale, causality, acausality, seq_length, data_dir='./data/sin/'):
    datasets = ['train', 'valid', 'test']
    assert len(sizes) == 3, 'Three dataset sizes are required [#train, #validation, #test]'

    # Create directory if it doesn't exists
    d = Path(data_dir)
    d.mkdir(exist_ok=True)
    subfilename = 'sin_%d_s%.2f_%d_%d_s%d' % (sizes[0], scale, causality, acausality, seq_length)
    print(subfilename)
    if seq_length <= causality + acausality:
        raise ValueError('Cannot create dataset. Sequence Length should be > filter size (acausality + causality).')

    # Draw the random filter
    h = np.random.uniform(0.0, 1.0, size=(causality+acausality,))
    print("filter", h, np.mean(h))

    # For each dataset do:
    for i, dataset in enumerate(datasets):
        print(dataset)
        data = np.random.uniform(0.0, 1.0, size=(sizes[i], seq_length))
        output = np.zeros_like(data, dtype=np.float)
        for j in range(sizes[i]):
            # convolve input with filter
            for k in range(seq_length):
                left = min(causality, k+1)
                right = min(acausality, seq_length-1-k)
                output[j, k] = np.sum(data[j, k+1-left:k+1+right] * h[causality-left:causality+right])
        targets = np.sin(scale * output)

        # Save the datasets to file
        np.savez(data_dir + '/' + subfilename + '_' + dataset + '.npz',
                 data=data,
                 targets=targets,
                 causality=causality,
                 acausality=acausality,
                 seq_length=seq_length,
                 filter=h,
                 scale=scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--acausality", type=int, default=5,
                        help="Number of acausal (future) timesteps for filter.")
    parser.add_argument('-c', "--causality", type=int, default=10,
                        help="Number of causal timesteps for filter.")
    parser.add_argument('-s', "--scale", type=float, default=0.1,
                        help="Number of causal timesteps for filter.")
    parser.add_argument('-l', "--seq_length", type=int, default=30, help="Length of sequences in dataset")
    parser.add_argument('-1', "--train_samples", type=int, default=10000, help="# samples on train dataset")
    parser.add_argument('-2', "--val_samples", type=int, default=2000, help="# samples on validation dataset")
    parser.add_argument('-3', "--test_samples", type=int, default=2000, help="# samples on test dataset")
    parser.add_argument("-o", "--output_dir", type=str, default='./data/sin/', help="Output directory")
    args = parser.parse_args()
    print(args)

    acausality = args.acausality
    causality = args.causality
    scale = args.scale
    seq_length = args.seq_length
    output_dir = args.output_dir
    sizes = [args.train_samples, args.val_samples, args.test_samples]
    main(sizes, scale, causality, acausality, seq_length, data_dir=output_dir)
