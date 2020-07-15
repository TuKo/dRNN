# Approximating Stacked and Bidirectional Recurrent Architectures with the Delayed Recurrent Neural Network (ICML 2020)
by Javier S. Turek (Intel Labs), Shailee Jain (UT Austin), Vy Vo (Intel Labs), Mihai Capota (Intel Labs), Ted Willke (Intel Labs), Alex Huth (UT Austin)

[Paper](https://proceedings.icml.cc/static/paper_files/icml/2020/5744-Paper.pdf)

## About
Recent work has shown that topological enhancements to recurrent neural networks (RNNs) can increase their expressiveness and representational capacity. Two popular enhancements are stacked RNNs, which increases the capacity for learning non-linear functions, and bidirectional processing, which exploits acausal information in a sequence. In this work, we explore the delayed-RNN, which is a single-layer RNN that has a delay between the input and output. We prove that a weight-constrained version of the delayed-RNN is equivalent to a stacked-RNN. We also show that the delay gives rise to partial acausality, much like bidirectional networks. Synthetic experiments confirm that the delayed-RNN can mimic bidirectional networks, solving some acausal tasks similarly, and outperforming them in others. Moreover, we show similar performance to bidirectional networks in a real-world natural language processing task. These results suggest that delayed-RNNs can approximate topologies including stacked RNNs, bidirectional RNNs, and stacked bidirectional RNNs -- but with equivalent or faster runtimes for the delayed-RNNs.

## Requirements
* Pytorch 1.1 and respective libraries
* Numpy
* Python 3.6

## Creating the synthetic datasets
The synthetic datasets used in experiments 4.1 and 4.2 in the paper can be generated with `build_reverse_dataset.py` and `build_sin_dataset.py`, respectively.
Each run of these scripts generates a dataset for a specific set of parameters.
Check the help with option `-h` for further details.

## Running the experiments
Each experiment is executed with one of the following scripts. Use `-h` to determine the command line options to run the experiment as described in the paper.
* `run_reverse.py` -- Experiment 4.1: Sequence reversal
* `run_sine.py` -- Experiment 4.2: Filtering with sine function 
* `run_mlm.py` -- Experiment 4.3a: Masked Language Model
* `run_time_mlm.py` -- Experiment 4.3b: Runtime 
* `run_pos.py` -- Experiment 4.4: Parts-of-Speech tagging


## Citing the paper

```
@incollection{icml2020_5744,
 author = {Turek, Javier and Jain, Shailee and Vo, Vy and Capot\u{a}, Mihai and Huth, Alexander and Willke, Theodore},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {9961--9971},
 title = {Approximating Stacked and Bidirectional Recurrent Architectures with the Delayed Recurrent Neural Network},
 year = {2020}
}
```
