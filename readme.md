# Pixel models

Implementations of autoregressive algorithms from:
* [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)
* [Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)
* [PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture Likelihood and Other Modifications](https://arxiv.org/abs/1701.05517)
* [PixelSNAIL: An Improved Autoregressive Generative Model](https://arxiv.org/abs/1712.09763)


## Usage

The architecture files `pixelcnn.py`, `pixelcnnpp.py`, and `pixelsnail.py` contain model classes, loss function and generation function; `optim.py` implements an exponential moving average wrapper around torch optimizers; `main.py` contains the common logic around training, evaluation, and generation.

To train a model:
```
python main.py --train
               --dataset      # choice from cifar10, mnist, colored-mnist
               --data_path    # path to dataset
               --[add'l options]
               --model        # choice from pixelcnn, pixelcnnpp, pixelsnail;
                              # activates subparser for specific model params
```
Additional options are in the `main.py` parser arguments:
* training options - e.g. number of epochs, learning rate, learning rate decay, polyak averaging, cuda device id, batch_size.
* model specific options - e.g. number of channels, number of residual layers, kernel size, etc.
* dataset options - e.g. number of bits, number of conditional classes, data location, etc.

To evaluate a model or generate from a model:
```
python main.py --generate     # [evaluate]; if evaluate, need to specify dataset and data_path
               --restore_file # path to .pt checkpoint
               --model        # choice from pixelcnn, pixelcnnpp, pixelsnail
```

## Results
Autoregressive models are particularly computationally intensive. I tested the above on a single batch of CIFAR10 and MNIST. I have not tried to replicate the published results since I only needed these as building blocks in other models.

## Datasets
For colored MNIST see Berkeley's [CS294-158](https://sites.google.com/view/berkeley-cs294-158-sp19/home); the dataset can be downloaded [here](https://drive.google.com/open?id=1hm077GxmIBP-foHxiPtTxSNy371yowk2).


## Useful resources
Tensorflow implementations by the authors of PixelCNN++ and PixelSNAIL
* https://github.com/openai/pixel-cnn
* https://github.com/neocxi/pixelsnail-public


## Dependencies
* python 3.7
* pytorch 1.1
* numpy
* tensorboardX
* tqdm
