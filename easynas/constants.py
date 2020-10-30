from torch import nn

## general parameters
MAX_N_KERNELS = 2

## convolution layers
MAX_CONV_KERNEL_SIZE = (20, 20)
MAX_CONV_STRIDE = (3, 3)
MAX_CONV_DILATION = (1, 1)
MAX_CONV_OUT_CHANNELS = 50

## max-pooling layers
MAX_POOLING_KERNEL_SIZE = (3, 3)
MAX_POOLING_STRIDE = (3, 3)

## genetic algorithm parameters
MAX_CROSSOVER_ATTEMPTS = 50
AVAILABLE_MODULES = [nn.Conv2d, nn.Dropout, nn.MaxPool2d, nn.Identity, nn.ReLU]