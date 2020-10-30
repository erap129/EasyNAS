import inspect
import random
from copy import deepcopy

from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from easynas import constants
import functools
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback


class ValAccCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')


class LitModel(pl.LightningModule):
    def __init__(self, model, loss_function):
        super().__init__()
        self.model = model
        self.loss_function = loss_function

    def forward(self, x):
        return self.model(x)

    def get_loss(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y.long())
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer


def partialclass(cls, *args, **kwds):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)
    return NewCls


def calculate_activation_sizes(layer_collection, input_shape):
    channels = input_shape[0]
    activation_sizes = [input_shape]
    for layer in layer_collection:
        if not inspect.isclass(layer) and type(layer) != nn.MaxPool2d:
            activation_sizes.append(activation_sizes[-1])
            continue
        new_activation = []
        for dimension in range(len(activation_sizes[-1][1:])):
            act_size = activation_sizes[-1][dimension+1]
            if hasattr(layer, "__bases__") and nn.Conv2d in layer.__bases__:
                dummy_layer = layer(channels)
                act_size = int(np.floor(((act_size - (dummy_layer.dilation[dimension] * (dummy_layer.kernel_size[dimension] - 1)) - 1) / dummy_layer.stride[dimension]) + 1))
                channels = dummy_layer.out_channels
                new_activation.append(act_size)
            elif type(layer) == nn.MaxPool2d:
                act_size = int(np.floor((act_size - layer.kernel_size[dimension]) / layer.stride[dimension] + 1))
                new_activation.append(act_size)
            if act_size < 1:
                return -1
        new_activation.insert(0, channels)
        activation_sizes.append(new_activation)
    return activation_sizes


def init_conv_layer(input_shape):
    kernel_limits = [1] * constants.MAX_N_KERNELS
    for sh_idx in range(len(input_shape[1:])):
        kernel_limits[sh_idx] = np.inf
    return partialclass(nn.Conv2d,
        out_channels=(random.randint(1, constants.MAX_CONV_OUT_CHANNELS)),
        kernel_size=(min(random.randint(1, constants.MAX_CONV_KERNEL_SIZE[0]), kernel_limits[0]),
                     min(random.randint(1, constants.MAX_CONV_KERNEL_SIZE[1]), kernel_limits[1])),
        stride=(min(random.randint(1, constants.MAX_CONV_STRIDE[0]), kernel_limits[0]),
                min(random.randint(1, constants.MAX_CONV_STRIDE[1]), kernel_limits[1])),
        dilation=(min(random.randint(1, constants.MAX_CONV_DILATION[0]), kernel_limits[0]),
                min(random.randint(1, constants.MAX_CONV_DILATION[1]), kernel_limits[1]))
    )


def init_maxpool_layer(input_shape):
    kernel_limits = [1] * constants.MAX_N_KERNELS
    for sh_idx in range(len(input_shape[1:])):
        kernel_limits[sh_idx] = np.inf
    return nn.MaxPool2d(
        kernel_size=(min(random.randint(1, constants.MAX_POOLING_KERNEL_SIZE[0]), kernel_limits[0]),
                     min(random.randint(1, constants.MAX_POOLING_KERNEL_SIZE[1]), kernel_limits[1])),
        stride=(min(random.randint(1, constants.MAX_POOLING_STRIDE[0]), kernel_limits[0]),
                min(random.randint(1, constants.MAX_POOLING_STRIDE[1]), kernel_limits[1]))
    )


def random_initialize_layer(layer, input_shape):
    init_functions = {
        nn.Conv2d: init_conv_layer,
        nn.Dropout: lambda _: nn.Dropout(),
        nn.MaxPool2d: init_maxpool_layer,
        nn.Identity: lambda _: nn.Identity(),
        nn.ReLU: lambda _: nn.ReLU()
    }
    return init_functions[layer](input_shape)


def generate_random_model(n_layers, input_shape, available_modules=constants.AVAILABLE_MODULES):
    layer_collection = []
    for i in range(n_layers):
        layer_collection.append(random_initialize_layer(random.choice(available_modules), input_shape))
    if calculate_activation_sizes(layer_collection, input_shape) != -1:
        return layer_collection
    else:
        return generate_random_model(n_layers, input_shape)


def fix_layer_list_by_input_dims(layer_list, input_dims):
    new_layer_list = []
    for layer in layer_list:
        layer = deepcopy(layer)
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
            layer.kernel_size = tuple(list(layer.kernel_size[:input_dims]) + [1] * (len(layer.kernel_size) - input_dims))
            layer.stride = tuple(list(layer.stride[:input_dims]) + [1] * (len(layer.stride) - input_dims))
        if isinstance(layer, nn.Conv2d):
            layer.dilation = tuple(list(layer.dilation[:input_dims]) + [1] * (len(layer.dilation) - input_dims))
        new_layer_list.append(layer)
    return new_layer_list


def generate_sequential(layer_list, input_shape, output_size):
    activation_sizes = calculate_activation_sizes(layer_list, input_shape)
    n_channels = input_shape[0]
    new_layer_list = []
    for layer, act_size in zip(layer_list, activation_sizes):
        if hasattr(layer, "__bases__") and nn.Conv2d in layer.__bases__:
            new_layer_list.append(layer(n_channels))
            n_channels = new_layer_list[-1].out_channels
        else:
            new_layer_list.append(layer)
    # new_layer_list = fix_layer_list_by_input_dims(new_layer_list, len(input_shape) - 1)
    pre_flatten_size = np.prod([x for x in activation_sizes[-1] if x < np.inf])
    new_layer_list.extend([nn.Flatten(), nn.Linear(pre_flatten_size, output_size)])
    return nn.Sequential(*new_layer_list)


def get_model_score(model, X_train, y_train, X_val, y_val):
    litmodel = LitModel(model, F.cross_entropy)
    trainer = pl.Trainer(callbacks=[ValAccCallback()], max_epochs=5)
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
    trainer.fit(litmodel, DataLoader(train_dataset), DataLoader(val_dataset))
    y_pred = torch.max(litmodel(torch.Tensor(X_val)), 1)[1]
    return accuracy_score(y_val, y_pred)

