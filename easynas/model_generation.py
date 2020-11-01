import inspect
import random
from copy import deepcopy
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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


def init_conv_layer(input_shape, max_n_kernels, max_conv_kernel_size, max_conv_stride, max_conv_dilation,
                    max_conv_out_channels):
    kernel_limits = [1] * max_n_kernels
    for sh_idx in range(len(input_shape[1:])):
        kernel_limits[sh_idx] = np.inf
    return partialclass(nn.Conv2d,
        out_channels=(random.randint(1, max_conv_out_channels)),
        kernel_size=(min(random.randint(1, max_conv_kernel_size[0]), kernel_limits[0]),
                     min(random.randint(1, max_conv_kernel_size[1]), kernel_limits[1])),
        stride=(min(random.randint(1, max_conv_stride[0]), kernel_limits[0]),
                min(random.randint(1, max_conv_stride[1]), kernel_limits[1])),
        dilation=(min(random.randint(1, max_conv_dilation[0]), kernel_limits[0]),
                min(random.randint(1, max_conv_dilation[1]), kernel_limits[1]))
    )


def init_maxpool_layer(input_shape, max_n_kernels, max_pooling_kernel_size, max_pooling_stride):
    kernel_limits = [1] * max_n_kernels
    for sh_idx in range(len(input_shape[1:])):
        kernel_limits[sh_idx] = np.inf
    return nn.MaxPool2d(
        kernel_size=(min(random.randint(1, max_pooling_kernel_size[0]), kernel_limits[0]),
                     min(random.randint(1, max_pooling_kernel_size[1]), kernel_limits[1])),
        stride=(min(random.randint(1, max_pooling_stride[0]), kernel_limits[0]),
                min(random.randint(1, max_pooling_stride[1]), kernel_limits[1]))
    )


def random_initialize_layer(layer, input_shape, max_n_kernels, max_conv_kernel_size, max_conv_stride, max_conv_dilation,
                                                        max_conv_out_channels, max_pooling_kernel_size,
                                                        max_pooling_stride):
    init_functions = {
        nn.Conv2d: functools.partial(init_conv_layer, max_n_kernels=max_n_kernels,
                                     max_conv_kernel_size=max_conv_kernel_size, max_conv_stride=max_conv_stride,
                                     max_conv_dilation=max_conv_dilation, max_conv_out_channels=max_conv_out_channels),
        nn.Dropout: lambda _: nn.Dropout(),
        nn.MaxPool2d: functools.partial(init_maxpool_layer, max_n_kernels=max_n_kernels,
                                        max_pooling_kernel_size=max_pooling_kernel_size,
                                        max_pooling_stride=max_pooling_stride),
        nn.Identity: lambda _: nn.Identity(),
        nn.ReLU: lambda _: nn.ReLU()
    }
    return init_functions[layer](input_shape)


def generate_random_model(n_layers, input_shape, available_modules, max_n_kernels, max_conv_kernel_size, max_conv_stride,
                          max_conv_dilation, max_conv_out_channels, max_pooling_kernel_size, max_pooling_stride):
    layer_collection = []
    for i in range(n_layers):
        layer_collection.append(random_initialize_layer(random.choice(available_modules), input_shape, max_n_kernels,
                                                        max_conv_kernel_size, max_conv_stride, max_conv_dilation,
                                                        max_conv_out_channels, max_pooling_kernel_size,
                                                        max_pooling_stride))
    if calculate_activation_sizes(layer_collection, input_shape) != -1:
        return layer_collection
    else:
        return generate_random_model(n_layers, input_shape, available_modules, max_n_kernels, max_conv_kernel_size,
                                     max_conv_stride, max_conv_dilation, max_conv_out_channels, max_pooling_kernel_size,
                                     max_pooling_stride)


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
    pre_flatten_size = np.prod([x for x in activation_sizes[-1] if x < np.inf])
    new_layer_list.extend([nn.Flatten(), nn.Linear(pre_flatten_size, output_size)])
    return nn.Sequential(*new_layer_list)


def get_model_score(model, X_train, y_train, X_val, y_val, batch_size, max_epochs, progress_bar=False):
    litmodel = LitModel(model, F.cross_entropy)
    trainer = pl.Trainer(callbacks=[ValAccCallback()], max_epochs=max_epochs, gpus=1,
                         progress_bar_refresh_rate=int(progress_bar), weights_summary=None)
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    trainer.fit(litmodel, DataLoader(train_dataset, batch_size=batch_size), valloader)
    all_preds = []
    for x, _ in valloader:
        y_pred = litmodel(x)
        all_preds.append(y_pred.detach().numpy())
    predictions = np.argmax(np.vstack(all_preds), axis=1)
    return accuracy_score(y_val, predictions)


