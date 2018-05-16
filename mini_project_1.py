from torch import nn
from torch.nn import ReLU

class Net(nn.Module):
    def __init__(self, conv_layer, linear_layer, act_func=ReLU(),
                 with_batchnorm_conv=False, with_dropout_conv=False,
                 with_batchnorm_lin=False, with_dropout_lin=False):
        """

        :param conv_layer:
        :param linear_layer:
        :param act_func:
        :param with_batchnorm_conv:
        :param with_dropout_conv:
        :param with_batchnorm_lin:
        :param with_dropout_lin:
        """

        super().__init__()

        modules_conv = []

        nb_in_channels = 28
        nb_measurements = 50
        linear_in_size = nb_measurements

        for conv in conv_layer:
            kernel_size = conv[0]
            nb_out_channels = conv[1]
            modules_conv.append(nn.Conv1d(nb_in_channels, nb_out_channels, kernel_size=kernel_size))
            modules_conv.append(act_func)

            if with_batchnorm_conv:
                modules_conv.append(nn.BatchNorm1d()) #TODO

            if with_dropout_conv:
                modules_conv.append(nn.Dropout2d()) #TODO

            nb_in_channels = nb_out_channels
            linear_in_size -= kernel_size - 1

        linear_in_size *= nb_out_channels

        modules_lin = []

        for linear in linear_layer:
            modules_lin.append(nn.Linear(linear_in_size, linear))
            modules_lin.append(act_func)
            linear_in_size = linear

            if with_batchnorm_lin:
                modules_lin.append(nn.BatchNorm1d()) #TODO

            if with_dropout_lin:
                modules_lin.append(nn.Dropout()) #TODO

        modules_lin.append(nn.Linear(linear_in_size, 2))

        self.seq_conv = nn.Sequential(*modules_conv)
        self.seq_lin  = nn.Sequential(*modules_lin)

    #Defining the different operations on the data in the right order
    def forward(self, x):
        return self.seq_lin(self.seq_conv(x).view(-1,self.seq_lin[0].in_features))
