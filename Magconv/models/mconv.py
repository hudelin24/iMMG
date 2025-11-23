import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_registry import MODEL_REGISTRY

class MagConv1D(nn.Module):
    def __init__(self, input_channels, layer_configs, l_seq=960, dropout_rate=0.1):
        """
        Args:
            input_channels:         3
            layer_configs:          [{"out_channels": , "kernel_size": }]
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.pool_flags = []
        in_channels = input_channels
        for config in layer_configs:
            out_channels = config["out_channels"]
            kernel_size = config["kernel_size"]
            do_pool = config["do_pool"]

            conv_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            bn_layer = nn.BatchNorm1d(out_channels)
            dropout_layer = nn.Dropout(dropout_rate)

            self.layers.append(nn.Sequential(conv_layer, bn_layer, nn.ReLU(), dropout_layer))
            in_channels = out_channels  
            self.pool_flags.append(do_pool)
        self.gap = nn.AdaptiveAvgPool1d(1)        
        #self.fc1 = nn.Linear(in_channels * l_seq // (2 ** len(layer_configs)), 32)
        self.fc1 = nn.Linear(out_channels, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        for layer, do_pool in zip(self.layers, self.pool_flags):
            if do_pool:
                x = F.max_pool1d(layer(x), kernel_size=2)
            else:
                x = layer(x)
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        #x = self.fc1(x)
        return x




@MODEL_REGISTRY.register()
class mcnn_base(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = MagConv1D(input_channels=cfg.MODEL.INPUT_CHANS, layer_configs=cfg.MODEL.LAYER_CONF, 
                                l_seq=cfg.MODEL.SEQ_LEN, dropout_rate=cfg.MODEL.DROP_RATE)

    def forward(self, x):
        x = self.model(x)
        return x


