import torch
class hmcn(torch.nn.Module):
    def __init__(self, config):
        super(hmcn, self).__init__()
        hidden_dimension = 1024
        self.hierarchical_class = [24, 200]
        self.global2local = [0, 1024, 1024]
        self.hierarchical_depth = [0, 4096, 4096]
        self.local_layers = torch.nn.ModuleList()
        self.global_layers = torch.nn.ModuleList()
        for i in range(1, len(self.hierarchical_depth)):
            self.global_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dimension + self.hierarchical_depth[i-1], self.hierarchical_depth[i]),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(self.hierarchical_depth[i]),
                    torch.nn.Dropout(p=config.modal_dropout)
                ))
            self.local_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.hierarchical_depth[i], self.global2local[i]),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(self.global2local[i]),
                    torch.nn.Linear(self.global2local[i], self.hierarchical_class[i-1])
                    ))

        self.global_layers.apply(self._init_weight)
        self.local_layers.apply(self._init_weight)
        self.linear = torch.nn.Linear(self.hierarchical_depth[-1], 200)
        self.linear.apply(self._init_weight)
        self.dropout = torch.nn.Dropout(p=config.modal_dropout)
    def _init_weight(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.1) 
    def forward(self, batch):
        local_layer_outputs = []
        batch_size = batch.shape[1]
        global_layer_activation = batch
        for i, (local_layer, global_layer) in enumerate(zip(self.local_layers, self.global_layers)):
            local_layer_activation = global_layer(global_layer_activation)
            local_layer_outputs.append(local_layer(local_layer_activation))
            if i < len(self.global_layers)-1:
                global_layer_activation = torch.cat((local_layer_activation, batch), 1)
            else:
                global_layer_activation = local_layer_activation

        global_layer_output = self.linear(global_layer_activation)
        # local_layer_output = torch.cat(local_layer_outputs, 1)
        return global_layer_output, local_layer_outputs, 0.5 * global_layer_output + 0.5 * local_layer_outputs[-1]   

