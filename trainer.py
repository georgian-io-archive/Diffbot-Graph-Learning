import math

import dgl
import torch
from model_han import HAN
from model_hrgcn import HeteroRGCN
from model_mlp import MLP, calc_mlp_dims


class Trainer:
    def __init__(self, dataset, model_name, training_args, device):
        self.dataset = dataset
        self.labels = torch.LongTensor(dataset.labels).to(device)

        self.features_dict = dataset.graph.ndata['feats']
        features_list = [None for _ in range(len(self.features_dict))]
        for node_type, features in self.features_dict.items():
            feats = features.type(torch.float).to(device)
            features_list[dataset.ntype_to_ntype_id_map[node_type]] = feats
            self.features_dict[node_type] = feats
        self.model_name = model_name
        if self.model_name == 'han':
            self.in_dims = [features.shape[1] for features in features_list]
            num_heads = [training_args.num_heads for _ in range(training_args.num_layers)]
            self.graph = dgl.heterograph(dataset.relations_dict).to(device)
            features_list = [features.float() for features in features_list]
            node_ind = -1
            for i, node_metapaths in enumerate(dataset.metapaths):
                if node_metapaths[0][0][0] == dataset.node_type_w_labels:
                    node_ind = i
                    break
            assert node_ind != -1, f"{dataset.node_type_w_labels} should be a source and tail node type in metapaths {dataset.metapaths}"
            metapaths = dataset.metapaths[node_ind]
            self.node_type_ind_w_labels = self.dataset.metapath_node_inds[node_ind]
            in_dim = self.in_dims[self.node_type_ind_w_labels]

            self.model = HAN(
                metapaths,
                in_dim,
                training_args.hidden_dim,
                out_size=dataset.num_classes,
                num_heads=num_heads,
                dropout=training_args.dropout_rate
            )
        elif model_name == 'hrgcn':
            self.graph = dgl.heterograph(dataset.relations_dict).to(device)
            self.model = HeteroRGCN(self.graph, self.features_dict, training_args.hidden_dim,
                                    dataset.num_classes, training_args.num_layers)
        elif model_name == 'mlp':
            in_dim = self.features_dict[self.dataset.node_type_w_labels].shape[1]
            dims = calc_mlp_dims(in_dim, division=int(math.sqrt(int((in_dim-1) / (dataset.num_classes)))), output_dim=dataset.num_classes,)
            self.model = MLP(in_dim, dataset.num_classes, hidden_channels=dims,
                             return_layer_outs=True, num_hidden_lyr=len(dims))
        else:
            raise ValueError(f'model {self.model_name} is not supported')
        self.features_list = features_list
        self.model.to(device)

    def forward(self):
        if self.model_name == 'han':
            features = self.features_list[self.node_type_ind_w_labels]
            logits, embeddings = self.model(self.graph,
                                            features)
        elif self.model_name == 'hrgcn':
            logits, embeddings = self.model(self.graph,
                                            self.dataset.node_type_w_labels)
        elif self.model_name == 'mlp':
            features = self.features_dict[self.dataset.node_type_w_labels]
            logits, outputs = self.model(features)
            embeddings = outputs[-2]
        else:
            raise ValueError(f'model {self.model_name} is not supported')

        return logits, embeddings

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()