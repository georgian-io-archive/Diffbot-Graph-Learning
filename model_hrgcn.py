import torch.nn.functional as F
import torch.nn as nn
import dgl.function as fn


class HeteroRGCNLayer(nn.Module):
    def __init__(self, etypes_dim_map, out_size):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                '_'.join(name): nn.Linear(in_dim, out_size) for name, in_dim in etypes_dim_map.items()
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            edge_key = '_'.join([srctype, etype, dsttype])
            Wh = self.weight[edge_key](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % edge_key] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[(srctype, etype, dsttype)] = (fn.copy_u('Wh_%s' % edge_key, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, G, feats_dict, hidden_size, out_size, num_layers):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        # embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
        #               for ntype in G.ntypes}
        # for key, embed in embed_dict.items():
        #     nn.init.xavier_uniform_(embed)
        # self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.feats = feats_dict
        etype_in_dim = {(src_type, etype, dst_type): feats_dict[src_type].shape[1]
                        for src_type, etype, dst_type in G.canonical_etypes}
        etype_hidden_in_dim = {c_etype: hidden_size
                               for c_etype in G.canonical_etypes}

        self.rgcn_layers = nn.ModuleList()
        self.rgcn_layers.append(HeteroRGCNLayer(etype_in_dim, hidden_size))
        for i in range(num_layers-1):
            self.rgcn_layers.append(HeteroRGCNLayer(etype_hidden_in_dim, hidden_size))
        # self.rgcn_layers.append(HeteroRGCNLayer(etype_hidden_in_dim, out_size))
        self.predict = nn.Linear(hidden_size, out_size)

    def forward(self, G, target_node_type):
        for layer in self.rgcn_layers[:-1]:
            in_feat_dict = layer(G, self.feats)
            in_feat_dict = {k: F.leaky_relu(h) for k, h in in_feat_dict.items()}
        in_feat_dict = self.rgcn_layers[-1](G, in_feat_dict)
        out = self.predict(in_feat_dict[target_node_type])
        # get paper logits
        return out, in_feat_dict[target_node_type]