from abc import ABC
from collections import defaultdict
import json
from os.path import join, exists
from statistics import median

import dgl
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, save_info, load_graphs, load_info
import networkx as nx
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch

import warnings
warnings.simplefilter("once", category=dgl.base.DGLWarning)

from data_preprocess.diffbot_constants import ORG_ENTITY_TYPES, LOCATION_ENTITY_TYPES
from util import get_data_path, create_dir_if_not_exists


def combine_diffbot_categories_labels_w_graph(nx_graph, diffbot_cat):
    labels_dict = {}
    for diffbot_uri, node_data in nx_graph.nodes(data=True):
        if 'diffbot_categories' in node_data:
            if diffbot_cat in [s.strip() for s in node_data['diffbot_categories'].split(',')]:
                labels_dict[diffbot_uri] = 1
            else:
                labels_dict[diffbot_uri] = 0
    return labels_dict


class NodeFeatureEncoder:
    def __init__(self, nodes_data):
        self.nodes_data = nodes_data
        self.encoders = defaultdict(dict)
        self.feats = {node_type: {}
                      for node_type in nodes_data}

    def combine_feats(self, node_type, feat_names=None):
        if feat_names is None:
            feat_names = self.feats[node_type].keys()
        return np.concatenate([self.feats[node_type][feat_name] for feat_name in feat_names], axis=1)

    def encode_node_feat(self, node_type, node_feat_name, encode_method):
        if encode_method == 'ohe':
            encoder = preprocessing.OneHotEncoder(sparse=False)
            node_feats = [[node[node_feat_name]] for node in self.nodes_data[node_type]]
            new_feats = encoder.fit_transform(node_feats)
        elif encode_method == 'str_list_mlb':
            encoder = preprocessing.MultiLabelBinarizer()
            node_feats = [[s.strip() for s in node[node_feat_name].split(',')] for node in self.nodes_data[node_type]]
            new_feats = encoder.fit_transform(node_feats)
        elif encode_method.startswith('numerical'):
            median_val = median([float(node[node_feat_name])
                                 for node in self.nodes_data[node_type]
                                 if node[node_feat_name] != -1])
            node_feats = [[float(node[node_feat_name])]
                          if node[node_feat_name] != -1 else [median_val]
                          for node in self.nodes_data[node_type]]
            if encode_method == 'numerical_yeo_johnson':
                encoder = preprocessing.PowerTransformer(method='yeo-johnson')
                new_feats = encoder.fit_transform(node_feats)
            elif encode_method == 'numerical_box_cox':
                encoder = preprocessing.PowerTransformer(method='box-cox')
                new_feats = encoder.fit_transform([[float(node[node_feat_name])] for node in self.nodes_data[node_type]])
        elif encode_method == 'identity':
            new_feats = [node[node_feat_name] for node in self.nodes_data[node_type]]
            encoder = None
        else:
            raise NotImplementedError(f'{encode_method} encoding method is not supported')

        self.encoders[node_type][node_feat_name] = encoder
        self.feats[node_type][node_feat_name] = new_feats


def get_relabel_node_by_type_mapping(nxgraph, treat_orgs_same, treat_locations_same):
    node_type_counter = defaultdict(int)
    node_label_mapping = defaultdict(dict)
    for i, node, in nxgraph.nodes(data=True):
        node_type = node['node_type']
        if treat_orgs_same and node_type in ORG_ENTITY_TYPES:
            node_type = 'Aggr_Organization'
        elif treat_locations_same and node_type in LOCATION_ENTITY_TYPES:
            node_type = 'Aggr_Location'
        node_label_mapping[node_type][i] = node_type_counter[node_type]
        node_type_counter[node_type] += 1
    return node_label_mapping


class DiffbotGraphDataset(DGLDataset, ABC):
    def __init__(self, name, natts_path, raw_dir=None, treat_orgs_same=True,
                 treat_locations_same=True, save_dir=None,

                 force_reload=False, verbose=False):
        self.graph = None
        self.homo_graph = None
        self.node_type_relabel_mapping = None
        self.relations_dict = defaultdict(lambda: ([], []))
        if save_dir is None:
            self._save_dir = join(get_data_path(), 'processed', name)
        else:
            self._save_dir = save_dir
        create_dir_if_not_exists(self._save_dir)
        if raw_dir is None:
            raw_dir = join(get_data_path(), 'raw', name)
        self.treat_orgs_same = treat_orgs_same
        self.treat_locations_same = treat_locations_same
        self.gexf_path = join(raw_dir, 'graph.gexf')
        self.natts_path = natts_path
        self.node_feat_encoder = None
        super().__init__(name, raw_dir=raw_dir, save_dir=self._save_dir,
                         force_reload=force_reload, verbose=verbose)
        self.process_feats()

    def download(self):
        assert exists(self.raw_dir), f"Could not find raw file directory {self.raw_dir}"
        assert exists(self.gexf_path), f"Need {self.gexf_path} file"

    def save(self):
        save_graphs(join(self.save_dir, "graph.bin"), [self.graph, self.homo_graph])
        save_info(join(self.save_dir, "relations_dict.pkl"), self.relations_dict)
        info = {'num_nodes': self.num_nodes, 'num_rels': self.num_rels}
        save_info(join(self.save_dir, "info.pkl"), info)
        save_info(join(self.save_dir, "node_type_relabel_mapping.pkl"), self.node_type_relabel_mapping)

    def has_cache(self):
        exists_dgl_graph = exists(join(self.save_dir, "graph.bin"))
        exists_relations_dict = exists(join(self.save_dir, "relations_dict.pkl"))
        exists_info = exists(join(self.save_dir, "info.pkl"))
        exists_relabel_mapping = exists(join(self.save_dir, "node_type_relabel_mapping.pkl"))
        return exists_dgl_graph and exists_relations_dict \
               and exists_info and exists_relabel_mapping

    def load(self):
        graphs = load_graphs(join(self.save_dir, "graph.bin"))[0]
        self.graph, self.homo_graph = graphs[0], graphs[1]
        self.relations_dict = load_info(join(self.save_dir, "relations_dict.pkl"))
        self.node_type_relabel_mapping = load_info(join(self.save_dir, "node_type_relabel_mapping.pkl"))
        info = load_info(join(self.save_dir, "info.pkl"))
        self._num_nodes = info['num_nodes']
        self._num_rels = info['num_rels']

    def process(self):
        nxgraph = nx.read_gexf(self.gexf_path)
        nxgraph.remove_edges_from(nx.selfloop_edges(nxgraph))

        # relabel nodes based on node_type mapping
        self.node_type_relabel_mapping = get_relabel_node_by_type_mapping(nxgraph,
                                                                          treat_locations_same=self.treat_locations_same,
                                                                          treat_orgs_same=self.treat_orgs_same)

        # build the bipartite edges in each metapath and its reverse
        for s, t, relation in nxgraph.edges.data('relation_name'):
            node_type_s = nxgraph.nodes[s]['node_type']
            node_type_t = nxgraph.nodes[t]['node_type']
            if self.treat_orgs_same and node_type_s in ORG_ENTITY_TYPES:
                node_type_s = 'Aggr_Organization'
            if self.treat_orgs_same and node_type_t in ORG_ENTITY_TYPES:
                node_type_t = 'Aggr_Organization'
            if self.treat_locations_same and node_type_s in LOCATION_ENTITY_TYPES:
                node_type_s = 'Aggr_Location'
            if self.treat_locations_same and node_type_t in LOCATION_ENTITY_TYPES:
                node_type_t = 'Aggr_Location'
            source = self.node_type_relabel_mapping[node_type_s][s]
            target = self.node_type_relabel_mapping[node_type_t][t]

            self.relations_dict[(node_type_s, relation, node_type_t)][0].append(source)
            self.relations_dict[(node_type_s, relation, node_type_t)][1].append(target)

            relation_reverse = relation + '_reverse'
            self.relations_dict[(node_type_t, relation_reverse, node_type_s)][0].append(target)
            self.relations_dict[(node_type_t, relation_reverse, node_type_s)][1].append(source)

        self.graph = dgl.heterograph(self.relations_dict)
        self._num_nodes = nxgraph.number_of_nodes()
        self._num_rels = len(self.relations_dict)
        self.relations_dict = dict(self.relations_dict)
        self.homo_graph = dgl.to_homogeneous(self.graph)

    def process_feats(self, nxgraph=None, feats_path=None):
        with open(self.natts_path, 'rb') as f:
            natts_dict = json.load(f)
        if nxgraph is None:
            nxgraph = nx.read_gexf(self.gexf_path)
        nodes_data = {node_type: [None for _ in range(self.graph.number_of_nodes(node_type))]
                      for node_type in self.graph.ntypes}
        feats_path = feats_path if feats_path else join(self.raw_dir, 'feats.pkl')
        if exists(feats_path):
            feats_df = pd.read_pickle(feats_path)
            feats_df.set_index('diffbot_uri', inplace=True)
            feats_df = feats_df.to_dict('index')
        else:
            feats_df = None
        for i, node_data in nxgraph.nodes(data=True):
            node_type = node_data['node_type']
            if self.treat_orgs_same and node_type in ORG_ENTITY_TYPES:
                node_type = 'Aggr_Organization'
            elif self.treat_locations_same and node_type in LOCATION_ENTITY_TYPES:
                node_type = 'Aggr_Location'
            if feats_df:
                node_data['embds'] = feats_df[i]['embds']
            node_num = self.node_type_relabel_mapping[node_type][i]
            nodes_data[node_type][node_num] = node_data
        assert all([not (None in nodes_info) for nodes_info in nodes_data.values()]), 'None in node data missing node features for constructed graph'
        self.node_feat_encoder = NodeFeatureEncoder(nodes_data)
        for node_type, encode_info in natts_dict.items():
            for encode_method, natt_name in encode_info:
                self.node_feat_encoder.encode_node_feat(node_type, natt_name, encode_method)
        for node_type in self.graph.ntypes:
            for node_feat_name, feats in self.node_feat_encoder.feats[node_type].items():
                self.graph.nodes[node_type].data[node_feat_name] = torch.tensor(feats)
        self.graph.ndata['feats'] = {node_type: torch.tensor(self.node_feat_encoder.combine_feats(node_type))
                                     for node_type in self.node_feat_encoder.feats}

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_rels(self):
        return self._num_rels

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph

    def __len__(self):
        return 1


class DiffbotGraphDatasetNodeLabels(DiffbotGraphDataset):
    def __init__(self, name, market, market_source, natts_path, metapaths=None, tvt_split=None, raw_dir=None,
                 treat_locations_same=True, save_dir=None,
                 force_reload=False, verbose=False):

        if tvt_split is None:
            self.tvt_split = [0.6, 0.2]
        else:
            assert type(tvt_split) is list and len(tvt_split) == 2, 'expected two numbers for tvt split'
        self.market = market
        self.market_source = market_source
        market_name = '_'.join(market.lower().replace('-', ' ').split())
        if save_dir is None:
            save_dir = join(get_data_path(), 'processed', name)
        self.metapaths = metapaths  # a list of lists of metapaths for each node type
        self.train_idx, self.val_idx, self.test_idx = None, None, None
        self.target_node_indices = None
        self.labels, self.labels_info = [], []
        super().__init__(name, natts_path, raw_dir, treat_orgs_same=True, treat_locations_same=treat_locations_same,
                         save_dir=save_dir, force_reload=force_reload,
                         verbose=verbose)
        self.etype_to_etype_id_map = {e_type: etype_id for etype_id, e_type in enumerate(self.graph.canonical_etypes)}
        self.ntype_to_ntype_id_map = {n_type: ntype_id for ntype_id, n_type in enumerate(self.graph.ntypes)}
        self.num_classes = len(np.unique(self.labels))
        self.node_type_w_labels = 'Aggr_Organization'  # used to tell HAN which nodes we focusing on
        if self.num_classes == 2:
            self.label_names = [f'{market_name}_negative', f'{market_name}_positive']
        else:
            self.label_names = [str(i) for i in range(self.num_classes)]
        if self.metapaths:
            self.metapath_node_inds = [self.ntype_to_ntype_id_map[metapath[0][0][0]] for metapath in self.metapaths]
        else:
            self.metapath_node_inds = []

    def process(self):
        print('Processing graph')
        super().process()
        self._process()

    def _process(self):
        nxgraph = nx.read_gexf(self.gexf_path)
        nype_to_ntype_id_map = {n_type: ntype_id for ntype_id, n_type in enumerate(self.graph.ntypes)}
        if self.market_source == 'diffbot':
            labels_dict = combine_diffbot_categories_labels_w_graph(nxgraph, self.market)
        else:
            raise NotImplementedError(f'Market source {self.market_source} not implemented')
        # labels only for a specific type of node
        node_types = self.homo_graph.ndata[dgl.NTYPE]
        # inds of nodes of whole graph that below to Aggr_organization node type
        node_label_inds = np.where(node_types == nype_to_ntype_id_map['Aggr_Organization'])[0]
        label_inds = []
        self.labels_info = []
        self.labels = []
        for diffbot_uri, label in labels_dict.items():
            node_type_id = self.node_type_relabel_mapping['Aggr_Organization'][diffbot_uri]  # inds of Aggr_Organization which match diffbot uri
            label_inds.append(node_type_id)
            self.labels.append(label)
            self.labels_info.append({'diffbot_uri': diffbot_uri, 'label': label,
                                     'name': nxgraph.nodes[diffbot_uri].get('name'),
                                     'url': nxgraph.nodes[diffbot_uri].get('homepage_uri')})
        self.target_node_indices = node_label_inds[label_inds]
        float_mask = np.random.permutation(np.linspace(0, 1, len(self.labels)))
        self.train_idx = np.where(float_mask <= self.tvt_split[0])[0]
        train_and_val_split = self.tvt_split[0] + self.tvt_split[1]
        self.val_idx = np.where((float_mask > self.tvt_split[0]) & (float_mask <= train_and_val_split))[0]
        self.test_idx = np.where(float_mask > train_and_val_split)[0]


    def save(self):
        super().save()
        labels_and_splits_dict = {
            'target_node_indices': self.target_node_indices,
            'labels': self.labels,
            'train_idx': self.train_idx,
            'val_idx': self.val_idx,
            'test_idx': self.test_idx,
            'label_info': self.labels_info
        }
        save_info(join(self.save_dir, 'labels_and_splits.pkl'), labels_and_splits_dict)

        # save_info(join(self.save_dir, 'metapaths.pkl'), self.metapaths)

    def has_cache(self):
        has_labels_and_splits = exists(join(self.save_dir, 'labels_and_splits.pkl'))
        return super().has_cache() and has_labels_and_splits

    def load(self):
        super().load()
        labels_and_splits_dict = load_info(join(self.save_dir, 'labels_and_splits.pkl'))
        self.target_node_indices = labels_and_splits_dict['target_node_indices']
        self.labels = labels_and_splits_dict['labels']
        self.train_idx = labels_and_splits_dict['train_idx']
        self.val_idx = labels_and_splits_dict['val_idx']
        self.test_idx = labels_and_splits_dict['test_idx']
        self.labels_info = labels_and_splits_dict['label_info']