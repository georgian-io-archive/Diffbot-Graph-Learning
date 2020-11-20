import argparse
import json
from os.path import join

import networkx as nx
import yaml

from data_preprocess.build_gexf_graph import FILTERS, build_gexf_from_uris_and_entity_folders
from data_preprocess.util import get_temp_path, create_dir_if_not_exists


def create_parser():
    parser = argparse.ArgumentParser('main_extract_entities')
    parser.add_argument('--config_file',
                        type=argparse.FileType(mode='r'),
                        help='Path to yaml argument file')
    parser.add_argument('--saved_entities_folder', nargs="+", default=None,
                        help='List of folders to saved entities. Should have at least one')
    parser.add_argument('--uri_info_map', type=str, default=None,
                        help='The path to a uri_info_map.json as saved by the enhance or bfs'
                             'diffbot querying')
    parser.add_argument('--output_folder', type=str, default=join(get_temp_path(), 'graphs'))
    parser.add_argument('--output_file_name', type=str,
                        default=None)
    parser.add_argument('--investor_uri_mapping', type=str, default=None,
                        help='The path to a investor_uri_mapping.json as saved '
                             'by the bfs diffbot querying')
    parser.add_argument('--take_largest_cc', type=str2bool, const=True, nargs='?',
                        default=True, help='Whether or not to take the largest connected component of the graph')
    parser.add_argument('--node_filter', default='saved_entities_and_in_uri_info_map_or_subregion',
                        choices=FILTERS, help='Filters we place on the nodes, generally we would'
                                              'want to have nodes that have saved entities as specified by'
                                              'saved_entities_folder')
    return parser


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(args.config_file, Loader=yaml.FullLoader)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            arg_dict[key] = value
    assert args.saved_entities_folder is not None, "Should have at least one saved_entities_folder"
    assert args.uri_info_map is not None, "Need to have uri_info_map"
    return args


def main(args):
    if args.output_file_name is None:
        append_str = '' if not args.take_largest_cc else '_LCC'
        args.output_file_name = f"{args.node_filter}{append_str}.gexf"

    if args.output_folder is None:
        args.output_folder = join(get_temp_path(), 'graphs')

    with open(args.uri_info_map) as f:
        uri_info_map = json.load(f)

    if args.investor_uri_mapping:
        with open(args.investor_uri_mapping) as f:
            investor_uri_mapping = json.load(f)
    else:
        investor_uri_mapping = None

    g, g_nodes_df = build_gexf_from_uris_and_entity_folders(uri_info_map,
                                                            args.saved_entities_folder,
                                                            investor_uri_map=investor_uri_mapping,
                                                            take_largest_cc=args.take_largest_cc,
                                                            node_filter=args.node_filter)
    create_dir_if_not_exists(args.output_folder)
    nx.write_gexf(g, join(args.output_folder, args.output_file_name))
    out_csv_name = args.output_file_name.split('.')[0] + '.csv'
    g_nodes_df.to_csv(join(args.output_folder,out_csv_name))
    print(f'saved graph to {join(args.output_folder, args.output_file_name)}')
    print(f'saved node info to {join(args.output_folder, out_csv_name)}')


if __name__ == '__main__':
    from pprint import pprint
    args = parse_args(create_parser())
    pprint(args.__dict__)
    main(args)