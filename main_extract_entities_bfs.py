import argparse
import json
import logging
from os.path import exists, join

import asyncio
import yaml

from data_preprocess import DiffBotGraphTraverser


def create_parser():
    parser = argparse.ArgumentParser('main_extract_entities')
    parser.add_argument('--config_file', dest='config_file',
                        type=argparse.FileType(mode='r'),
                        help='Path to yaml argument file')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Diffbot API key to extract graph from')
    parser.add_argument('--saved_entities_folder', nargs="+", default=[],
                        help='List of folders to saved entities if any.')
    parser.add_argument('--output_folder', type=str,
                        default=None)
    parser.add_argument('--num_nodes', type=int, default=500,
                        help='Number of nodes with downloaded entities in the '
                             'graph including the ones already saved as specified'
                             'by saved_entity_folders')
    parser.add_argument('--starting_node_uris', type=list, nargs="+",
                        help='List of starting diffbot uris to begin BFS on Diffbot KG API',
                        default=['C4PQqufphPOam9WR7_U0DRQ'])  # Sequoia Capital
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of asynchronous process workers')
    parser.add_argument('--investor_uri_mapping_file', type=str, default=None,
                        help='Path to json file containing investor name to uri file name')
    parser.add_argument(
        '-d', '--debug',
        help="Debug logging",
        action="store_const", dest="loglevel", const=logging.DEBUG,
        default=logging.INFO,
    )
    return parser


def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(args.config_file, Loader=yaml.FullLoader)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            arg_dict[key] = value
    assert args.api_key is not None, 'Need to pass in Diffbot API Key'
    return args


def main(args):
    diffbot_api_key = args.api_key
    saved_entities_folder = args.saved_entities_folder
    output_folder = args.output_folder
    num_nodes = args.num_nodes
    starting_node_uris = args.starting_node_uris
    investor_uri_mapping_file = args.investor_uri_mapping_file
    if investor_uri_mapping_file:
        with open(investor_uri_mapping_file) as f:
            investor_uri_mapping = json.load(f)
    else:
        investor_uri_mapping = None

    if exists(join(output_folder, 'entities')):
        saved_entities_folder.append(join(output_folder, 'entities'))

    async def main_routine():
        diffbot_kg_traverser = DiffBotGraphTraverser(starting_node_uris,
                                                     diffbot_api_key, num_nodes,
                                                     output_folder=output_folder,
                                                     saved_entities_folders=saved_entities_folder,
                                                     investor_uri_mapping=investor_uri_mapping
                                                     )
        diffbot_kg_traverser.logger.setLevel(args.loglevel)
        await diffbot_kg_traverser.explore_graph(args.num_workers)

    asyncio.run(main_routine())


if __name__ == '__main__':
    from pprint import pprint
    args = parse_args(create_parser())
    pprint(args)
    main(args)