"""
Extract organization and or person entities given url or name or both
"""

import argparse
import asyncio
import logging

import numpy as np
import pandas as pd
from pprint import pprint
import yaml

from data_preprocess import download_enhance_entities


def create_parser():
    parser = argparse.ArgumentParser('main_extract_entities')
    parser.add_argument('--config_file',
                        type=argparse.FileType(mode='r'),
                        help='Path to yaml argument file')
    parser.add_argument('--api_key', type=str, default=None,
                        help='Diffbot API key to extract graph from')
    parser.add_argument('--output_folder', type=str,
                        default=None)
    parser.add_argument('--org_info_csv_path', type=str,
                        default=None,
                        help='The path to a csv containing company information '
                             'for the Diffbot Enhance API. The csv should contain'
                             'at least one of the following two columns: name, url'
                             'for each company. The name and or url is used by'
                             'the Diffbot Enhance api to find the associated company.')
    parser.add_argument('--people_info_csv_path', type=str,
                        default=None,
                        help='The path to a csv containing person information '
                             'for the Diffbot Enhance API. The csv should contain'
                             'at least one of the following two columns: name, url'
                             'for each person. The name and or url is used by'
                             'the Diffbot Enhance api to find the associated person.')
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
    assert args.org_info_csv_path or args.people_info_csv_path, \
        'At least organization info or person info csvs needs to be specified'
    if args.org_info_csv_path:
        orgs_info = pd.read_csv(args.org_info_csv_path)
        orgs_info.replace('', np.nan, inplace=True)
        orgs_info.replace({np.nan: None}, inplace=True)
        orgs_info = list(orgs_info.T.to_dict().values())
        pprint(orgs_info)
    else:
        orgs_info = None

    if args.people_info_csv_path:
        people_info = pd.read_csv(args.people_info_csv_path)
        people_info.replace('', np.nan, inplace=True)
        people_info.replace({np.nan: None}, inplace=True)
        people_info = list(people_info.T.to_dict().values())
        pprint(people_info)
    else:
        people_info = None

    asyncio.run(download_enhance_entities(args.api_key, args.output_folder,
                                          orgs_info=orgs_info,
                                          people_info=people_info,
                                          log_level=args.loglevel))


if __name__ == '__main__':
    args = parse_args(create_parser())
    pprint(args.__dict__)
    main(args)