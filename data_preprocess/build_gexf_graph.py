from collections import defaultdict
from os.path import join, basename, exists
from glob import glob
import json

import networkx as nx
import pandas as pd
from tqdm import tqdm

from .diffbot_constants import LINKS_TO_ID, ORG_ENTITY_TYPES
from .diffbot_process_entities import (
    get_links_from_org_entity,
    get_links_from_person_entity,
    parse_org_links_for_edges,
    parse_person_links_for_edges
)

from .util import create_dir_if_not_exists

FILTERS = ['saved_entities_and_in_uri_info_map',
           'saved_entities_and_in_uri_info_map_or_subregion',
           'saved_entities_or_subregion',
           'saved_entities',
           'all_entities_with_name_in_uri_info_map'
           ]


def build_gexf_from_uris_and_entity_folders(uri_info_map, entity_folders, investor_uri_map=None,
                                            take_largest_cc=False,
                                            node_filter='saved_entities'):
    assert node_filter in FILTERS, f'unsupported filter {FILTERS}'
    in_uri_info_map_only = 'in_uri_info_map' in node_filter

    edges, all_uris = build_relations_w_given_entities(entity_folders,
                                                       investor_uri_map=investor_uri_map,
                                                       saved_entities_uri_info_map=uri_info_map,
                                                       in_uri_info_map_only=in_uri_info_map_only)
    id_to_link_map = {v: k for k, v in LINKS_TO_ID.items()}
    edges['relation_name'] = edges['relation'].apply(lambda x: id_to_link_map[x])

    g = nx.from_pandas_edgelist(edges, source='head', target='tail',
                                edge_attr=['relation_name'], create_using=nx.DiGraph())
    uri_info_map_from_folders, locations_dict = build_uri_infomap_from_entity_folders(entity_folders, build_locations=True)
    if node_filter == 'saved_entities_and_in_uri_info_map':
        nodes_after_filter = [n for n in g.nodes() if
                              uri_info_map.get(n) and uri_info_map[n].get('has_entity')
                              and n in uri_info_map_from_folders]
    elif node_filter == 'saved_entities_and_in_uri_info_map_or_subregion':
        nodes_after_filter = [n for n in g.nodes() if
                              (uri_info_map.get(n) and uri_info_map[n].get('has_entity'
                                                                           and n in uri_info_map_from_folders))
                              or (n in locations_dict and locations_dict[n]['node_type'] == 'SubRegion')]
    elif node_filter == 'saved_entities':
        nodes_after_filter = [n for n in g.nodes() if
                              uri_info_map_from_folders.get(n)]
    elif node_filter == 'saved_entities_or_subregion':
        nodes_after_filter = [n for n in g.nodes() if uri_info_map_from_folders.get(n)
                              or (n in locations_dict and locations_dict[n]['node_type'] == 'SubRegion')]
    elif node_filter == 'all_entities_with_name_in_uri_info_map':
        nodes_after_filter = list(uri_info_map.keys())
    else:
        raise NotImplementedError
    g_filtered = g.subgraph(nodes_after_filter).copy()

    all_info_map = {**uri_info_map, **uri_info_map_from_folders, **locations_dict}
    if 'saved' in node_filter:
        g = add_entity_info(g_filtered, entity_folders, all_info_map)
    else:
        g = add_entity_info(g_filtered, [], all_info_map)
    if take_largest_cc:
        largest_cc = max(nx.weakly_connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
    g_nodes_df = pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient='index')

    return g, g_nodes_df


def add_entity_info(graph, entity_folders, uri_info_map):
    n_atts = {}
    for node in tqdm(graph.nodes, desc='getting node attributes'):
        for folder in entity_folders:
            fname = join(folder, f"{node}.json")
            if exists(fname):
                with open(fname) as f:
                    entity_info = json.load(f)
                if 'enhanced' in entity_info:
                    entity_info = entity_info['enhanced']
                elif 'data' in entity_info:
                    if entity_info is None or entity_info['data'] is None or len(entity_info['data']) == 0:
                        return None
                    entity_info = entity_info['data'][0]
                node_type = uri_info_map[node]['node_type']
                n_atts[node] = add_node_information(entity_info, node_type)
                break
        if node not in n_atts:
            n_atts[node] = uri_info_map[node]
    nx.set_node_attributes(graph, n_atts)
    return graph


def build_relations_w_given_entities(entities_folders, save_folder=None,
                                     saved_entities_uri_info_map=None,
                                     investor_uri_map=None, in_uri_info_map_only=False):
    """
    Given a folder of organization and/or person entity jsons creates all edge
    relations from these entities. This function saves a edges.tsv and
    uri_info_map.json.

    Args:
        entities_folder (str): The path to the folder containing the entities.
        save_folder (str): The path to the folder to save the relations to.
        saved_entities_uri_info_map (dict): A dictionary containing all saved
            entities that may not be in the entities folder. Helps to resolve
            names for linked diffbot uris.
        investor_uri_map (dict): A dictionary containing the names of investors
            entities as keys and the corresponding diffbot uri as values.

    Returns:
        edges DataFrame and dict of all explore nodes to node info
    """

    if saved_entities_uri_info_map is None:
        saved_entities_uri_info_map = {}
        assert in_uri_info_map_only is False, 'need uri_info_map if in_uri_info_map_only set to True'
    all_files = []
    for folder in entities_folders:
        files = glob(join(folder, '*'))
        all_files.extend(files)
    all_edges = []
    all_explored_nodes = {}

    for file in tqdm(all_files, 'building edges'):
        diffbot_uri = basename(file).split('.')[0]
        if in_uri_info_map_only and diffbot_uri not in saved_entities_uri_info_map:
            continue
        with open(file, 'rb') as f:
            entity_info = json.load(f)
        if 'enhanced' in entity_info:
            entity_info = entity_info['enhanced']
        elif 'data' in entity_info:
            entity_info = entity_info['data'][0]
        if diffbot_uri not in saved_entities_uri_info_map:
            all_explored_nodes[diffbot_uri] = {'name': entity_info['name'],
                                               'node_type': entity_info['type'],
                                               'has_saved_entity': True}
        else:
            all_explored_nodes[diffbot_uri] = saved_entities_uri_info_map[diffbot_uri]
        if entity_info['type'] in ORG_ENTITY_TYPES:
            links = get_links_from_org_entity(entity_info)

            edges, _ = parse_org_links_for_edges(diffbot_uri, links,
                                                 all_explored_nodes=all_explored_nodes,
                                                 saved_entities_uri_info_map=saved_entities_uri_info_map,
                                                 investors_uri_mapping=investor_uri_map)

        elif entity_info['type'] == 'Person':
            links = get_links_from_person_entity(entity_info)
            edges, _ = parse_person_links_for_edges(diffbot_uri, links,
                                                    all_explored_nodes=all_explored_nodes,
                                                    saved_entities_uri_info_map=saved_entities_uri_info_map)
        else:
            raise ValueError(f"Entity type {entity_info['type']} not supported for parsing")
        all_edges.extend(edges)
    edges_df = pd.DataFrame(all_edges)
    if save_folder:
        create_dir_if_not_exists(save_folder)
        all_uri_name_path = join(save_folder, 'uri_info_map.json')
        edges_path = join(save_folder, 'edges.tsv')

        with open(all_uri_name_path, 'w') as f:
            json.dump(all_explored_nodes, f, indent=4)
        print(f'Saved all uri name mapping to {all_uri_name_path}')
        edges_df.to_csv(edges_path, sep='\t')
        print(f'Saved edges to {edges_path}')
    return edges_df, all_explored_nodes


def build_uri_infomap_from_entity_folders(entity_folders, build_locations=False):
    """
    Given folders containing entities, builds a dictionary containing all
    diffbot uris to basic entity info.

    Args:
        entity_folders (list of str): The paths to the folders containing the
            downloaded entities.

    Returns:
        A dictionary
    """
    all_files = []
    uri_info_map = {}
    for folder in entity_folders:
        files = glob(join(folder, '*'))
        all_files.extend(files)
    all_locations_dict = {}
    for file in tqdm(all_files, desc='looping through all entities'):
        with open(file, 'rb') as f:
            entity_info = json.load(f)
        if 'enhanced' in entity_info:
            entity_info = entity_info['enhanced']
        elif 'data' in entity_info:
            entity_info = entity_info['data'][0]
        uri = basename(file).split('.')[0]

        uri_info_map[uri] = {'name': entity_info['name'],
                             'node_type': entity_info['type'],
                             'has_entity': True}
        if build_locations:
            locations = entity_info.get('locations', [])
            for location in locations:
                locations_dict = get_links_from_location_dict(location)
                all_locations_dict.update(locations_dict['located_in'])
    return uri_info_map, all_locations_dict


def get_links_from_location_dict(location_dict):
    country = location_dict.get('country')
    city = location_dict.get('city')
    subregion = location_dict.get('subregion')
    region = location_dict.get('region')

    all_links = defaultdict(dict)

    def get_uri_and_add_to_dict(info, links_dict, link_type, node_type):
        if 'diffbotUri' in info:
            uri = info['diffbotUri'].split('/')[-1]
            links_dict[link_type][uri] = {'name': info.get('name'),
                                          'node_type': node_type}
    if country is not None:
        get_uri_and_add_to_dict(country, all_links, 'located_in', 'Country')
    if city is not None:
        get_uri_and_add_to_dict(city, all_links, 'located_in', 'City')
    if subregion is not None:
        get_uri_and_add_to_dict(subregion, all_links, 'located_in', 'SubRegion')
    if region is not None:
        get_uri_and_add_to_dict(region, all_links, 'located_in', 'Region')
    return all_links


def add_node_information(node_info, node_type):
    if node_type in ORG_ENTITY_TYPES:
        name = node_info.get('name')
        nb_active_employee_edges = node_info.get('nbActiveEmployeeEdges', -1)
        nb_employee_max = node_info.get('nbEmployeesMax', -1)
        nb_employee_min = node_info.get('nbEmployeeMin', -1)
        is_public = node_info.get('isPublic', -1)
        diffbot_categories = node_info.get('categories', [])
        diffbot_categories = [cat['name'] for cat in diffbot_categories]
        diffbot_cats_str = ', '.join(sorted(diffbot_categories)) if diffbot_categories else '-1'
        naics_classifications = node_info.get('naicsClassification', [])
        naics_classifications = [clss['name'] for clss in naics_classifications]
        naics_str = ', '.join(sorted(naics_classifications)) if naics_classifications else '-1'
        description = node_info.get('description', -1)
        if type(description) is str:
            description = description.encode('utf-8').replace(b'\x03', b'').replace(b'\x0b', b'').decode()

        uris = node_info.get('allUris')
        uris_str = ', '.join(sorted(uris)) if uris else '-1'
        n_atts = {'name': name,
                  'node_type': node_type,
                  'homepage_uri': node_info.get('homepageUri', ''),
                  'all_uris': uris_str,
                  'nb_active_employee_edges': nb_active_employee_edges,
                  'nb_employee_max': nb_employee_max,
                  'nb_employee_min': nb_employee_min,
                  'is_public': int(is_public),
                  'diffbot_categories': diffbot_cats_str,
                  'na_ics_classifications': naics_str,
                  'description': description,
                  'importance': node_info.get('importance', -1),
                  'has_entity': True
                  }
    elif node_type == 'Person':
        description = node_info.get('description', -1)
        if type(description) is str:
            description = description.encode('utf-8').replace(b'\x03', b'').replace(b'\x0b', b'').decode()

        n_atts= {'name': node_info.get('name', '-1'),
                 'node_type': node_type,
                 'age': node_info.get('age', -1),
                 'description': description,
                 'importance': node_info.get('importance', -1),
                 'has_entity': True
                 }
    else:
        raise ValueError(f'node type {node_type} not supported for obtaining data')
    return n_atts