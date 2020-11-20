from copy import deepcopy
import json
import logging
import pickle
import time

from os.path import join, exists

import asyncio
import aiofiles
import aiohttp
import async_timeout
import pandas as pd
from pprint import pformat

from .diffbot_constants import LINKS_TO_ID, ORG_ENTITY_TYPES
from .diffbot_process_entities import get_links_from_org_entity, get_links_from_person_entity
from .util import get_data_path, create_dir_if_not_exists, log_wrap, get_current_ts


class DiffbotRequests:
    """
    Class to encapsulate async requests to Diffbot API. Includes Diffbot
    enhance API and the Diffbot knowledge graph API for entity id queries.

    Parameters:
        api_key (str): Diffbot API key
        kg_endpoint (str): The Diffbot knowledge graph API endpoint
        enhance_endpoint (str): The Diffbot enhance API endpoint
        logger (:class: `logging.Logger`): Logger object used for logging
        session (:class: `aiohttp.ClientSession`): HTTP session object for repeated
            queries to the endpoints. From aiohttp docs, keeping one session
            enables connection pooling.

    """
    KG_ENDPOINT = 'https://kg.diffbot.com/kg/dql_endpoint'
    ENHANCE_ENDPOINT = 'https://kg.diffbot.com/kg/enhance_endpoint'

    def __init__(self, api_key, logger, session):
        self.api_key = api_key
        self.logger = logger
        self.session = session

    async def fetch_kg(self, node, timeout=10):
        params = self.build_diffbot_uri_request(node, self.api_key)
        try:
            with async_timeout.timeout(timeout):
                async with self.session.get(self.KG_ENDPOINT, params=params) as response:
                    if response.status != 200:
                        self.logger.warning(f'Status code {response.status} received. Skipping uri: {node}')
                        self.logger.warning(await response.text())
                        return None, None
                    return await response.json(), await response.text()
        except asyncio.TimeoutError as e:
            self.logger.warning(e)
            return None, None

    async def fetch_enhance(self, type, name, url=None, timeout=10):
        params = self.build_diffbot_enhance_request(type, self.api_key, name, url)
        try:
            with async_timeout.timeout(timeout):
                async with self.session.get(self.ENHANCE_ENDPOINT, params=params) as response:
                    if response.status != 200:
                        self.logger.warning(f'Status code {response.status} received. Skipping name: {name}, url: {url}')
                        self.logger.warning(await response.text())
                        return None, None
                    entity_identifier = name if name is not None else url
                    self.logger.debug(f'Fetch response 200 on {entity_identifier}')
                    return await response.json(), await response.text()
        except asyncio.TimeoutError as e:
            self.logger.warning('asyncio Timeout')
            return await self.fetch_enhance(type, name, url, timeout*2)

    @staticmethod
    def build_diffbot_uri_request(uri, api_key):
        params = {
            "token": api_key,
            "type": "query",
            "query": f'id:{uri}',
            "size": 1
        }
        return params

    @staticmethod
    def build_diffbot_enhance_request(type, api_key, name, url):
        params = {
            'token': api_key,
            'type': type
        }
        if name is not None:
            params['name'] = name
        if url is not None:
            params['url'] = url
        return params


class DiffBotEnhanceDownloader:

    def __init__(self, api_key, output_folder,
                 session):
        self.api_key = api_key
        if output_folder is None:
            output_folder = join(get_data_path(), f'enhance_API_{get_current_ts()}')
        self.output_folder = output_folder
        self.entity_save_folder = join(output_folder, 'entities')
        create_dir_if_not_exists(self.entity_save_folder)
        self.logger = log_wrap(log_name='diffbot_traverser', console=True)

        self.diffbot_requests = DiffbotRequests(api_key, self.logger, session)
        self.uri_info_mapping = {}

    def save_uri_info_mapping(self):
        save_path = join(self.output_folder, 'uri_info_map.json')
        with open(save_path, 'w') as f:
            json.dump(self.uri_info_mapping, f, indent=4)

    async def save_diffbot_entity(self, entity_info, entity_type):
        allowed_entity_types = ['Person', 'Organization']
        assert entity_type in allowed_entity_types, f'entity_type needs to be one of {allowed_entity_types}'
        assert self.diffbot_requests is not None, 'Need to have DiffbotRequests object'
        name = entity_info.get('name')
        url = entity_info.get('url')
        resp, text = await self.diffbot_requests.fetch_enhance(entity_type,
                                                               name, url)
        if resp is None or resp['enhanced'] is None \
                or 'diffbotUri' not in resp['enhanced']:
            self.logger.info(f'No match for entity with url: {url} and name: {name}')
            return
        diffbot_uri = resp['enhanced']['diffbotUri'].split('/')[-1]
        entity_info['has_entity'] = True
        self.uri_info_mapping[diffbot_uri] = entity_info
        async with aiofiles.open(join(self.entity_save_folder, f'{diffbot_uri}.json'), 'w') as f:
            await f.write(text)

        if len(self.uri_info_mapping) % 25 == 0:
            self.logger.info(f'Saved {len(self.uri_info_mapping)} entities')


class DiffBotGraphTraverser:
    """ Class for handling BFS traversal of Diffbot graph.

    This class does BFS traversal and saves the results of Diffbot Knowledge Graph
    through repeated asynchronous queries to its enhance and KG endpoints.
    Enhance endpoint is used when we do not have the uri available
    (for example, a company's investor firms). The main method to run is
    `explore_graph`


    Parameters:
        starting_nodes (:obj:`list` of str): List of entity uris to begin BFS
        api_key (str): Diffbot API key
        graph_size (int): The number of entities to query before we stop
        traverse (bool): Whether we explore beyond our starting nodes
        output_folder (str): The folder to put the saved results in during BFS.
        visited (:obj: `dict` of str to :obj: dict): A dictionary mapping the
            entity uri to a dict containing the entity name and the entity node
            type. For ex: {"CPHquPDRoMxyYxFUP99t8lA": {"name": "Georgian", "node_type": "Organization"}}
        saved_entities_folders (:obj:`list` of str): List of folder paths containing diffbot entities
            with naming scheme {diffbot_uri}.json
        investor_uri_mapping (:obj: `dict` of str to str): A dictionary mapping
            investor firm name to its uri. This saves us a enhance API query if we
            have this mapping already for firm
    """

    def __init__(self, starting_nodes, api_key,
                 graph_size,
                 traverse=True, output_folder=None,
                 visited=None, saved_entities_folders=None,
                 investor_uri_mapping=None):
        self.api_key = api_key
        self.graph_size = graph_size
        self.api_called_nodes = set()
        self.explored_nodes = set()  # nodes we explored that we don't want the API to call anymore
        self.all_explored_nodes = {}  # includes the nodes put onto the queue but we have not explored yet
        if saved_entities_folders is None:
            self.saved_entities_folders = []
        else:
            self.saved_entities_folders = saved_entities_folders
        if visited is not None:
            self.explored_nodes = set(visited.keys())
            self.all_explored_nodes.update(visited)
        self.nodes_queue = asyncio.Queue()
        self.traverse = traverse
        if output_folder is None:
            self.output_folder = join(get_data_path(), f'{len(starting_nodes)}_starting_nodes_BFS_{get_current_ts()}')
        else:
            self.output_folder = output_folder
        create_dir_if_not_exists(join(self.output_folder, 'entities'))

        for node in starting_nodes:
            self.nodes_queue.put_nowait(node)

        self.investors_uri_mapping = {} if investor_uri_mapping is None else investor_uri_mapping
        self.edges = []
        self.logger = log_wrap(log_name='diffbot_traverser', console=True)
        self.diffbot_request = None
        self.saved_nodes = deepcopy(self.explored_nodes)

    async def explore_graph(self, num_workers=3):
        """
        Method to do BFS search on the Diffbot KG. Saves the result to
        `self.output_folder` containing the following:
            - `entities`: a folder containing the json of each entity that was
                queried using the Diffbot API with the file name being the entity uri.
                If a entity is found in one of the folders in `saved_entities_folders` then
                it will not add that entity to `self.output_folder`
            - `{num}_nodes`: a folder that contains intermittently saved result
                containing `num` nodes in `self.all_explored_nodes`. Inside this folder there is a
                `uri_info_map.json` containing the mapping between
                entity uri and entity name, a `edges.tsv` on the explored
                relations built so far, and `investors_uri_mapping.json` containing
                the obtained mappings between investor firm name and its
                entity uri
        Args:
            num_workers (int): The number of asynchronous tasks running concurrently
                to query the KG

        Returns:
            None
        """
        tasks = []
        for i in range(num_workers):
            tasks.append(self.__explore_graph_worker(f'WORKER{i}'))

        async with aiohttp.ClientSession() as session:
            self.diffbot_request = DiffbotRequests(self.api_key,
                                                   self.logger, session)
            await asyncio.gather(*tasks)
        self.__save_graph_info_to_file()
        self.__save_graph_info_to_file(final=True)

    def __save_graph_info_to_file(self, final=False):
        if final:
            folder = join(self.output_folder, f'{len(self.all_explored_nodes)}_nodes')
        else:
            folder = join(self.output_folder, f'final')
        create_dir_if_not_exists(folder)
        all_uri_name_path = join(folder, 'uri_info_map.json')
        investor_uri_path = join(folder, 'investor_uri_mapping.json')
        edges_path = join(folder, 'edges.tsv')
        api_called_entities_path = join(folder, 'api_called_entities.pkl')

        uri_info_map = {}
        for k, v in self.all_explored_nodes.items():
            if k in self.saved_nodes:
                new_v = deepcopy(v)
                new_v['has_entity'] = True
                uri_info_map[k] = new_v
            else:
                uri_info_map[k] = v

        with open(all_uri_name_path, 'w') as f:
            json.dump(uri_info_map, f, indent=4)
        self.logger.info(f'Saved all uri name mapping to {all_uri_name_path}')

        with open(investor_uri_path, 'w') as f:
            json.dump(self.investors_uri_mapping, f, indent=4)
        self.logger.info(f'Saved investors uri mapping to {investor_uri_path}')

        df = pd.DataFrame(self.edges)
        df.to_csv(edges_path, sep='\t')
        self.logger.info(f'Saved edges to {edges_path}')

        with open(api_called_entities_path, 'wb') as f:
            pickle.dump(list(self.api_called_nodes), f)
        self.logger.info(f'Saved API called nodes to {api_called_entities_path}')

    async def _file_exists(self, node):
        """
        check if saved entity already exists with node
        if saved entity exists return the saved entity dict, else return None
        """
        for folder in self.saved_entities_folders:
            fname = join(folder, f"{node}.json")
            if exists(fname):
                async with aiofiles.open(fname) as f:
                    data = await f.read()
                    entity_info = json.loads(data)
                if 'enhanced' in entity_info:
                    entity_info = entity_info['enhanced']
                elif 'data' in entity_info:
                    if entity_info is None or entity_info['data'] is None or len(entity_info['data']) == 0:
                        return None
                    entity_info = entity_info['data'][0]
                return entity_info
        return None

    async def __explore_graph_worker(self, name):
        while True:
            if self.traverse and len(self.explored_nodes) > self.graph_size:
                self.logger.info(f'Reached {self.graph_size} number nodes explored')
                self.logger.info(f'Total Number Times API called {len(self.api_called_nodes)}')
                self.logger.info(f'{name} terminating')
                break
            elif self.nodes_queue.empty():
                await asyncio.sleep(5)
                if self.nodes_queue.empty():
                    self.logger.info(f'Node queue is empty')
                    self.logger.info(f'{name} terminating')
                    break

            node = await self.nodes_queue.get()

            if node in self.explored_nodes:
                continue
            else:
                self.explored_nodes.add(node)
                await self.__process_entity(node, name)
            if (len(self.explored_nodes) + 1) % 250 == 0:
                self.logger.info(f'========= {len(self.explored_nodes)} Explored Nodes ========')
            # self.logger.info(f"{name} processed entity with name {self.all_explored_nodes[node]['name']}, and uri {node}")

    async def __process_entity(self, node, name):
        data = await self._file_exists(node)
        if data is None:  # entity json does not exist
            resp, text = await self.diffbot_request.fetch_kg(node)
            if resp is None or resp['data'] is None or len(resp['data']) == 0:
                return
            async with aiofiles.open(join(self.output_folder, 'entities', f'{node}.json'), 'w') as f:
                await f.write(text)
            data = resp['data'][0]
            self.api_called_nodes.add(node)
            if (len(self.api_called_nodes) + 1) % 250 == 0:
                self.logger.info(f'========= {len(self.explored_nodes)} API Explored Nodes ========')
                self.__save_graph_info_to_file()
            self.logger.info(f"{name} downloaded entity with name {data.get('name')}, and uri {node}")
        else:
            self.logger.info(f"{name} loaded from disk entity with name {data.get('name')}, and uri {node}")
        self.saved_nodes.add(node)
        self.all_explored_nodes[node] = {'name': data.get('name'), 'node_type': data.get('type')}

        if data['type'] in ORG_ENTITY_TYPES:
            await self.__process_org_entity(data, node)
        elif data['type'] == 'Person':
            self.__process_person_entity(data, node)

    async def __process_org_entity(self, data, node):
        all_links = get_links_from_org_entity(data)
        self.logger.debug(f"Links from org {data.get('name')} {node}: {pformat(dict(all_links), indent=4)}")
        for link_type, links in all_links.items():
            if link_type == 'investor_of':
                for name, info in links.items():
                    if name not in self.investors_uri_mapping:
                        diffbot_uri = info.get('diffbot_uri')
                        if diffbot_uri:
                            self.investors_uri_mapping[name] = diffbot_uri
                        else:
                            url = info.get('website_uri')
                            if url is None:
                                continue
                            resp_enhance, _ = await self.diffbot_request.fetch_enhance('Organization', name, url)
                            self.logger.debug(f'Response from enhance on investor {name}, {url}:\n{pformat(resp_enhance, indent=4)}')
                            if resp_enhance is None or resp_enhance['enhanced'] is None \
                                    or 'diffbotUri' not in resp_enhance['enhanced']:
                                continue
                            investor_uri = resp_enhance['enhanced']['diffbotUri'].split('/')[-1]
                            self.investors_uri_mapping[name] = investor_uri
                    uri = self.investors_uri_mapping[name]
                    self.nodes_queue.put_nowait(uri)
                    self.all_explored_nodes[uri] = {'name': name, 'node_type': info.get('node_type')}
                    self.edges.append({'head': uri,
                                       'relation': LINKS_TO_ID[link_type],
                                       'tail': node})

            elif link_type == 'located_in':
                for uri, info in links.items():

                    self.edges.append({'head': node,
                                       'relation': LINKS_TO_ID[link_type],
                                       'tail': uri})
                    self.all_explored_nodes[uri] = info
            else:
                for uri, info in links.items():
                    self.nodes_queue.put_nowait(uri)
                    self.edges.append({'head': uri,
                                       'relation': LINKS_TO_ID[link_type],
                                       'tail': node})
                    self.all_explored_nodes[uri] = info

    def __process_person_entity(self, data, node):
        all_links = get_links_from_person_entity(data)
        self.logger.debug(f'Links from people node {node}: {pformat(dict(all_links), indent=4)}')
        for link_type, links in all_links.items():
            if link_type in ['board_member_of', 'ceo_of', 'founder_of']:
                for uri, info in links.items():
                    self.nodes_queue.put_nowait(uri)
                    self.edges.append({'head': node,
                                       'relation': LINKS_TO_ID[link_type],
                                       'tail': uri})
                    self.all_explored_nodes[uri] = info

            elif link_type == 'located_in':
                for uri, info in links.items():
                    self.edges.append({'head': node,
                                       'relation': LINKS_TO_ID[link_type],
                                       'tail': uri})
                    self.all_explored_nodes[uri] = info


async def download_enhance_entities(api_key, output_folder,
                                    orgs_info=None, people_info=None,
                                    log_level=logging.INFO,
                                    uri_info_map=None):
    """
    Downloads entities based on enhance api

    Args:
        api_key: diffnot api key
        output_folder: output folder to store downloaded entities
        orgs_info: a list of dict containing url and or names of companies to match with diffbot enhance api
        people_info: a list of dict containing url and or names of people to match with diffbot enhance api
        log_level: logging level
        uri_info_map: a dict of diffbot uri to url

    Returns:
        None
    """
    assert people_info or orgs_info, "at least one of orgs_info or people_info should be not None"
    if orgs_info: assert type(orgs_info) is list, "orgs_info should be a list"
    if people_info: assert type(people_info) is list, "people_info should be list"

    async with aiohttp.ClientSession() as session:
        diffbot_enhance_downloader = DiffBotEnhanceDownloader(api_key, output_folder,
                                                              session)

        diffbot_enhance_downloader.logger.setLevel(log_level)

        diffbot_enhance_downloader.uri_info_mapping = uri_info_map if uri_info_map else {}
        tasks = []
        if uri_info_map is None:
            saved_enhanced_entity_urls = set()
            saved_enhanced_entity_names = set()
        else:
            saved_enhanced_entity_urls = {info['url'] for info in uri_info_map.values() if 'url' in info}
            saved_enhanced_entity_names = {info['name'] for info in uri_info_map.values() if 'name' in info}
        if orgs_info:
            for org_info in orgs_info:
                bool_url_not_saved = org_info['url'] not in saved_enhanced_entity_urls if 'url' in orgs_info else True
                bool_name_not_saved = org_info['name'] not in saved_enhanced_entity_names if 'name' in orgs_info else True
                if bool_url_not_saved and bool_name_not_saved:
                    tasks.append(diffbot_enhance_downloader.save_diffbot_entity(org_info, 'Organization'))
        if people_info:
            for person_info in people_info:
                bool_url_not_saved = person_info['url'] not in saved_enhanced_entity_urls if 'url' in person_info else True
                bool_name_not_saved = person_info['name'] not in saved_enhanced_entity_names if 'name' in person_info else True
                if bool_url_not_saved and bool_name_not_saved:
                    tasks.append(diffbot_enhance_downloader.save_diffbot_entity(person_info, 'Person'))
        await asyncio.gather(*tasks)
    time.sleep(5)
    diffbot_enhance_downloader.save_uri_info_mapping()
    return diffbot_enhance_downloader.uri_info_mapping