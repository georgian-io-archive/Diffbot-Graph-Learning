from collections import defaultdict

from .diffbot_constants import CEO_URI, BOARD_MEMBER_URI, FOUNDER_URI, LINKS_TO_ID


def get_links_from_org_entity(response_dict):
    """ Returns associated links given an organization entity response dictionary

    Args:
        response_dict (`obj`: dict): The dictionary containing the organization's
            information as returned from Diffbot's KG API

    Returns:
        A dictionary of dictionary of dictionary mapping the link type
        founder, ceo, location, investors and board members' to a dictionary mapping the
        entity uris to a dictionary containing their name and entity type. For investors,
        this innermost dictionary maps investor name to a dictionary containing
        entity type and the diffbot_uri if it exists. Otherwise, it is the
        associated website_uri that we can use for the enhance api.

        Small example for IEX Group
        {
            'founder_of': {
                'PILTd1j21OfepcwAjR9hkAQ': {'name': 'Ronan Ryan', 'node_type': 'Person'}
            },
            'ceo_of': {
                'P6shM4PG7NvW_ybCbkbkanw': {'name': 'Brad Katsuyama', 'node_type': 'Person'}
            },
            'investor_of': {
                'Belfer Management': {'website_uri': 'crunchbase.com/organization/belfer-management', 'node_type': 'Organization'},
                'Steve Wynn': {'diffbot_uri': 'P-Njz-5kyPkqbdZJFu3UGig', 'node_type': 'Person'}
            },
            'located_in': {
                'A01d4EK33MmCosgI2KXa4-A': {'name': 'United States of America', 'node_type': 'Location'},
                'AcMmgf99wMQ6XYnbChv5HiQ': {'name': 'New York City', 'node_type': 'Location'},
                'A1NxI_KXaMbiP5g2aM9MRdw': {'name': 'New York', 'node_type': 'Location'}
            }
        }

    """
    founders = response_dict.get('founders', [])
    ceo = response_dict.get('ceo')
    investments = response_dict.get('investments', [])
    board_members = response_dict.get('boardMembers', [])
    location = response_dict.get('location')

    all_links = defaultdict(dict)
    for founder in founders:
        get_uri_and_add_to_dict(founder, all_links, 'founder_of', 'Person')

    if ceo is not None:
        get_uri_and_add_to_dict(ceo, all_links, 'ceo_of', 'Person')

    for investment in investments:
        if 'investors' in investment.keys():
            for investor in investment['investors']:
                diffbot_uri = investor.get('diffbotUri')
                if diffbot_uri is not None:
                    all_links['investor_of'][investor['name']] = {'diffbot_uri': diffbot_uri.split('/')[-1],
                                                                  'node_type': 'Person'}
                website_uris = investor.get('websiteUris', [])
                if len(website_uris) >= 1:
                    all_links['investor_of'][investor['name']] = {'website_uri': investor['websiteUris'][0],
                                                                  'node_type': 'Organization'}

    for board_member in board_members:
        get_uri_and_add_to_dict(board_member, all_links, 'board_member_of', 'Person')

    if location is not None:
        all_links.update(get_links_from_location_dict(location))

    return all_links


def get_links_from_person_entity(response_dict):
    """ Returns associated links given an given Person entity response dictionary.

    Args:
        response_dict (`obj`: dict): The dictionary containing the Person's
            information as returned from Diffbot's KG API:

    Returns:
        A dictionary of dictionary of dictionary mapping the link type
        education, current location, and current employments' to a dictionary mapping the
        entity uris to a dictionary containing their name and entity type.

        Example.
        {
            'educated_at': {
                'Ozfdb2vswMHOFMy003XxkTA': {'name': 'Harvard Business School', 'node_type': 'Institution'},
                'OL-LKCyT7NJWjBwJS1vEDmg': {'name': 'University of Wisconsin-Madison', 'node_type': 'Institution'}
            },
            'board_member_of': {
                {'CDCA8VkXSN0myqrh9tEI-dQ': {'name': 'Kony', 'node_type': 'Organization'}}
            },
            'located_in': {
                'A01d4EK33MmCosgI2KXa4-A': {'name': 'United States of America', 'node_type': 'Location'},
                'A1NxI_KXaMbiP5g2aM9MRdw': {'name': 'New York', 'node_type': 'Location'}
            }
        }
    """
    educations = response_dict.get('educations', [])
    employments = response_dict.get('employments', [])
    locations = response_dict.get('locations', [])

    all_links = defaultdict(dict)
    for education in educations:
        institution = education.get('institution')
        if institution is not None:
            get_uri_and_add_to_dict(institution, all_links, 'educated_at', 'Institution')

    for employment in employments:
        if not employment.get('isCurrent'):
            continue
        employer = employment.get('employer', {})

        for category in employment.get('categories', []):
            if 'diffbotUri' not in category:
                continue
            diffbot_uri = category['diffbotUri'].split('/')[-1]
            if diffbot_uri == FOUNDER_URI:
                get_uri_and_add_to_dict(employer, all_links, 'founder_of', 'Organization')
            elif diffbot_uri == BOARD_MEMBER_URI:
                get_uri_and_add_to_dict(employer, all_links, 'board_member_of', 'Organization')
            elif diffbot_uri == CEO_URI:
                get_uri_and_add_to_dict(employer, all_links, 'ceo_of', 'Organization')

    for location in locations:
        if not location.get('isCurrent'):
            continue
        all_links.update(get_links_from_location_dict(location))

    return all_links


def get_links_from_location_dict(location_dict):
    country = location_dict.get('country')
    city = location_dict.get('city')
    subregion = location_dict.get('subregion')
    region = location_dict.get('region')

    all_links = defaultdict(dict)
    if country is not None:
        get_uri_and_add_to_dict(country, all_links, 'located_in', 'Country')
    if city is not None:
        get_uri_and_add_to_dict(city, all_links, 'located_in', 'City')
    if subregion is not None:
        get_uri_and_add_to_dict(subregion, all_links, 'located_in', 'SubRegion')
    if region is not None:
        get_uri_and_add_to_dict(region, all_links, 'located_in', 'Region')
    return all_links


def parse_org_links_for_edges(node, all_links,
                              investors_uri_mapping=None,
                              saved_entities_uri_info_map=None,
                              all_explored_nodes=None):
    if all_explored_nodes is None:
        all_explored_nodes = {}
    if saved_entities_uri_info_map is None:
        saved_entities_uri_info_map = {}
    if investors_uri_mapping is None:
        investors_uri_mapping = {}
    edges = []
    for link_type, links in all_links.items():
        if link_type == 'investor_of':
            for name, info in links.items():
                if name not in investors_uri_mapping:
                    diffbot_uri = info.get('diffbot_uri')
                    if diffbot_uri:
                        investors_uri_mapping[name] = diffbot_uri
                    else:
                        all_explored_nodes[name] = {'name': name,
                                                    'website_uri': info.get('website_uri'),
                                                    'node_type': 'Organization'}
                        edges.append({'head': name,
                                      'relation': LINKS_TO_ID[link_type],
                                      'tail': node})
                        continue
                uri = investors_uri_mapping[name]
                edges.append({'head': uri,
                              'relation': LINKS_TO_ID[link_type],
                              'tail': node})
                if uri not in all_explored_nodes:
                    if uri in saved_entities_uri_info_map:
                        all_explored_nodes[uri] = saved_entities_uri_info_map[uri]
                    else:
                        all_explored_nodes[uri] = {'name': name, 'node_type': info.get('node_type')}

        elif link_type == 'located_in':
            for uri, info in links.items():

                edges.append({'head': node,
                              'relation': LINKS_TO_ID[link_type],
                              'tail': uri})
                if uri not in all_explored_nodes:
                    all_explored_nodes[uri] = info

        else:
            for uri, info in links.items():
                edges.append({'head': uri,
                              'relation': LINKS_TO_ID[link_type],
                              'tail': node})
                if uri not in all_explored_nodes:
                    if uri in saved_entities_uri_info_map:
                        all_explored_nodes[uri] = saved_entities_uri_info_map[uri]
                    else:
                        all_explored_nodes[uri] = info
    return edges, all_explored_nodes


def parse_person_links_for_edges(node, all_links,
                                 saved_entities_uri_info_map=None,
                                 all_explored_nodes=None):
    if all_explored_nodes is None:
        all_explored_nodes = {}
    if saved_entities_uri_info_map is None:
        saved_entities_uri_info_map = {}
    edges = []
    for link_type, links in all_links.items():
        if link_type in ['board_member_of', 'ceo_of', 'founder_of']:
            for uri, info in links.items():
                edges.append({'head': node,
                              'relation': LINKS_TO_ID[link_type],
                              'tail': uri})
                if uri not in all_explored_nodes:
                    if uri in saved_entities_uri_info_map:
                        all_explored_nodes[uri] = saved_entities_uri_info_map[uri]
                    else:
                        all_explored_nodes[uri] = info
        else:
            for uri, info in links.items():
                edges.append({'head': node,
                              'relation': LINKS_TO_ID[link_type],
                              'tail': uri})
                if uri not in all_explored_nodes:
                    if uri in saved_entities_uri_info_map:
                        all_explored_nodes[uri] = saved_entities_uri_info_map[uri]
                    else:
                        all_explored_nodes[uri] = info
    return edges, all_explored_nodes


def get_uri_and_add_to_dict(info, links_dict, link_type, node_type):
    if 'diffbotUri' in info:
        uri = info['diffbotUri'].split('/')[-1]
        links_dict[link_type][uri] = {'name': info.get('name'), 'node_type': node_type}