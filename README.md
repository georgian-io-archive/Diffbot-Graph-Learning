# Diffbot Graph Learning
Use extracted company and people entities from Diffbot to build a heterogenous subgraph of Diffbot's Knowledge Graph. 

![](demo/imgs/example_w_subregion.svg)

## Diffbot Entity Extraction
For querying the diffbot api and downloading the entity respoinses this repo includes two options:
1. BFS given initial diffbot uri starting entities. This uses Diffbot's 
[knowledge graph API](https://docs.diffbot.com/kgapi).
2. [Diffbot enhance API](https://docs.diffbot.com/enhance) which matches the entity in
Diffbot's knowledge graph given a name and/or url.

The implementation relies on Python's asyncio for quickly sending requests and saving responses.  

To see how to use the BFS extraction and enhance api scripts use the `-h` option
```shell script
$ python main_extract_entities_bfs.py -h
$ python main_extract_entities_enhance_api.py -h
```
A demo example for each of these extraction methods can be ran as follows.
An api key in each of the yaml files needs to be specified before running.
```shell script
$ python main_extract_entities_bfs.py --config_file ./demo/extract_entities_bfs.yaml
$ python main_extract_entities_enhance_api.py --config_file ./demo/extract_entities_enhance.yaml 
```
The keys and values in the yaml config files can also be passed directly as command line arguments.

## Building Graph from Diffbot Downloaded Entities
Once we download the entities from diffbot we can build a graph from
the saved jsons. We can save this as a gexf file.
Building on the bfs diffbot entity extraction demo we can
```shell script
$ python main_build_gexf_graph.py --config_file ./demo/build_gexf_graph.yaml
```
We can change the `node_filter` method to build different graphs.

## Running Heterogenous Graph Representation Models
This repo includes the [Deep Graph Library](https://github.com/dmlc/dgl) implementation of
the following two models
1. Heterogenous Relational Graph Convolutional Network (HRGCN). This model
builds on RGCN from Schlichtkrull *et al.*: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (ESWC 2018) 
to handle heterogenous node types.
2. Heterogenous Graph Attention Network (HAN) from Wang *et al.*: [Heterogeneous Graph Attention Network
](https://arxiv.org/abs/1903.07293) (WWW 2019)

There is also a MLP module on node features to run sanity checks on.

### Custom Diffbot Dataset
For your own dataset created from the bfs extraction and build graph scripts, 
make a folder with the name of the dataset in `data/raw`. Inside the folder
place the `gexf` file built from `main_build_gexf_graph.py` and name it
`graph.gexf`.

### Included Example
The repo includes a demo graph dataset in the `data/raw/top_100_VCs_BFS_20000_LCC`. This is a graph
built using `main_extract_entities_bfs.py` and `main_build_gexf_graph.py` with
the [top 100 venture capital investors and firms](https://www.cbinsights.com/research/top-venture-capital-partners/) in 2019 as the starting seed nodes for
Breadth First Search and taking the lagest connected component.

![Alt text](demo/imgs/top_100_VCs_BFS_20000_LCC.png)

From this graph we can for example, do node classification on the 
Diffbot [categories](https://docs.diffbot.com/ontology) of each organization.

Some examples of categories are  `Software Companies`, `Financial Services Companies`, and `Software As A Service Companies`.

### Running Experiment
Run `python main.py -h` to see the hyperparameter and experiment configurations.
To quickly run an example
```shell script
$ python main.py ./demo/example_train_config_hrgcn.json
```
There are also train config examples for `HAN` and `MLP`.

Some results for the demo graph are as follows for 5 fold cross validation.


Diffbot Category | Num Positive | Num Negative
----------------- | ---------------- | ----------------- 
Software Companies | 5140 | 6124
Financial Services Companies | 2014   | 9250 |
Software As A Service Companies | 685 |10579


Model | Diffbot Category |F1 | ROC AUC | PR AUC
--------|-------------|---------|------- | -------
HAN | Software Companies | 0.683 | 0.752 | 0.710
HRGCN | Software Companies | **0.695** |  **0.762** | **0.714**
MLP | Software Companies | 0.644 | 0.623| 0.546
**** |
HAN | Financial Services Companies | 0.471 | 0.741 | 0.461
HRGCN | Financial Services Companies | **0.487** |  **0.745** | **0.462**
MLP | Financial Services Companies | 0.332 | 0.591 | 0.227
**** |
HAN | Software As A Service Companies | 0.181 | 0.668 | 0.098
HRGCN | Software As A Service Companies | **0.234** | **0.696** | **0.148**
MLP | Software As A Service Companies | 0.163 | 0.628 | 0.099