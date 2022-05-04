---
title: "Graph"
permalink: /docs/graph/
excerpt: "About graph."
last_modified_at: 2022-04-13T10:22:56-04:00
toc: true
layout: tuto
---

## Background
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/320363/1648440825677-28730162-d69a-4f25-85c1-f936b67c37f0.png#clientId=u74d07bf1-4c20-4&crop=0&crop=0&crop=1&crop=1&from=paste&id=u4035381f&name=image.png&originHeight=2310&originWidth=3760&originalType=url&ratio=1&rotation=0&showTitle=false&size=520449&status=done&style=none&taskId=ub49659ec-839f-469c-a4ac-e0eddf274de&title=)<br />For privacy reasons, there are many graphs in scenarios that are split into different subgraphs in different clients, which leads to missing of the cross-client edges and data non.i.i.d., etc.

Not only in areas such as CV and NLP, but FederatedScope also provides support for graph learning researchers with a rich collection of datasets, the latest federated graph algorithms and benchmarks.

In this tutorial, you will learn:

- How to start graph learning with FederatedScope [[click]](#start)
- How to reproduce the main experimental results in EasyFGL paper [[click]](#reproduce)
- How to use build-in or create a new federated graph dataset [[click]](#dataset)
- How to run with built-in or new models [[click]](#model)
- How to develop new federated graph algorithms [[click]](fedgnn)
- How to enable FedOptimizer, PersonalizedFL and FedHPO [[click]](#fedalgo)
- Benchmarkcketing Federated GNN [[click]](#benchmark)

## <span id="start">Quick start</span>

Let's start with a two-layer GCN on (fed) Cora to familiarize you with FederatedScope.

### Start with built-in functions

You can easily run through a `yaml` file:

```python
# Whether to use GPU
use_gpu: True

# Deciding which GPU to use
device: 0

# Federate learning related options
federate:
  # `standalone` or `distributed`
  mode: standalone
  # Evaluate in Server or Client test set
  make_global_eval: True
  # Number of dataset being split
  client_num: 5
  # Number of communication round
  total_round_num: 400
  # Number of local update steps
  local_update_steps: 4

# Dataset related options
data:
  # Root directory where the data stored
  root: data/
  # Dataset name
  type: cora
  # Use Louvain algorithm to split `Cora`
  splitter: 'louvain'
  # Use fullbatch training, batch_size should be `1`
  batch_size: 1

# Model related options
model:
  # Model type
  type: gcn
  # Hidden dim
  hidden: 64
  # Dropout rate
  dropout: 0.5
  # Number of Class of `Cora`
  out_channels: 7
    
# Optimizer related options
optimizer:
  # Learning rate
  lr: 0.25
  # Weight decay
  weight_decay: 0.0005
  # Optimizer type
  type: SGD
    
# Criterion related options
criterion:
  # Criterion type
  type: CrossEntropyLoss
    
# Trainer related options
trainer:
  # Trainer type
  type: nodefullbatch_trainer
    
# Evaluation related options
eval:
  # Frequency of evaluation
  freq: 1
  # Evaluation metrics, accuracy and number of correct items
  metrics: ['acc', 'correct']
```

If the `yaml` file is named as `example.yaml`, just run:

```python
python federatedscope/main.py --cfg example.yaml
```

Then, the FedAVG performance is around `0.87`.

### Start with customized functions

FederatedScope also provides `register` function to set up the FL procedure. Here we only provide an example about two-layer GCN on (fed) Cora, please refer to Start with your own case for details.

-  Load Cora dataset and split into 5 subgraph 

```python
# federatedscope/contrib/data/my_cora.py

import torch
import copy
import numpy as np

from torch_geometric.datasets import Planetoid
from federatedscope.gfl.dataset.splitter import LouvainSplitter
from federatedscope.register import register_data


def my_cora(config=None):
  	path = config.data.root
    
    num_split = [232, 542, np.inf]
  	dataset = Planetoid(path,
                        'cora',
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])
    global_data = copy.deepcopy(dataset)[0]
    dataset = LouvainSplitter(config.federate.client_num)(dataset[0])
    
    data_local_dict = dict()
    for client_idx in range(len(dataset)):
        data_local_dict[client_idx + 1] = dataset[client_idx]
        
    data_local_dict[0] = global_data
    return data_local_dict, config

def call_my_data(config):
    if config.data.type == "mycora":
        data, modified_config = MyData(config)
        return data, modified_config

register_data("mycora", call_my_data)
```


-  Build a two-layer GCN

```python
# federatedscope/contrib/model/my_gcn.py

import torch
import torch.nn.functional as F

from torch.nn import Parameter, Linear, ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from federatedscope.register import register_model


class MyGCN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0):
        super(GCN_Net, self).__init__()
        self.convs = ModuleList()
        for i in range(max_depth):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden))
            elif (i + 1) == max_depth:
                self.convs.append(GCNConv(hidden, out_channels))
            else:
                self.convs.append(GCNConv(hidden, hidden))
        self.dropout = dropout

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if (i + 1) == len(self.convs):
                break
            x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x

def GCNBuilder(model_config, data):
    model = MyGCN(data.x.shape[-1],
                  model_config.out_channels,
                  hidden=model_config.hidden,
                  max_depth=model_config.gnn_layer,
                  dropout=model_config.dropout)
    return model

def call_my_net(model_config, local_data):
    if model_config.type == "mygcn":
        model = GCNBuilder(model_config, local_data)
        return model

register_model("mygcn", call_my_net)
```


-  Run with following command to start:

```bash
python federatedscope/main.py --cfg example.yaml data.type mycora model.type mygcn
```


## <span id="reproduce">Reproduce the main experimental results </span>

We also provide configuration files to help you easily reproduce the results in our `EasyFGL` paper. All the `yaml` files are in `federatedscope/gfl/baseline`.

-  Train two-layer GCN with Node-level task dataset Cora

```bash
python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml
```
<br />Then, the FedAVG performance is around `0.87`. 

-  Train two-layer GCN with Link-level task dataset WN18

```bash
python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gcn_minibatch_on_kg.yaml
```
<br />Then, the FedAVG performance is around `hits@1: 0.30`, `hits@5: 0.79`, `hits@10: 0.96`. 

-  Train two-layer GCN with Graph-level task dataset HIV

```bash
python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gcn_minibatch_on_hiv.yaml
```
<br />Then, the FedAVG performance is around `accuracy: 0.96` and `roc_aucs: 0.62`. 

<a name="4fb9498c"></a>
## <span id="dataset">DataZoo</span>

FederatedScope provides a rich collection of datasets for graph learning researchers, including real federation datasets as well as simulated federation datasets split by some sampling or clustering algorithms. The dataset statistics are shown in the table and **more datasets are coming soon**:

<table>
<thead>
  <tr>
    <th>Task</th>
    <th>Domain</th>
    <th>Dataset</th>
    <th>Splitter</th>
    <th># Graph</th>
    <th>Avg. # Nodes</th>
    <th>Avg. # Edges</th>
    <th># Class</th>
    <th>Evaluation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="4">Node-level</td>
    <td>Citation network</td>
    <td>Cora [1]</td>
    <td>random&amp;community</td>
    <td>1</td>
    <td>2,708</td>
    <td>5,429</td>
    <td>7</td>
    <td>ACC</td>
  </tr>
  <tr>
    <td>Citation network</td>
    <td>CiteSeer [2]</td>
    <td>random&amp;community</td>
    <td>1</td>
    <td>4,230</td>
    <td>5,358</td>
    <td>6</td>
    <td>ACC</td>
  </tr>
  <tr>
    <td>Citation network</td>
    <td>PubMed [3]</td>
    <td>random&amp;community</td>
    <td>1</td>
    <td>19,717</td>
    <td>44,338</td>
    <td>5</td>
    <td>ACC</td>
  </tr>
  <tr>
    <td>Citation network</td>
    <td>FedDBLP [4]</td>
    <td>meta</td>
    <td>1</td>
    <td>52,202</td>
    <td>271,054</td>
    <td>4</td>
    <td>ACC</td>
  </tr>
  <tr>
    <td rowspan="4">Link-level</td>
    <td>Recommendation System</td>
    <td>Ciao [5]</td>
    <td>meta</td>
    <td>28</td>
    <td>5,875.68</td>
    <td>20,189.29</td>
    <td>6</td>
    <td>ACC</td>
  </tr>
  <tr>
    <td>Recommendation System</td>
    <td>Taobao</td>
    <td>meta</td>
    <td>3</td>
    <td>443,365</td>
    <td>2,015,558</td>
    <td>2</td>
    <td>ACC</td>
  </tr>
  <tr>
    <td>Knowledge Graph</td>
    <td>WN18 [6]</td>
    <td>label_space</td>
    <td>1</td>
    <td>40,943</td>
    <td>151,442</td>
    <td>18</td>
    <td>Hits@n</td>
  </tr>
  <tr>
    <td>Knowledge Graph</td>
    <td>FB15k-237 [6]</td>
    <td>label_space</td>
    <td>1</td>
    <td>14,541</td>
    <td>310,116</td>
    <td>237</td>
    <td>Hits@n</td>
  </tr>
  <tr>
    <td rowspan="4">Graph-level</td>
    <td>Molecule</td>
    <td>HIV [7]</td>
    <td>instance_space</td>
    <td>41,127</td>
    <td>25.51</td>
    <td>54.93</td>
    <td>2</td>
    <td>ROC-AUC</td>
  </tr>
  <tr>
    <td>Proteins</td>
    <td>Proteins [8]</td>
    <td>instance_space</td>
    <td>1,113</td>
    <td>39.05</td>
    <td>145.63</td>
    <td>2</td>
    <td>ACC</td>
  </tr>
  <tr>
    <td>Social network</td>
    <td>IMDB [8]</td>
    <td>label_space</td>
    <td>1,000</td>
    <td>19.77</td>
    <td>193.06</td>
    <td>2</td>
    <td>ACC</td>
  </tr>
  <tr>
    <td>Multi-task</td>
    <td>Mol [8]</td>
    <td>multi_task</td>
    <td>18,661</td>
    <td>55.62</td>
    <td>1,466.83</td>
    <td>-</td>
    <td>ACC</td>
  </tr>
</tbody>
</table>

<a name="b16fb017"></a>

### Dataset format

Let's start `Dataset` with `torch_geometric.data`. Our `DataZoo` contains three levels of tasks which are node-level, link-level and graph-level. Different levels of data have different attributes:

-  Node-level dataset<br />Node-level dataset contains one `torch_geometric.data` with attributes: `x` represents the node attribute, `y` represents the node label, `edge_index` represents the edges of the graph, `edge_attr`  represents the edge attribute which is optional, and `train_mask`, `val_mask`,`test_mask` are the node mask of each splits.

```python
# Cora
Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
```


-  Link-level dataset<br />Link-level dataset contains one `torch_geometric.data` with attributes: `x` represents the node attribute, `edge_index` represents the edges of the graph, `edge_type` represents the link label, `edge_attr`  represents the edge attribute which is optional, `train_edge_mask`, `valid_edge_mask`,`test_edge_mask` are the link mask of each splits, and `input_edge_index` is optional if the input is `edge_index.T[train_edge_mask].T`.

```python
# WN18
Data(x=[40943, 1], edge_index=[2, 151442], edge_type=[151442], num_nodes=40943, train_edge_mask=[151442], valid_edge_mask=[151442], test_edge_mask=[151442], input_edge_index=[2, 282884])
```


-  Graph-level dataset<br />Graph-level dataset contains several `torch_geometric.data`, and the task is to predict the label of each graph.

```python
# HIV[0]
Data(x=[19, 9], edge_index=[2, 40], edge_attr=[40, 3], y=[1, 1], smiles='CCC1=[O+][Cu-3]2([O+]=C(CC)C1)[O+]=C(CC)CC(CC)=[O+]2')
...
# HIV[41126]
Data(x=[37, 9], edge_index=[2, 80], edge_attr=[80, 3], y=[1, 1], smiles='CCCCCC=C(c1cc(Cl)c(OC)c(-c2nc(C)no2)c1)c1cc(Cl)c(OC)c(-c2nc(C)no2)c1')
```


### Dataloader

For node-level and link-level tasks, we use full-batch training by default. However, some large graphs can not be adopted to full-batch training due to the video memory limitation. Fortunately, we also provide some graph sampling algorithms, like `GraphSAGE` and `GraphSAINT` which are subclasses of `torch_geometric.loader`.

-  In node-level task, you should set:

```python
cfg.data.loader = 'graphsaint-rw' # or `neighbor`
cfg.model.type = 'sage'
cfg.trainer.type = 'nodeminibatch_trainer'
```


-  In link-level task, you should set:

```python
cfg.data.loader = 'graphsaint-rw'
cfg.model.type = 'sage'
cfg.trainer.type = 'linkminibatch_trainer'
```


### Splitter

Existing graph datasets are a valuable source to meet the need for more FL datasets. Under the federated learning setting, the dataset is decentralized. To simulate federated graph datasets by existing standalone ones, our `DataZoo` integrates a rich collection of `federatedscope.gfl.dataset.splitter`.  Except for `meta_splitter` which comes from the meta information of datasets, we have the following splitters:

-  Node-level task 
   -  `community_splitter`: **Split by cluster** `cfg.data.splitter = 'louvain'`<br />Community detection algorithms such as Louvain are at first applied to partition a graph into several clusters. Then these clusters are assigned to the clients, optionally with the objective of balancing the number of nodes in each client. 
   -  `random_splitter`: **Split by random** `cfg.data.splitter = 'random'`<br />The node set of the original graph is randomly split into 𝑁 subsets with or without intersections. Then, the subgraph of each client is deduced from the nodes assigned to that client. Optionally, a specified fraction of edges is randomly selected to be removed. 
-  Link-level task 
   -  `label_space_splitter`: **Split by latent dirichlet allocation** `cfg.data.splitter = 'rel_type'`<br />It is designed to provide label distribution skew via latent dirichlet allocation (LDA). 
-  Graph-level task 
   -  `instance_space_splitter`: **Split by index **`cfg.data.splitter = 'scaffold' or 'rand_chunk'`<br />It is responsible for creating feature distribution skew (i.e., covariate shift). To realize this, we sort the graphs based on their values of a certain aspect. 
   -  `multi_task_splitter`: **Split by dataset **`cfg.data.splitter = 'louvain'`<br />Different clients have different tasks. 


## <span id="model">ModelZoo</span>


### GNN

We implemented GCN [9], GraphSAGE [10], GAT [11], GIN [12], and GPR-GNN [13] on different levels of tasks in `federatedscope.gfl.model`, respectively. In order to run your FL procedure with these models, set `cfg.model.task` to `node`, `link` or `graph`, and all models can be instantiated automatically based on the data provided. More GNN models are coming soon!


### Trainer

We provide several `Trainers` for different models and for different tasks.

- `NodeFullBatchTrainer` 
   - For node-level tasks.
   - For full batch training.
- `NodeMiniBatchTrainer` 
   - For node-level tasks.
   - For GraphSAGE, GraphSAINT and other graph sampling methods.
- `LinkFullBatchTrainer` 
   - For link-level tasks.
   - For full batch training.
- `LinkMiniBatchTrainer` 
   - For link-level tasks.
   - For GraphSAGE, GraphSAINT and other graph sampling methods.
- `GraphMiniBatchTrainer` 
   - For graph-level tasks.

## <span id="fedgnn">Develop federated GNN algorithms</span>

FederatedScope provides comprehensive support to help you develop federated GNN algorithms. Here we will go through `FedSage+` [14] and `GCFL+` [15] as examples.

-  FedSage+, [_Subgraph Federated Learning with Missing Neighbor Generation_](https://arxiv.org/pdf/2106.13430v6.pdf)_, in NeurIPS_ 2021<br />FedSage+ try to "restore" the missing graph structure by jointly training a `Missing Neighbor Generator`, each client sends `Missing Neighbor Generator` to other clients, and the other clients optimize it with their own local data and send the model gradient back in order to achieve joint training without privacy leakage.<br />We implemented FedSage+ in `federatedscope/gfl/fedsageplus` with `FedSagePlusServer` and `FedSagePlusClient`. In FederatedScope, we need to define new message types and the corresponding handler functions.

```python
# FedSagePlusServer
self.register_handlers('clf_para', self.callback_funcs_model_para)
self.register_handlers('gen_para', self.callback_funcs_model_para)
self.register_handlers('gradient', self.callback_funcs_gradient)
```
 <br />Because FedSage+ has multiple stages, please carefully deal with the `msg_buffer` in `check_and_move_on()` in different states. 

-  GCFL+, [_Federated Graph Classification over Non-IID Graphs_](https://arxiv.org/pdf/2106.13423v5.pdf)_, NeurIPS_ 2021<br />GCFL+ clusters clients according to the sequence of the gradients of each local model, and those with a similar sequence of the gradients share the same model parameters.<br />We implemented GCFL+ in `federatedscope/gfl/gcflplus` with `FedSagePlusServer` and `FedSagePlusClient`. Since no more messages are involved, we can implement GCFL+ by simply defining how to clustering clients and adding gradients to message `model_para`. 

## <span id="fedalgo">Enable build-in Federated Algorithms</span>

FederatedScope provides many built-in FedOptimize, PersonalizedFL and FedHPO algorithms. You can adapt them to graph learning by simply turning on the switch.

For more details, see:

- FedOptimize
- PersonalizedFL
- FedHPO

## <span id="benchmark">Benchmarks</span>

We've conducted extensive experiments to build the benchmarks of FedGraph, which simultaneously gains<br />many valuable insights for the community.

### Node-level task

- Results on representative node classification datasets with `random_splitter` Mean accuracy (%) ± standard deviation.  

  <table>
  <thead>
    <tr>
      <th></th>
      <th colspan="5">Cora</th>
      <th colspan="5">CiteSeer</th>
      <th colspan="5">PubMed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td>Local</td>
      <td>FedAVG</td>
      <td>FedOpt</td>
      <td>FedPeox</td>
      <td>Global</td>
      <td>Local</td>
      <td>FedAVG</td>
      <td>FedOpt</td>
      <td>FedPeox</td>
      <td>Global</td>
      <td>Local</td>
      <td>FedAVG</td>
      <td>FedOpt</td>
      <td>FedPeox</td>
      <td>Global</td>
    </tr>
    <tr>
      <td>GCN</td>
      <td>80.95±1.49</td>
      <td>86.63±1.35</td>
      <td>86.11±1.29</td>
      <td>86.60±1.59</td>
      <td>86.89±1.82</td>
      <td>74.29±1.35</td>
      <td>76.48±1.52</td>
      <td>77.43±0.90</td>
      <td>77.29±1.20</td>
      <td>77.42±1.15</td>
      <td>85.25±0.73</td>
      <td>85.29±0.95</td>
      <td>84.39±1.53</td>
      <td>85.21±1.17</td>
      <td>85.38±0.33</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>75.12±1.54</td>
      <td>85.42±1.80</td>
      <td>84.73±1.58</td>
      <td>84.83±1.66</td>
      <td>86.86±2.15</td>
      <td>73.30±1.30</td>
      <td>76.86±1.38</td>
      <td>75.99±1.96</td>
      <td>78.05±0.81</td>
      <td>77.48±1.27</td>
      <td>84.58±0.41</td>
      <td>86.45±0.43</td>
      <td>85.67±0.45</td>
      <td>86.51±0.37</td>
      <td>86.23±0.58</td>
    </tr>
    <tr>
      <td>GAT</td>
      <td>78.86±2.25</td>
      <td>85.35±2.29</td>
      <td>84.40±2.70</td>
      <td>84.50±2.74</td>
      <td>85.78±2.43</td>
      <td>73.85±1.00</td>
      <td>76.37±1.11</td>
      <td>76.96±1.75</td>
      <td>77.15±1.54</td>
      <td>76.91±1.02</td>
      <td>83.81±0.69</td>
      <td>84.66±0.74</td>
      <td>83.78±1.11</td>
      <td>83.79±0.87</td>
      <td>84.89±0.34</td>
    </tr>
    <tr>
      <td>GPR-GNN</td>
      <td>84.90±1.13</td>
      <td>89.00±0.66</td>
      <td>87.62±1.20</td>
      <td>88.44±0.75</td>
      <td>88.54±1.58</td>
      <td>74.81±1.43</td>
      <td>79.67±1.41</td>
      <td>77.99±1.25</td>
      <td>79.35±1.11</td>
      <td>79.67±1.42</td>
      <td>86.85±0.39</td>
      <td>85.88±1.24</td>
      <td>84.57±0.68</td>
      <td>86.92±1.25</td>
      <td>85.15±0.76</td>
    </tr>
    <tr>
      <td>FedSage+</td>
      <td>-</td>
      <td>85.07±1.23</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>78.04±0.91</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>88.19±0.32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
- Results on representative node classification datasets with `community_splitter`: Mean accuracy (%) ± standard deviation.  

  <table>
  <thead>
    <tr>
      <th></th>
      <th colspan="5">Cora</th>
      <th colspan="5">CiteSeer</th>
      <th colspan="5">PubMed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td>Local</td>
      <td>FedAVG</td>
      <td>FedOpt</td>
      <td>FedProx</td>
      <td>Global</td>
      <td>Local</td>
      <td>FedAVG</td>
      <td>FedOpt</td>
      <td>FedProx</td>
      <td>Global</td>
      <td>Local</td>
      <td>FedAVG</td>
      <td>FedOpt</td>
      <td>FedProx</td>
      <td>Global</td>
    </tr>
    <tr>
      <td>GCN</td>
      <td>65.08±2.39</td>
      <td>87.32±1.49</td>
      <td>87.29±1.65</td>
      <td>87±16±1.51</td>
      <td>86.89±1.82</td>
      <td>67.53±1.87</td>
      <td>77.56±1.45</td>
      <td>77.80±0.99</td>
      <td>77.62±1.42</td>
      <td>77.42±1.15</td>
      <td>77.01±3.37</td>
      <td>85.24±0.69</td>
      <td>84.11±0.87</td>
      <td>85.14, 0.88</td>
      <td>85.38±0.33</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>61.29±3.05</td>
      <td>87.19±1.28</td>
      <td>87.13±1.47</td>
      <td>87.09±1.46</td>
      <td>86.86±2.15</td>
      <td>66.17±1.50</td>
      <td>77.80±1.03</td>
      <td>78.54±1.05</td>
      <td>77.70±1.09</td>
      <td>77.48±1.27</td>
      <td>78.35±2.15</td>
      <td>86.87±0.53</td>
      <td>85.72±0.58</td>
      <td>86.65±0.60</td>
      <td>86.23±0.58</td>
    </tr>
    <tr>
      <td>GAT</td>
      <td>61.53±2.81</td>
      <td>86.08±2.52</td>
      <td>85.65±2.36</td>
      <td>85.68±2.68</td>
      <td>85.78±2.43</td>
      <td>66.17±1.31</td>
      <td>77.21±0.97</td>
      <td>77.34±1.33</td>
      <td>77.26±1.02</td>
      <td>76.91±1.02</td>
      <td>75.97±3.32</td>
      <td>84.38±0.82</td>
      <td>83.34±0.87</td>
      <td>84.34±0.63</td>
      <td>84.89±0.34</td>
    </tr>
    <tr>
      <td>GPR-GNN</td>
      <td>69.32±2.07</td>
      <td>88.93±1.64</td>
      <td>88.37±2.12</td>
      <td>88.80±1.29</td>
      <td>88.54±1.58</td>
      <td>71.30±1.65</td>
      <td>80.27±1.28</td>
      <td>78.32±1.45</td>
      <td>79.73±1.52</td>
      <td>79.67±1.42</td>
      <td>78.52±3.61</td>
      <td>85.06±0.82</td>
      <td>84.30±1.57</td>
      <td>86.77±1.16</td>
      <td>85.15±0.76</td>
    </tr>
    <tr>
      <td>FedSage+</td>
      <td>-</td>
      <td>87.68±1.55</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>77.98±1.23</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>87.94, 0.27</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>


### Link-level task

- Results on representative link prediction datasets with `label_space_splitter`: Hits@$n$.  

  <table>
  <thead>
    <tr>
      <th></th>
      <th colspan="15">WN18</th>
      <th colspan="15">FB15k-237</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td colspan="3">Local</td>
      <td colspan="3">FedAVG</td>
      <td colspan="3">FedOpt</td>
      <td colspan="3">FedProx</td>
      <td colspan="3">Global</td>
      <td colspan="3">Local</td>
      <td colspan="3">FedAVG</td>
      <td colspan="3">FedOpt</td>
      <td colspan="3">FedProx</td>
      <td colspan="3">Global</td>
    </tr>
    <tr>
      <td></td>
      <td>1</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
      <td>5</td>
      <td>10</td>
      <td>1</td>
      <td>5</td>
      <td>10</td>
    </tr>
    <tr>
      <td>GCN</td>
      <td>20.70</td>
      <td>55.34</td>
      <td>73.85</td>
      <td>30.00</td>
      <td>79.72</td>
      <td>96.67</td>
      <td>22.13</td>
      <td>78.96</td>
      <td>94.07</td>
      <td>27.32</td>
      <td>83.01</td>
      <td>96.38</td>
      <td>29.67</td>
      <td>86.73</td>
      <td>97.05</td>
      <td>6.07</td>
      <td>20.29</td>
      <td>30.35</td>
      <td>9.86</td>
      <td>34.27</td>
      <td>48.02</td>
      <td>4.12</td>
      <td>18.07</td>
      <td>31.79</td>
      <td>4.66</td>
      <td>28.74</td>
      <td>41.67</td>
      <td>7.80</td>
      <td>32.46</td>
      <td>44.64</td>
    </tr>
    <tr>
      <td>GraphSAGE</td>
      <td>21.06</td>
      <td>54.12</td>
      <td>79.88</td>
      <td>23.14</td>
      <td>78.85</td>
      <td>93.70</td>
      <td>22.82</td>
      <td>79.86</td>
      <td>93.12</td>
      <td>23.14</td>
      <td>78.52</td>
      <td>93.67</td>
      <td>24.24</td>
      <td>79.86</td>
      <td>93.84</td>
      <td>3.95</td>
      <td>14.64</td>
      <td>24.47</td>
      <td>7.13</td>
      <td>23.38</td>
      <td>36.60</td>
      <td>2.20</td>
      <td>19.21</td>
      <td>27.64</td>
      <td>5.85</td>
      <td>24.05</td>
      <td>36.33</td>
      <td>6.19</td>
      <td>23.57</td>
      <td>35.98</td>
    </tr>
    <tr>
      <td>GAT</td>
      <td>20.89</td>
      <td>49.42</td>
      <td>72.48</td>
      <td>23.14</td>
      <td>77.62</td>
      <td>93.49</td>
      <td>23.14</td>
      <td>74.64</td>
      <td>93.52</td>
      <td>23.53</td>
      <td>78.40</td>
      <td>93.00</td>
      <td>24.24</td>
      <td>80.18</td>
      <td>93.76</td>
      <td>3.44</td>
      <td>15.02</td>
      <td>25.14</td>
      <td>6.06</td>
      <td>25.76</td>
      <td>39.04</td>
      <td>2.71</td>
      <td>18.89</td>
      <td>32.76</td>
      <td>6.19</td>
      <td>25.09</td>
      <td>38.00</td>
      <td>6.94</td>
      <td>24.43</td>
      <td>37.87</td>
    </tr>
    <tr>
      <td>GPR-GNN</td>
      <td>22.86</td>
      <td>60.45</td>
      <td>80.73</td>
      <td>26.67</td>
      <td>82.35</td>
      <td>96.18</td>
      <td>24.46</td>
      <td>73.33</td>
      <td>87.18</td>
      <td>27.62</td>
      <td>81.87</td>
      <td>95.68</td>
      <td>29.19</td>
      <td>82.34</td>
      <td>96.24</td>
      <td>4.45</td>
      <td>13.26</td>
      <td>21.24</td>
      <td>9.62</td>
      <td>32.76</td>
      <td>45.97</td>
      <td>2.01</td>
      <td>9.81</td>
      <td>16.65</td>
      <td>3.72</td>
      <td>15.62</td>
      <td>27.79</td>
      <td>10.62</td>
      <td>33.87</td>
      <td>47.45</td>
    </tr>
  </tbody>
  </table>


### Graph-level task

- Results on representative graph classification datasets: Mean accuracy (%) ± standard deviation.  

  <table>
  <thead>
    <tr>
      <th></th>
      <th colspan="5">PROTEINS</th>
      <th colspan="5">IMDB</th>
      <th colspan="5">Multi-task</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td>
      <td>Local</td>
      <td>FedAVG</td>
      <td>FedOpt</td>
      <td>FedProx</td>
      <td>Global</td>
      <td>Local</td>
      <td>FedAVG</td>
      <td>FedOpt</td>
      <td>FedProx</td>
      <td>Global</td>
      <td>Local</td>
      <td>FedAVG</td>
      <td>FedOpt</td>
      <td>FedProx</td>
      <td>Global</td>
    </tr>
    <tr>
      <td>GCN</td>
      <td>71.10±4.65</td>
      <td>73.54±4.48</td>
      <td>71.24±4.17</td>
      <td>73.36±4.49</td>
      <td>71.77±3.62</td>
      <td>50.76±1.14</td>
      <td>53.24±6.04</td>
      <td>50.49±8.32</td>
      <td>48.72±6.73</td>
      <td>53.24±6.04</td>
      <td>66.37±1.78</td>
      <td>65.99±1.18</td>
      <td>69.10±1.58</td>
      <td>68.59±1.99</td>
      <td>-</td>
    </tr>
    <tr>
      <td>GIN</td>
      <td>69.06±3.47</td>
      <td>73.74±5.71</td>
      <td>60.14±1.22</td>
      <td>73.18±5.66</td>
      <td>72.47±5.53</td>
      <td>55.82±7.56</td>
      <td>64.79±10.55</td>
      <td>51.87±6.82</td>
      <td>70.65±8.35</td>
      <td>72.61±2.44</td>
      <td>75.05±1.81</td>
      <td>63.40±2.22</td>
      <td>63.33±1.18</td>
      <td>63.01±0.44</td>
      <td>-</td>
    </tr>
    <tr>
      <td>GAT</td>
      <td>70.75±3.33</td>
      <td>71.95±4.45</td>
      <td>71.07±3.45</td>
      <td>72.13±4.68</td>
      <td>72.48±4.32</td>
      <td>53.12±5.81</td>
      <td>53.24±6.04</td>
      <td>47.94±6.53</td>
      <td>53.82±5.69</td>
      <td>53.24±6.04</td>
      <td>67.72±3.48</td>
      <td>66.75±2.97</td>
      <td>69.58±1.21</td>
      <td>69.65±1.14</td>
      <td>-</td>
    </tr>
    <tr>
      <td>GCFL+</td>
      <td>-</td>
      <td>73.00±5.72</td>
      <td>-</td>
      <td>74.24±3.96</td>
      <td>-</td>
      <td>-</td>
      <td>69.47±8.71</td>
      <td>-</td>
      <td>68.90±6.30</td>
      <td>-</td>
      <td>-</td>
      <td>65.14±1.23</td>
      <td>-</td>
      <td>65.69±1.55</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
- Results with **PersonalizedFL** on representative graph classification datasets: Mean accuracy (%) ± standard deviation. (GIN)  

  <table>
  <thead>
    <tr>
      <th></th>
      <th>Multi-task</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FedBN [16]</td>
      <td>72.90±1.33</td>
    </tr>
    <tr>
      <td>ditto [17]</td>
      <td>63.35±0.69</td>
    </tr>
  </tbody>
  </table>


## References

[1] McCallum, Andrew Kachites, et al. "Automating the construction of internet portals with machine learning." _Information Retrieval_ 2000

[2] Giles, C. Lee, Kurt D. Bollacker, and Steve Lawrence. "CiteSeer: An automatic citation indexing system." _Proceedings of the third ACM conference on Digital libraries_. 1998.

[3] Sen, Prithviraj, et al. "Collective classification in network data." _AI magazine_ 2008.

[4] Tang, Jie, et al. "Arnetminer: extraction and mining of academic social networks." _SIGKDD_ 2008.

[5] Tang, Jiliang, Huiji Gao, and Huan Liu. "mTrust: Discerning multi-faceted trust in a connected world." _WSDM_ 2012.

[6] Bordes, Antoine, et al. "Translating embeddings for modeling multi-relational data." _NeurIPS_ 2013.

[7] Wu, Zhenqin, et al. "MoleculeNet: a benchmark for molecular machine learning." _Chemical science_ 2018.

[8] Ivanov, Sergei, Sergei Sviridov, and Evgeny Burnaev. "Understanding isomorphism bias in graph data sets." _arXiv_ 2019.

[9] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." _arXiv_ 2016.

[10] Veličković, Petar, et al. "Graph attention networks." _ICLR_ 2018.

[11] Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs." _NeurIPS_ 2017.

[12] Xu, Keyulu, et al. "How powerful are graph neural networks?."  _ICLR_ 2019.

[13] Chien, Eli, et al. "Adaptive universal generalized pagerank graph neural network." _ICLR_ 2021.

[14] Zhang, Ke, et al. "Subgraph federated learning with missing neighbor generation." _NeurIPS_ 2021.

[15] Xie, Han, et al. "Federated graph classification over non-iid graphs." _NeurIPS_ 2021.

[16] Li, Xiaoxiao, et al. "Fedbn: Federated learning on non-iid features via local batch normalization." _ICLR_ 2021.

[17]Li, Tian, et al. "Ditto: Fair and robust federated learning through personalization." _PMLR_ 2021.
