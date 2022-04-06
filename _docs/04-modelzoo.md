---
title: "ModelZoo"
permalink: /docs/modelzoo/
excerpt: "About ModelZoo."
last_modified_at: 2021-06-23T08:15:34-04:00
toc: true
layout: tuto
---

FederatedScope provides many built-in models in different deep learning fields, including Computer Vision, Natural Language Processing, Graph, Recommendation Systems, and Speech. Furthermore, more models are on the way!

To use our `ModelZoo`, set `cfg.model.type = Model_NAME`. And you can configure the model-related hyperparameters via a`yaml`file.

```python
# Some methods may leverage more than one model in each trainer
cfg.model.model_num_per_trainer = 1 
# Model name
cfg.model.type = 'lr'
cfg.model.use_bias = True
# For graph model
cfg.model.task = 'node'
# Hidden dim
cfg.model.hidden = 256
# Drop out ratio
cfg.model.dropout = 0.5
# in_channels dim. If 0, model will be built by data.shape
cfg.model.in_channels = 0
# out_channels dim. If 0, model will be built by label.shape
cfg.model.out_channels = 1
# In GPR-GNN, K = gnn_layer
cfg.model.gnn_layer = 2
cfg.model.graph_pooling = 'mean'
cfg.model.embed_size = 8
cfg.model.num_item = 0
cfg.model.num_user = 0
```

For more model-related settings, please refer to each model.

<a name="753a2225"></a>
## Computer Vision

-  **ConvNet2**<br />ConvNet2 (from `federatedscope/cv/model`) is a two-layer CNN for image classification. (`cfg.model.type = 'convnet2'`) 
```python
class ConvNet2(Module):
    def __init__(self, in_channels, h=32, w=32, hidden=2048, class_num=10, use_bn=True):
        ...
```


-  **ConvNet5**<br />ConvNet5 (from `federatedscope/cv/model`) is a five-layer CNN for image classification. (`cfg.model.type = 'convnet5'`) 
```python
class ConvNet5(Module):
    def __init__(self, in_channels, h=32, w=32, hidden=2048, class_num=10):
        ...
```


-  **VGG11**<br />VGG11 [1] (from `federatedscope/cv/model`) is an 11 layer CNN with very small (3x3) convolution filters for image classification. It is from [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556v6.pdf). (`cfg.model.type = 'vgg11'`) 
```python
class VGG11(Module):
    def __init__(self, in_channels, h=32, w=32, hidden=128, class_num=10):
        ...
```


<a name="49f53d4e"></a>
## Natural Language Processing

-  **LSTM**<br />LSTM [2] (from `federatedscope/nlp/model`) is a type of RNN that solves the vanishing gradient problem through additional cells, input and output gates. (`cfg.model.type = 'lstm'`) 
```python
class LSTM(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, n_layers=2, embed_size=8):
        ...
```


<a name="Graph"></a>
## Graph

-  **GCN**<br />GCN [3] (from `federatedscope/gfl/model`) is a kind of Graph Neural Networks from [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), which is adapted for node-level, link-level and graph-level tasks. (`cfg.model.type = 'gcn'`, `cfg.model.task = 'node'`) 
```python
class GCN_Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden=64, max_depth=2, dropout=.0):
        ...
```


-  **GAT**<br />GAT [4] (from `federatedscope/gfl/model`) is a kind of Graph Neural Networks from [Graph Attention Networks](https://arxiv.org/abs/1710.10903). GAT employ attention mechanisms to node neighbors to learn attention coefficients, which is adapted for node-level, link-level and graph-level tasks.  (`cfg.model.type = 'gat'`, `cfg.model.task = 'node' # node, link or graph`) 
```python
class GAT_Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden=64, max_depth=2, dropout=.0):
        ...
```


-  **GraphSAGE**<br />GraphSAGE [5] (from `federatedscope/gfl/model`) is a general inductive GNN framework, from [Inductive Representation Learning on Large Graphs](https://arxiv.org/pdf/1706.02216v4.pdf). GraphSAGE learns a function that generates embeddings by sampling and aggregating from the local neighborhood of each node, which is adapted for node-level and link-level tasks.  (`cfg.model.type = 'sage'`, `cfg.model.task = 'node' # node, link or graph`) 
```python
class SAGE_Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden=64, max_depth=2, dropout=.0):
        ...
```


-  **GPR-GNN**<br />GPR-GNN [6] (from `federatedscope/gfl/model`) adaptively learns the Generalized PageRank weights so as to jointly optimize node feature and topological information extraction from [Adaptive Universal Generalized PageRank Graph Neural Network](https://arxiv.org/pdf/2006.07988v6.pdf), which is adapted for node-level and link-level tasks.  (`cfg.model.type = 'gpr'`, `cfg.model.task = 'node' # node or link`) 
```python
class GPR_Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden=64, K=10, dropout=.0, ppnp='GPR_prop', alpha=0.1, Init='PPR'):
        ...
```


-  **GIN**<br />GIN [7] (from `federatedscope/gfl/model`) generalizes the Weisfeiler-Lehman test and achieves maximum discriminative power among GNNs from [How Powerful are Graph Neural Networks?](https://arxiv.org/pdf/1810.00826v3.pdf) which is adapted for graph-level tasks.  (`cfg.model.type = 'gin'`, `cfg.model.task = 'graph'`) 
```python
class GIN_Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden=64, max_depth=2, dropout=.0):
        ...
```


<a name="0b35d755"></a>
## Recommendation System

- **MF models**<br />MF model [8] (from `federatedscope/mf/model`) has two trainable parameters: user embedding and item embedding. Based on the given federated setting, they share different embedding with the other participators. FederatedScope achieves `VMFNet`and `HMFNet`to support federated MF, and both of them inherit the basic MF model class `BasicMFNet`. 

  
```python
class VMFNet(BasicMFNet):
    name_reserve = "embed_item"


class HMFNet(BasicMFNet):
    name_reserve = "embed_user"
```

<a name="Speech"></a>

## Speech

Coming Soon!

<a name="Reference"></a>
## References

[1] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." _arXiv preprint arXiv_ 2014.

[2] Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." _Neural computation_ 1997.

[3] Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." _arXiv_ 2016.

[4] Veličković, Petar, et al. "Graph attention networks." _ICLR_ 2018.

[5] Hamilton, Will, Zhitao Ying, and Jure Leskovec. "Inductive representation learning on large graphs." _NeurIPS_ 2017.

[6] Chien, Eli, et al. "Adaptive universal generalized pagerank graph neural network." _ICLR_ 2021.

[7] Xu, Keyulu, et al. "How powerful are graph neural networks?."  _ICLR_ 2019.

[8] Yongjie, Du, et al. "Federated matrix factorization for privacy-preserving recommender systems" _VLDB_ 2022.
