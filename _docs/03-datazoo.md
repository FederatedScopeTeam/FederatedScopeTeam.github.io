---
title: "DataZoo"
permalink: /docs/datazoo/
excerpt: "About DataZoo."
last_modified_at: 2019-08-20T21:36:18-04:00
toc: true
---

**FederatedScope** provides a rich collection of datasets for researchers, including images, texts, graphs, recommendation systems and speeches. Our `DataZoo` contains real federation datasets as well as simulated federation datasets with a different `splitter`. And more datasets are coming soon!

To use our `DataZoo`, set `cfg.data.type = DATASET_NAME`. Downloading and pre-processing of our datasets is automatic, and you do not need to perform additional operations. Please feel free to use it! For more dataset-related settings, please refer to each dataset.

<a name="rn0zN"></a>
## Dataset

<a name="K2VER"></a>
### Computer Vision

-  **FMNIST**<br />FEMNIST is a federation image dataset from [LEAF](https://leaf.cmu.edu/) [1], and the task is image classification. The images are split by writers into clients. Moreover, you can set the sampling rate to sample from the clients via `cfg.data.subsample`.<br />Statistics: 62 classes, 805263 images, about 3500 clients. 
-  **Celeba**<br />Celeba is a federation image dataset from [LEAF](https://leaf.cmu.edu/) [1], and the task is image classification.  The images are split by humans into clients. Moreover, you can set the sampling rate to sample from the clients via `cfg.data.subsample`.<br />Statistics: 2 classes (smiling or not), 200288 images, about 9300 clients. 

<a name="wv41Y"></a>
### Natural Language Processing

-  **Shakespeare**<br />Shakespeare is a federation text dataset of Shakespeare Dialogues from [LEAF](https://leaf.cmu.edu/) [1], and the task is next-character prediction. Moreover, you can set the sampling rate to sample from the clients via `cfg.data.subsample`.<br />Statistics: 422615 sentences, about 1100 clients. 
-  **SubReddit**<br />SubReddit is a federation text dataset and subsampled of Reddit from [LEAF](https://leaf.cmu.edu/) [1], and the task is next-word prediction. Moreover, you can set the sampling rate to sample from the clients via `cfg.data.subsample`.<br />Statistics: 216858 sentences, about 800 clients. 
-  **Sentiment140**<br />Sentiment140 is a federation text dataset of Twitter from [LEAF](https://leaf.cmu.edu/) [1], and the task is Sentiment Analysis. Moreover, you can set the sampling rate to sample from the clients via `cfg.data.subsample`.<br />Statistics: 1600498 sentences, about 660000  clients. 

<a name="jhPKC"></a>
### Graph

For details of statistics, see FedGraph.

<a name="GmP1e"></a>
#### Node-level dataset

-  **FedDBLP** (including _dblp_conf_ and _dblp_org_)<br />FedDBLP is a federation citation network from the latest [DBLP](https://originalstatic.aminer.cn/misc/dblp.v13.7z) [2] dump, where each node corresponds to a published paper, and each edge corresponds to a citation. We use the bag-of-words of each paper's abstract as its node attributes and regard the theme of paper as its label. To simulate the scenario that a venue or an organizer forbids others to cite its papers, we allow users to split this dataset by each node's **venue** or the **organizer** of that venue. 
-  **FedcSBM**<br />FedcSBM from cSBM [3] can produce the synthetic graph dataset. For more details, see FedGraph. 
-  **FedCora**, **FedCiteSeer**, **FedPubMed**<br />FedCora, FedCiteSeer and FedPubMed are simulated federation datasets split from Cora [4], CiteSeer [5], PubMed [6] by `community_splitter` or `random_splitter`. For more details, see FedGraph. 

<a name="xcd5t"></a>
#### Link-level dataset

-  **Ciao**<br />Ciao [7] is a federation recommendation dataset from [FedGraphNN](https://arxiv.org/pdf/2104.07145v2.pdf), and its task is link classification. For more details, see FedGraph. 
-  **FedWN18**, **FedFB15K-237**<br />FedWN18 and FedFB15K-237 are simulated federation datasets split from WN18 and FB15K-237 [8] by `label_space_splitter`. For more details, see FedGraph. 

<a name="xVvAd"></a>
#### Graph-level dataset

-  **Multi-task dataset**<br />Multi_task dataset is a federation dataset contains several Sub-datasets named `graph_multi_domain_mol`, `graph_multi_domain_small`, `graph_multi_domain_mix`,  `graph_multi_domain_biochem`, and `graph_multi_domain_molv1`. In these datasets, each client holds some graphs from different domains from TUDataset [9], and their task is different from each other. 
   - `graph_multi_domain_mol`: 'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1'
   - `graph_multi_domain_small`: 'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'ENZYMES', 'DD', 'PROTEINS'
   - `graph_multi_domain_mix`: 'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1', 'ENZYMES', 'DD', 'PROTEINS', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI'
   - `graph_multi_domain_biochem`: 'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1', 'ENZYMES', 'DD', 'PROTEINS'
   - `graph_multi_domain_molv1`: 'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1', 'Mutagenicity', 'NCI109', 'PTC_MM', 'PTC_FR'
-  **FedHIV**, **FedProteins**, **FedIMDB**<br />FedHIV,  FedProteins and FedIMDB are simulated federation datasets split from HIV [10], Proteins [9], IMDB [9] by `instance_space_splitter`. For more details, see FedGraph. 

<a name="EW19L"></a>
### Recommendation System

-  **MovieLens**<br />MovieLens [11] is a series of movie recommendation datasets collected from the website MovieLens. To support different federated settings (VFL/HFL) and various datasets, all the MovieLens datasets inherit two parent classes `VMFDataset` and `HMFDataset`, which specific the splitting of MF datasets (VFL or HFL). 

<a name="GK2xc"></a>
### Audio and Speech

Coming soon...

<a name="HSzGF"></a>
### Synthetic

-  **Synthetic Mixture**<br />Synthetic_Mixture is a synthetic federated dataset from [FedEM](https://arxiv.org/pdf/2108.10252.pdf) [12], and its task is binary classification. The data distribution of each client is the mixture of (M) underlying distributions. 

<a name="UaFPw"></a>
## Tools

<a name="vU9eP"></a>
### Splitter

To generate simulated federation datasets, we provide `splitter` who are responsible for dispersing a given standalone dataset into multiple clients, with configurable statistical heterogeneity among them.

For euclidean data:

-  `random_splitter`<br />The data is randomly split into 𝑁 subsets with or without intersections. 
-  `label_space_splitter`<br />It is designed to provide label distribution skew via latent Dirichlet allocation (LDA). 

For graph data:

-  Node-level task 
   -  `community_splitter`:<br />Community detection algorithms such as Louvain are at first applied to partition a graph into several clusters. 
   -  `random_splitter`:<br />The node-set of the original graph is randomly split into 𝑁 subsets with or without intersections. 
-  Link-level task 
   -  `label_space_splitter`:<br />It is designed to provide label distribution skew via latent Dirichlet allocation (LDA). 
-  Graph-level task 
   -  `instance_space_splitter`:<br />It is responsible for creating feature distribution skew (i.e., covariate shift). 
   -  `multi_task_splitter`:<br />Different clients have different tasks. 

<a name="imW5m"></a>
## How to use

For the built-in datasets, you can configure them via a `.yaml` file.

```python
# Dataset related options
data:
	# Root directory where the data stored
	root: 'data'
	# Dataset name
	type: 'femnist'
  # Batch_size for DataLoader
	batch_size: 64
  # Drop last batch of DataLoader
	drop_last: False
  # Shuffle the train DataLoader
	shuffle: True
  # Transforms of data
	transforms: ''
  # Subsample of total client
	subsample: 1.0
  # Train, valid, test splits
	splits: [0.6, 0.2, 0.2]
	# Splitter, if not simulated dataset, disabled.
  splitter: 'random'
```

<a name="Reference"></a>
## Reference

[1] Caldas, Sebastian, et al. "Leaf: A benchmark for federated settings." _arXiv_ 2018.

[2] Tang, Jie, et al. "Arnetminer: extraction and mining of academic social networks." _SIGKDD_ 2008.

[3] Deshpande, Yash, et al. "Contextual stochastic block models." _NeurIPS_ 2018.

[4] McCallum, Andrew Kachites, et al. "Automating the construction of internet portals with machine learning." _Information Retrieval_ 2000

[5] Giles, C. Lee, Kurt D. Bollacker, and Steve Lawrence. "CiteSeer: An automatic citation indexing system." _Proceedings of the third ACM conference on Digital libraries_. 1998.

[6] Sen, Prithviraj, et al. "Collective classification in network data." _AI magazine_ 2008.

[7] Tang, Jiliang, Huiji Gao, and Huan Liu. "mTrust: Discerning multi-faceted trust in a connected world." _WSDM_ 2012.

[8] Bordes, Antoine, et al. "Translating embeddings for modeling multi-relational data." _NeurIPS_ 2013.

[9] Ivanov, Sergei, Sergei Sviridov, and Evgeny Burnaev. "Understanding isomorphism bias in graph data sets." _arXiv_ 2019.

[10] Wu, Zhenqin, et al. "MoleculeNet: a benchmark for molecular machine learning." _Chemical science_ 2018.

[11] Harper, F. Maxwell, and Joseph A. Konstan. "The movielens datasets: History and context." _Acm transactions on interactive intelligent systems_ 2015.

[12] Marfoq, Othmane, et al. "Federated multi-task learning under a mixture of distributions." _NeurIPS_ 2021.
