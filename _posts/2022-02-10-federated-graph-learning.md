---
title: "FederatedScope-GNN: Towards a Unified, Comprehensive and Efficient Package for Federated Graph Learning"
excerpt: "FederatedScope-GNN is an easy-to-use python package for federated graph learning. We built it upon FederatedScope so that the requirements for expressing federated graph learning algorithms can be met effortlessly. Benefiting from the comprehensive off-the-shelf datasets, splitters, graph neural network models, etc., we have set up many useful benchmarks for federated graph learning."
excerptheader:
  image: "https://img.alicdn.com/imgextra/i3/O1CN01zMCqUB1uWeyGfML3Y_!!6000000006045-0-tps-3238-2627.jpg"
  caption: "Overview of FederatedScope-GNN."
tags:
  - "federated learning"
  - "graph"
---

The incredible development of federated learning (FL) has benefited various tasks in the domains of computer vision and natural language processing, and the existing frameworks such as TFF and FATE has made the deployment easy in real-world applications.
However, federated graph learning (FGL), even though graph data are prevalent, has not been well supported due to its unique characteristics and requirements. The lack of FGL-related framework increases the efforts for accomplishing reproducible research and deploying in real-world applications.
Motivated by such strong demand, in this paper, we first discuss the challenges in creating an easy-to-use FGL package and accordingly present our implemented package FederatedScope-GNN, which provides (1) a unified view for modularizing and expressing FGL algorithms; (2) comprehensive DataZoo and ModelZoo for out-of-the-box FGL capability; (3) an efficient model auto-tuning component; and (4) off-the-shelf privacy attack and defense abilities. 
We validate the effectiveness of FederatedScope-GNN by conducting extensive experiments, which simultaneously gains many valuable insights about FGL for the community. Moreover, we employ FederatedScope-GNN to serve the FGL application in real-world E-commerce scenarios, where the attained improvements indicate great potential business benefits. We publicly release FederatedScope-GNN, as submodules of FederatedScope, at [https://github.com/alibaba/FederatedScope](https://github.com/alibaba/FederatedScope) to promote FGL's research and enable broad applications that would otherwise be infeasible due to the lack of a dedicated package.

Zhen Wang, Weirui Kuang, Yuexiang Xie, Liuyi Yao, Yaliang Li, Bolin Ding, Jingren Zhou:
FederatedScope-GNN: Towards a Unified, Comprehensive and Efficient Package for Federated Graph Learning
<a href="https://arxiv.org/pdf/2204.05562.pdf">download</a>
