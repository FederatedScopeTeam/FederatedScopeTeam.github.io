---
title: "FederatedScope: A Flexible Federated Learning Platform for Heterogeneity"
excerpt: "FederatedScope is a flexible and comprehensive federated learning platform proposed for tackling the heterogeneity in real-world federated learning applications. FederatedScope exploits an event-driven architecture to frame an FL course into multiple event-handler pairs for flexibly describing asynchronous federated learning with heterogeneous information exchanging, and provides rich built-in algorithms and the federated hyperparameter optimizer for conveniently resolving the unstable issues brought by heterogeneity."
excerptheader:
  image: "https://img.alicdn.com/imgextra/i1/O1CN015NinK21QkupBdABHe_!!6000000002015-0-tps-5088-2277.jpg"
  caption: "The standard FL course viewed from the perspective of event-driven architecture"
tags:
  - "federated learning"
  - "system"
---

Although remarkable progress has been made by the existing federated learning (FL) platforms to provide fundamental functionalities for development, these platforms cannot well tackle the challenges brought by the heterogeneity of FL scenarios from both academia and industry. 
To fill this gap, in this paper, we propose a flexible federated learning platform, named FederatedScope, for handling various types of heterogeneity in FL.
Considering both flexibility and extensibility, FederatedScope adopts an event-driven architecture to frame an FL course into event-handler pairs: the behaviors of participants are described in handlers, and triggered by events of message passing or meeting certain conditions in training.
For a new FL application, developers only need to specify the adopted FL algorithm by defining new types of events and the corresponding handling functions based on participants' behaviors, which would be automatically executed in an asynchronous way for balancing effectiveness and efficiency in FederatedScope.
Meanwhile, towards an easy-to-use platform, FederatedScope provides rich built-in algorithms, including personalization, federated aggregation, privacy protection, and privacy attack, for users to conveniently customize participant-specific training, fusing, aggregating, and protecting.
Besides, a federated hyperparameter optimization module is integrated into FederatedScope for users to automatically tune their FL systems for resolving the unstable issues brought by heterogeneity.
We conduct a series of experiments on the provided easy-to-use and comprehensive FL benchmarks to validate the correctness and efficiency of FederatedScope. 
We have released FederatedScope for users on [https://github.com/alibaba/FederatedScope](https://github.com/alibaba/FederatedScope) to promote research and industrial deployment of federated learning in a variety of real-world applications.

Yuexiang Xie, Zhen Wang, Daoyuan Chen, Dawei Gao, Liuyi Yao, Weirui Kuang, Yaliang Li, Bolin Ding, Jingren Zhou:
FederatedScope: A Flexible Federated Learning Platform for Heterogeneity
<a href="https://arxiv.org/pdf/2204.05011.pdf">download</a>
