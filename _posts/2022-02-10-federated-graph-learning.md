---
title: "EasyFGL: Towards a Unified, Comprehensive and Efficient Platform for Federated Graph Learning"
excerpt: "EasyFGL is an easy-to-use python package for federated graph learning. We built it upon FederatedScope so that the requirements for expressing federated graph learning algorithms can be met effortlessly. Benefiting from the comprehensive off-the-shelf datasets, splitters, graph neural network models, etc., we have set up many useful benchmarks for federated graph learning."
excerptheader:
  image: "/assets/images/fed-gl-stack.jpg"
  caption: "Overview of EasyFGL."
categories:
  - "federated learning"
  - "graph"
tags:
  - "federated learning"
  - "graph"
---

Most existing benchmarks (e.g., LEAF) and federated learning packages focus on vision and language tasks, while federated graph learning has not been supported as well as them. However, there are lots of graph related federated learning scenarios in real-world business, including anti-money laudering and, medicine. Thus, we decide to build EasyFGL based on FederatedScope to meet the demand for an easy-to-use and comprehensive federated graph learning platform.

Benefiting from the message-oriented paradigm of FederatedScope, EasyFGL provides a unified view to express various federated graph learning algorithms that require exchanges of heterogeneous data across the participants and diverse participants' behaviors. Meanwhile, EasyFGL has provided many federated graph datasets, splitters for constructing federated datasets from standalone ones, many state-of-the-art graph neural networks. As a result, it is convenient for our users to set up benchmarks that can be easily reproduced. Moreover, we have implemented the auto-tuning component and personalization interface to further improve the federated graph learning algorithms. Considering the exchanges of node embeddings, generative models, etc., in many representative federated graph learning algorithms, we also provided off-the-shelf privacy attack and defense methods, which help our users validate the privacy-preserving property of their algorithms.
