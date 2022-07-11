---
layout: single
permalink: /competition/
title: "CIKM 2022 AnalytiCup Competition"
author_profile: false
classes: wide
---

# Federated Hetero-Task Learning


## Introduction
We propose a new task, federated hetero-task learning, which meets the requirements of a wide range of real-world scenarios, while also promoting the interdisciplinary research of Federated Learning with Multi-task Learning, Model Pre-training, and AutoML. We have prepared an easy-to-use toolkit based on **FederatedScope [1,2]** to help participants easily explore this challenging yet manageable task from several perspectives, and also set up a fair testbed and different formats of awards for participants.

**We are going to run this competition on Tianchi competition platform, where the** <span style="color:blue">link</span> **to the dedicated page will be coming soon (July 15, 2022)!**


## Awards

- Prizes: 
  - 1st place: 5000 USD
  - 2nd place: 3000 USD
  - 3rd place: 1500 USD
  - 4th-10th prize: 500 USD each

- Certification: 
  - 1st – 20th: Certification with rank
  - Others: Certification with participation


## Schedule

- July 15, 2022: Competition launch. Sample dataset releases and simulation environment opens. Participants can register, join the discussion forum, upload the code for training and get feedback from leadboard.
- Sept 1, 2022: Registration ends.
- Sept 11, 2022: Submission ends. 
- Sept 12, 2022: Submitting the technical report and code ends. Codes of top 30 teams will automatically be migrated into a checking phase. 
- Sept 18, 2022: Notification of checking results. 
- Sept 21, 2022: Announcement of the CIKM 2022 AnalytiCup Winner.
- Oct 17, 2022: Beginning of CIKM 2022.

All deadlines are at 11:59 PM UTC on the corresponding day. The organizers reserve the right to update the contest timeline if necessary.


## Problem description
In federated hetero-task learning, the learning goals of different clients are different. In practice, this setting is often observed due to personalized requirements of different clients, or the difficulty in aligning goals among multiple clients. Specifically, the problem is defined as follows:
  - Input: Several clients, each one is associated with a different dataset (feature space might be different) and a different learning objective.
  - Output: A learned model for each client, and a central model across clients (this central model is the outputted model in traditional federated learning).
  - Evaluation metric: The averaged improvement ratio (against isolated learning) across all the clients.

We will provide the dataset for this competition via Tianchi. For now, we encourage you to see the exemplary federated hetero-task learning datasets defined in **B-FHTL [3]**, where the design and construction of these datasets are illustrated in the following picture:
<img src="https://img.alicdn.com/imgextra/i3/O1CN01yVaEBB25d2Gnu9mnh_!!6000000007548-0-tps-3422-1888.jpg" width="480" class="align-center">


## References

[1] FederatedScope: A Flexible Federated Learning Platform for Heterogeneity. arXiv preprint 2022. [pdf](https://arxiv.org/pdf/2204.05011.pdf)

[2] FederatedScope-GNN: Towards a Unified, Comprehensive and Efficient Package for Federated Graph Learning. KDD 2022. [pdf](https://arxiv.org/pdf/2204.05562.pdf)

[3] A Benchmark for Federated Hetero-Task Learning. arXiv preprint 2022. [pdf](https://arxiv.org/pdf/2206.03436v2.pdf)
