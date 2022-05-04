---
title: "Federated Matrix Factorization with Privacy Guarantee"
excerpt: "Matrix factorization algorithms for recommender systems under both horizontal and vertical federated settings. Published in VLDB 2022."
excerptheader:
  image: "https://img.alicdn.com/imgextra/i3/O1CN01cfwdl01WW9l1yGIIb_!!6000000002795-2-tps-1906-660.png"
  caption: "Different settings of federated learning (a)-(c) with sensitive information exchanged on privacy boundaries (the dashed lines). Each u_i is a user embeddings, v_j is an item embedding."
tags:
  - "federated learning"
  - "recommender system"
---

Matrix factorization (MF) approximates unobserved ratings in a rating matrix, whose rows correspond to users and columns correspond to items to be rated, and has been serving as a fundamental building block in recommendation systems. This paper comprehensively studies the problem of matrix factorization in different federated learning (FL) settings, where a set of parties want to cooperate in training but refuse to share data directly. We first propose a generic algorithmic framework for various settings of federated matrix factorization (FMF) and provide a theoretical convergence guarantee. We then systematically characterize privacy-leakage risks in data collection, training, and publishing stages for three different settings and introduce privacy notions to provide end-to-end privacy protections. The first one is vertical federated learning (VFL), where multiple parties have the ratings from the same set of users but on disjoint sets of items. The second one is horizontal federated learning (HFL), where parties have ratings from different sets of users but on the same set of items. The third setting is local federated learning (LFL), where the ratings of the users are only stored on their local devices. We introduce adapted versions of FMF with the privacy notions guaranteed in the three settings. In particular, a new private learning technique called embedding clipping is introduced and used in all the three settings to ensure differential privacy. For the LFL setting, we combine differential privacy with secure aggregation to protect the communication between user devices and the server with a strength similar to the local differential privacy model, but much better accuracy. We perform experiments to demonstrate the effectiveness of our approaches.

Zitao Li, Bolin Ding, Ce Zhang, Ninghui Li, Jingren Zhou:
Federated Matrix Factorization with Privacy Guarantee. Proc. VLDB Endow. 15(4): 900-913 (2021)
<a href="https://www.bolin-ding.com/papers/vldb22fedmf.pdf">download</a>
