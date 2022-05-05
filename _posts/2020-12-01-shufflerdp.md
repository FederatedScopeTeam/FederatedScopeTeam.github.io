---
title: "Improving Utility and Security of the Shuffler-based Differential Privacy"
excerpt: "For frequency queries, introducing a new algorithm that achieves a better privacy-utility tradeoff via shuffling and a novel protocol that provides better protection against various attacks. Published in VLDB 2021."
excerptheader:
  image: "https://img.alicdn.com/imgextra/i1/O1CN01O7YNQy1wp69Cy5a6X_!!6000000006356-0-tps-819-262.jpg"
  caption: "Privacy amplification result: the privacy guarantee is amplified from epsilon_l to epsilon_c via shuffling."
tags:
  - "frequency queries"
  - "local differential privacy"
  - "privacy amplification"
  - "multiple-party computation"
---

When collecting information, local differential privacy (LDP) alleviates privacy concerns of users because their private information is randomized before being sent it to the central aggregator. LDP imposes large amount of noise as each user executes the randomization independently. To address this issue, recent work introduced an intermediate server with the assumption that this intermediate server does not collude with the aggregator. Under this assumption, less noise can be added to achieve the same privacy guarantee as LDP, thus improving utility for the data collection task.

This paper investigates this multiple-party setting of LDP. We analyze the system model and identify potential adversaries. We then make two improvements: a new algorithm that achieves a better privacy-utility tradeoff; and a novel protocol that provides better protection against various attacks. Finally, we perform experiments to compare different methods and demonstrate the benefits of using our proposed method.

Tianhao Wang, Bolin Ding, Min Xu, Zhicong Huang, Cheng Hong, Jingren Zhou, Ninghui Li, Somesh Jha:
Improving Utility and Security of the Shuffler-based Differential Privacy. Proc. VLDB Endow. 13(13): 3545-3558 (2020)
<a href="https://www.bolin-ding.com/papers/vldb20shufflerdp.pdf">download</a>
