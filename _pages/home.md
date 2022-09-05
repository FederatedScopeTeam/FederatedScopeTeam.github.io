---
layout: splash
permalink: /
hidden: true
header:
  overlay_color: "#5e616c"
  overlay_image: https://gw.alicdn.com/imgextra/i1/O1CN01qYK90Z29q60y56h36_!!6000000008118-2-tps-1919-390.png
  actions:
    - label: "<i class='fab fa-github'></i> View on GitHub"
      url: "https://github.com/alibaba/FederatedScope"
      target: "_blank"
excerpt: >
  An easy-to-use federated learning platform providing comprehensive functionalities.<br />
  <!--- <small><a href="https://github.com/alibaba/FederatedScope">View on GitHub</a></small> --->
feature_row:
  - image_path: https://img.alicdn.com/imgextra/i4/O1CN01fzGros1hKDCnLqMxd_!!6000000004258-2-tps-1440-1440.png
    alt: "easy-to-use"
    title: "Easy-to-use"
    excerpt: "Users are allowed to integrate their own components, including datasets, models, etc., into FederatedScope to conduct federated learning for specific applications."
    url: "/docs/own-case/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: https://img.alicdn.com/imgextra/i2/O1CN01J04V9G1LcbZ3c6kfo_!!6000000001320-2-tps-1440-1440.png
    alt: "event-driven architecture"
    title: "Event-driven"
    excerpt: "Fderated learning algorithms are modularized and expressed via defining events and corresponding handlers for the participants."
    url: "/docs/event-driven-architecture/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: https://img.alicdn.com/imgextra/i3/O1CN01xHTXRM1U6qdxTfvUl_!!6000000002469-2-tps-1440-1440.png
    alt: "flexible&extendible"
    title: "Flexible&Extendable"
    excerpt: "Developers can flexibly enrich the exchanged data and participants' behaviors, which is helpful for various real-world federated learning applications."
    url: "/docs/new-type/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: https://img.alicdn.com/imgextra/i2/O1CN01DFj5ml1hJIPdKe3Ho_!!6000000004256-2-tps-1440-1440.png
    alt: "personalization"
    title: "Personalization"
    excerpt: "We have implemented state-of-the-art personalized federated learning methods, and the well-designed interfaces make the development of new methods easy."
    url: "/docs/pfl/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: https://img.alicdn.com/imgextra/i4/O1CN0121gp8S21diO3rZUw6_!!6000000007008-2-tps-1440-1440.png
    alt: "auto-tuning"
    title: "Auto-tuning"
    excerpt: "Out-of-the-box HPO functionalities can save users from the tedious loop of model tuning, allowing them to focus on their innovations."
    url: "/docs/use-hpo/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
  - image_path: https://img.alicdn.com/imgextra/i3/O1CN01k24Ubf1UtOcpXSl9c_!!6000000002575-2-tps-1440-1440.png
    alt: "privacy-protection"
    title: "Privacy Protection"
    excerpt: "Technologies, including differential privacy, encryption, multi-party computation, etc., are provided to enhance the strength of privacy protection."
    url: "/docs/protected-msg/"
    btn_class: "btn--primary"
    btn_label: "Learn more"
---

{% include feature_row %}

## News [(more)]({{ "/news/" | relative_url }})

- 08/2022, Our KDD 2022 paper on federated graph learning receives the KDD Best Paper Award for ADS track!

- 07/2022, We are hosting the CIKM'22 AnalytiCup competition. For more details, please see [link]({{ "/competition/" | relative_url }})

- 06/2022, We release pFL-Bench, a comprehensive benchmark for personalized Federated Learning (pFL), containing 10+ datasets and 20+ baselines. [GitHub](https://github.com/alibaba/FederatedScope/tree/master/benchmark/pFL-Bench), [pdf](https://arxiv.org/pdf/2206.03655.pdf) 

- 06/2022, We release FedHPO-B, a benchmark suite for studying federated hyperparameter optimization. [GitHub](https://github.com/alibaba/FederatedScope/tree/master/benchmark/FedHPOB), [pdf](https://arxiv.org/abs/2206.03966.pdf)

- 06/2022, We release B-FHTL, a benchmark suit for studying federated hetero-task learning. [GitHub](https://github.com/alibaba/FederatedScope/tree/master/benchmark/B-FHTL), [pdf](https://arxiv.org/pdf/2206.03436v2.pdf)

- 05/2022, Our paper on federated graph learning package is accepted at KDD'22. [pdf](https://arxiv.org/pdf/2204.05562.pdf)

- 05/2022, FederatedScope v0.1.0, our first release, is available at [GitHub](https://github.com/alibaba/FederatedScope).
