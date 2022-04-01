---
title: "Accelerating Federated Hyperparameter Optimization"
permalink: /docs/improve-hpo/
excerpt: "About improving HPO."
last_modified_at: 2020-05-01T10:22:56-04:00
toc: true
---

In essence, hyperparameter optimization (HPO) is a trial-and-error procedure, where each trial often means an entire training course and then evaluation. Under the federated learning (FL) setting, each FL training course consists of hundereds of communication rounds, which makes it unaffordable to conduct such trials again and again.

Hence, FederatedScope has provided functionalities to conduct low-fidelity HPO, which allows users to trade evaluation precision for efficiency.

<a name="e038721e"></a>
## Achieving low-fidelity HPO under the federated learning setting

We encourage users to try low-fidelity HPO with our provided toy example:

```bash
python federatedscope/hpo.py --cfg federatedscope/example_configs/toy_hpo.yaml hpo.scheduler sha hpo.sha.budgets [1,3,9]
```

where Successive Halving Algorithm (SHA) [2] is employed as our HPO scheduler. At first, the number of initial candidates is determined by:

> #initial candidates = elimination rate ** #elimination rounds,


where, in this example, `hpo.sha.elim_round_num=3` and `hpo.sha.elim_rate=3`. Thus, the scheduler begins with `3**3=27` initial candidate configurations randomly sampled from the declared search space. As we have introduced in the [primary HPO tutorial]({{ "/docs/use-hpo/" | relative_url }}), SHA iteratively filters the maintained candidates round-by-round untill only one configuration remaining. According to the settings for elimination rate, the number of elimination rounds, and the budgets (i.e., `hpo.sha.budgets`), the scheduler proceeds as follow:

> 1st iteration: Each of the 27 candidates will be trained for 1 round.

2nd iteration: Each of the 9 candidates outstanding from last iteration will be trained for 3 rounds

3rd iteration: Each of the 3 candidates outstanding from last iteration will be trained for 9 rounds


As your can see, by controlling the resource allocation (here the assigned number of training rounds), the winning configuratioin enjoys the most resource---1+3+9=13 training rounds, while we didn't waste much resource on those poor configurations. The total resource consumed is 27 _ 1 + 9 _ 3 + 3 _ 9 = 81 training rounds. In contrast, the SHA example presented in our primary HPO tutorial has not specified the budget and consumes (27 + 9 + 3) _ 20 = 780 training rounds. Although insufficient training rounds may lead to configuration rankings that are less correlated with the ground-truth rankings, training round provides us an aspect to control the fidelity to trade-off between accuracy and efficiency.

The eventual results of the above example are as follow:

![](https://img.alicdn.com/imgextra/i3/O1CN01eeT2dt2ADSCtJCmms_!!6000000008169-2-tps-374-70.png#crop=0&crop=0&crop=1&crop=1&id=Kjuna&originHeight=70&originWidth=374&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

where the test loss cannot be minimized as small as that of the SHA example presented in our primary HPO tutorial. However, the goal of HPO is to determine the hyperparameters. In this sense, this SHA example with low-fidelity has attained the same decisions as that example, while consuming much less resource.

Another aspect that enables to control the fidelity is the ratio of clients sampled in each training round. FederatedScope allows users to specify either the sample rate (via `federate.sample_client_rate`) or the number of sampled client (via `federate.sample_client_num`), with the former prioritized. If none of them has been set, all clients would be involved in each training round.

<a name="hb47h"></a>
## Empirical study

We evaluate the effectiveness of SHA with low-fidelity on a case where graph convolutional network (GCN) is to be trained on the citation network Cora. In this experiment, we use the same setting as the above example, and thus there are 81 training rounds in total can be consumed. For the purpose of comparison, we adopt random search (RS) algorithm [4] as our baselines, where sample size of 81, 27, and 9 are considered, with training rounds per trial 1, 3, and 9, respectively.

We show the corresponding performances of their searched hyperparameters in the following table:

| Scheduler | Test accuracy (%) |
| --- | --- |
| SHA | **88.83** |
| RS (81-1) | 88.51 |
| RS (27-3) | 88.67 |
| RS (9-9) | 88.67 |


where SHA with the given allocation outperforms all RS settings. As we sequentially simulate each trial, the HPO procedure can be visualized by plotting the best test accuracy achieved up to the latest trial:

![](https://img.alicdn.com/imgextra/i2/O1CN01XRpufx1kBurnLVkS0_!!6000000004646-2-tps-687-517.png#crop=0&crop=0&crop=1&crop=1&height=301&id=F22Iq&originHeight=517&originWidth=687&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=400)

Users can easily reproduce this HPO experiment by executing:

```bash
python federatedscope/hpo.py --cfg federatedscope/example_configs/hpo_for_gnn.yaml
```

Furthermore, we present an empirical comparison for different fidelities considered in optimizing the hyperparameters of GCN and GPR-GNN, respectively. Again, we employ SHA as our scheduler and controll the fidelity by considering:

- Training rounds for each trial in `{1, 2, 4, 8}`;
- Client sampling rate in `{20%, 40%, 80%, 100%}`.

With different combinations of the them, we conduct HPO with different fideilities, which might result in different optimal hyperparameters. As we have construct the ground-truth rankings, the accuracy of each combination can be measured by the rank of its resulting optimal hyperparameters. We illustrate the results in the following two figures:

| GCN | GPR-GNN |
| --- | --- |
| ![](https://img.alicdn.com/imgextra/i2/O1CN01dYewvC1Lj166Em6JK_!!6000000001334-0-tps-1819-1348.jpg#crop=0&crop=0&crop=1&crop=1&id=TYXwT&originHeight=1348&originWidth=1819&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) | ![](https://img.alicdn.com/imgextra/i1/O1CN01Xqfglq1KdgsQKc0j9_!!6000000001187-0-tps-1766-1348.jpg#crop=0&crop=0&crop=1&crop=1&id=ViKv8&originHeight=1348&originWidth=1766&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) |


where, as expected, higher fidelity leads to better configuration for both kinds of graph neural network models. At first, we want to remind our readers that the left-upper region in each grid table corresponds to extremely low-fidelity HPO. Although their performances cannot be comparable to those in the other regions, they have successfully eliminated a considerable fraction of poor configurations. Meanwhile, increasing fidelity through the two aspects, i.e., client sampling rate and the number of training rounds, reveal comparable efficiency in improving the quality of searched configurations. This property provides valuable flexibility for practitioners to keep a fixed fidelity while trade-off between these two aspects according to their system status (e.g., network latency and how the dragger behaves) [3].

<a name="cSb9N"></a>
## Weight-sharing and personalized HPO

In general, the hyperparameters of an FL algorithm can be classified into two categories:

- Server-side: The hyperparameters impact the aggregation, e.g., the learning rate of server's optimizer in FedOPT [5].
- Client-side: The hyperparameters impact the local updates, e.g., the local update steps, the batch size, the learning rate, etc.

In traditional standalone machine learning, only one specific configuration can be evaluated in each trial. When we consider an FL training course, since there are often more than one clients sampled in each round, it is possible to let different sampled clients explore different client-side configurations. From the perspective of multi-arm bandit, the agent (i.e., the HPO scheduler) can interact with many bandits under the FL setting, as the following figure shows.

![](https://img.alicdn.com/imgextra/i4/O1CN014NbGMH1HuEKXaKLnv_!!6000000000817-0-tps-810-346.jpg#crop=0&crop=0&crop=1&crop=1&id=N47md&originHeight=346&originWidth=810&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

To utilize this idea, FexEx [1] makes an analogy to the weight-sharing trick widely adopted in one-shot neural architecture search (NAS). Roughly speaking, one-shot NAS regards all candidate operators as an super-graph, evaluates a sampled subgraph at each step, and updates the controller (i.e., sampler) according to the feedback. In analogy, we could design a controller to sample configuration and, at each round, independently sample the client-side configuration for each client.

FederatedScope has provided flexible interfaces to instantiate such an idea into Federated HPO algorithm (more details can be found at this [post]({{ "/docs/new-type/" | relative_url }})). For instance, implementing FedEx can be sketched as the following steps:

1. To inherit the base server class and integrate your implementation of the controller into the server;
1. To augment the parameter broadcast method, including the sampled client-side configuration in the message;
1. To inherit the base client class and extend the handler---initializing the local model with received parameters and reset hyperparameters by the received choices;
1. To extend the handler of server, that is, updating the controller w.r.t. received performances.

We will provide an implementation of FedEx later, and we encourage users to contribute more latest federated HPO algorithms to FederatedScope.

It is worth noting that the bandits sampled in each round are _different_ due to the non-i.i.d.ness of client-wise data distributions. Thus, a promising future direction is to explore the [personalization functionalities]({{ "/docs/pfl/" | relative_url }}) to achieve personalized HPO.

---

<a name="Reference"></a>
## Reference

- [1] Khodak, Mikhail, et al. "Federated hyperparameter tuning: Challenges, baselines, and connections to weight-sharing." Advances in Neural Information Processing Systems 34 (2021).
- [2] Li, Lisha, et al. "Hyperband: A novel bandit-based approach to hyperparameter optimization." The Journal of Machine Learning Research 18.1 (2017): 6765-6816.
- [3] Zhang, Huanle, et al. "Automatic Tuning of Federated Learning Hyper-Parameters from System Perspective." arXiv preprint arXiv:2110.03061 (2021).
- [4] Bergstra, James, and Yoshua Bengio. "Random search for hyper-parameter optimization." Journal of machine learning research 13.2 (2012).
- [5] Asad, Muhammad, Ahmed Moustafa, and Takayuki Ito. "FedOpt: Towards communication efficiency and privacy preservation in federated learning." Applied Sciences 10.8 (2020): 2864.
