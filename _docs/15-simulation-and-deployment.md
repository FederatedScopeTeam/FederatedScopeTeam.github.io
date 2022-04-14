---
title: "Simulation and Deployment"
permalink: /docs/simulation-and-deployment/
excerpt: "About simulation and deployment"
last_modified_at: 2022-04-12T19:46:43-05:00
toc: true
layout: tuto
---

## Efficient and memory saving simulation

As we have discussed in [the post "Cross-device"]({{ "/docs/cross-device/" | relative_url }}), there are usually tremendous clients, ranging from hundreds to tens of thousands. In simulating such a scenario, there are mainly two challenges:

- How to handle such a large number of client-wise models?
- How to cache such a large number of aggregated updates?

Taking the famous benchmark LEAF [1] as an example, and suppose we apply a two-layer convolutional neural network to the FEMNIST dataset, although this model only occupies ~200MB, there are around 300 clients, and thus trivially maintaining 300 such models would consume more than 50GB.

In general, the neural architecture is consistent among these clients (exceptions mainly result from [personalization]({{ "/docs/pfl/" | relative_url }})). Meanwhile, the local updates happened at each client is often simulated one-by-one. Hence, it is feasible to maintain only **one** model for **all** the clients, which reduces the space complexity by the order of number of clients. FederatedScope has provided this memory saving mode, and users can enjoy it by setting:

```python
federate.share_local_model=True
```

Even though we only keep one model in GPU for simulating the local updates happened at each client, each client will produce its updated model, where caching all these models in GPU is impossible due to the very limited video memory. As the following figure shows, one compromise is to cache these updated models in the main RAM, which is often much larger than the video memory.

![](https://img.alicdn.com/imgextra/i2/O1CN01IR7eiv1ltFxfwBxRD_!!6000000004876-0-tps-616-200.jpg#crop=0&crop=0&crop=1&crop=1&id=WDZm8&originHeight=200&originWidth=616&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

However, as a well-known fact, the swap in-and-out between video memory and the main RAM is costly, leading to inefficient simulation.

Therefore, FederatedScope has provided an online counterpart for the vanilla FedAvg [2] aggregator, which makes it unnecessary to cache all these updated models. The rationale behind this online couterpart is straightforward. First, according to FedAvg, the aggregation operation is defined as follow:

$$
x = w_1 \times x_1 + w_2 \times x_2 + \ldots + w_n \times x_n,
$$

where $x_i$ denotes the $i$-th client's updated model, $w_i$ denotes the weight of $i$-th client, and $x$ denotes the aggregation. When the simulation is conducted one-by-one, the server sequentially receives $(w_1, x_1), (w_2, x_2), \ldots, (w_n, x_n)$. Then we maintain the result update now as follow:

$$
m_0 = 0, c_0 = 0\\

m_i = \frac{c_{i-1} \times m_{i-1} + w_i \times x_i}{ c_{i-1} + w_i }, c_i = c_{i-1} + w_i,
$$

where $m_n$ will be equal to $x$. Users can utilize this online aggregator by setting:

```python
federate.online_aggr=True
```

With both shared local model and this online aggregator, the simulation can be entirely conducted on GPU, if three times of the model size have not exceeded the video memory size. The procedure can be described as follow:

![](https://img.alicdn.com/imgextra/i4/O1CN01XvADNu1gBDnwbg32c_!!6000000004103-0-tps-740-322.jpg#crop=0&crop=0&crop=1&crop=1&id=hHPzf&originHeight=322&originWidth=740&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

## Empirical study

We use FederatedScope to conduct simulation for federally training a two-layer convolutional neural network on FEMNIST. We compare FederatedScope (with the efficient and memory saving mode) to FedML [3], regarding both the memory and time consumption. The results are as follow:

|  | Memory consumption (GB) | Time per round (s) |
| --- | --- | --- |
| FedML | 7.88 | 4.4 |
| FederatedScope | **2.76** | **1.4** |


As you can see from this table, FederatedScope enables users to conduct more efficient and memory saving simulation. We encourage users to reproduce this experiment by:

```bash
python federatedscaope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml federate.share_local_model True federate.online_aggr True
```

We also encourage contributions about accelerating FL simulation, which must be very helpful for both research and application.


## From simulation to deployment

FederatedScope provides a unified interface for both standalone simulation and distributed deployment. Therefore, to transfer from simulation to deployment, users only need to:

- Modify the configurations, add data_path, role, and communication address;
- Run multiple FL procedures, each denotes a participant, to build an FL course.

An example can be found in [Distributed Mode]({{ "/docs/quick-start/#distributed-mode" | relative_url }}).

We aim to provide more support for different data storage,  various software/hardware environments,  distributed systems scheduling in the future.

## Cross ML backends

One of the biggest challenges when applying federated learning in practice is to be compatible with different ML backends. Considering the situation that some participants perform local training based on Tensorflow while others use Pytorch, and these participants want to build up an FL course.
![](https://img.alicdn.com/imgextra/i1/O1CN01bw6qmA1ouZx51CAJq_!!6000000005285-0-tps-1617-433.jpg)

The most straightforward solution is to force all the participants to use the same ML backend. It can be feasible for some cross-device scenarios such as Gboard [4], where exists a powerful manager to unify the types of software and hardware environments. However, for cross-silo scenarios where each participant (usually a department or company) has already built up a large and complete system for local training, it is not practical and economical to unify the backends. 

Another solution is getting help from intermedia representation, such as ONNX [5] and TensorRT [6], which transfers the original program into a defined intermedia representation, and further interprets into the target language during the runtime. 

In order to be compatible with different ML backends, FederatedScope decouples the federal behaviors and training behaviors. The participants hold a trainer/aggregator object to encapsulate their training behaviors that might be related to their ML backends. Thus they can only care about high-level behaviors such as train or eval. For example, a client uses `trainer.train(model, data, configuration)`to perform local training but ignores what is the backend used behind.

In summary, if developers want to build up an FL course that involves participants with different ML backends, developers might need to:

- Customize backend-specific Trainers accordingly;
- A transformation to match the computation graph described in different backends.

We aim to provide more Trainers to support more widely-used ML backends in the future. 

## References

[1] Caldas, Sebastian, et al. "Leaf: A benchmark for federated settings." arXiv preprint arXiv:1812.01097 (2018).

[2] McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.

[3] He, Chaoyang, et al. "Fedml: A research library and benchmark for federated machine learning." arXiv preprint arXiv:2007.13518 (2020).

[4] Hard A, Rao K, Mathews R, et al. Federated learning for mobile keyboard prediction. arXiv preprint arXiv:1811.03604, 2018.

[5] Open standard for machine learning interoperability. [https://github.com/onnx/onnx](https://github.com/onnx/onnx)

[6] NVIDIA TensorRT. [https://developer.nvidia.com/zh-cn/tensorrt](https://developer.nvidia.com/zh-cn/tensorrt)
