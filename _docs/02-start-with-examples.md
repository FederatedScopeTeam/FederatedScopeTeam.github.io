---
title: "Start With Examples"
permalink: /docs/examples/
excerpt: "Federated Learning (FL) is a learning paradigm for collaboratively learning models from isolated data without directly sharing privacy information, which helps to satisfy the requirements of privacy protection of the public."
last_modified_at: 2022-08-04T08:48:05-04:00
redirect_from:
  - /theme-setup/
toc: true
layout: tuto
---

## Prepare datasets & models

To run an FL course, firstly you should prepare datasets for FL. The DataZoo provided in FederatedScope can help to automatically download and preprocess widely-used public datasets from various FL applications, including computer vision, natural language processing, graph learning, recommendation, etc. Users can conveniently conduct experiments on the provided dataset via specifying `cfg.data.type = DATASET_NAME`in the configuration.  
We also support users to adopt customized datasets, please refer to [DataZoo]({{ "/docs/datazoo/" | relative_url }}) for more details about the provided datasets, and refer to [Customized Datasets]({{ "/docs/own-case/#load-a-dataset" | relative_url }}) for introducing customized datasets in FederatedScope.

Secondly, you should specify the model architecture that will be federally trained, such as ConvNet or LSTM. FederatedScope provides the ModelZoo that contains the implementation of widely-used model architectures for various FL applications. Users can set up `cfg.model.type = MODEL_NAME` to apply a specific model architecture in FL tasks. We allow users to use customized models via registering without caring about the federated process. You can refer to [ModelZoo]({{ "/docs/modelzoo/" | relative_url }}) for more details about how to customize models.

For a vanilla FL course, all participants share the same model architecture and training configuration. And FederatedScope also supports adopting client-specific models and training configurations (known as personalized FL) to handle the non-IID issue in practical  FL applications, please refer to [Personalized FL]({{ "/docs/pdf/" | relative_url }}) for more details. 

## Run an FL course with configurations

Note that FederatedScope provides a unified view for both standalone simulation and distributed deployment, therefore users can easily run an FL course with standalone mode or distributed mode via configuring. 

### Standalone mode

The standalone mode in FederatedScope means to simulate multiple participants (servers and clients) in a single device, while participants' data are isolated from each other and their models might be shared via message passing. 

Here we demonstrate how to run a vanilla FL course with FederatedScope, with setting `cfg.data.type = 'FEMNIST'`and `cfg.model.type = 'ConvNet2'` to run vanilla FedAvg [1] for an image classification task.
Users can include more training configurations, such as `cfg.federated.total_round_num`, `cfg.data.batch_size`, and `cfg.train.optimizer.lr`, in the configuration (a .yaml file), and run a vanilla FL course as: 

```bash
# Run with default configurations
python federatedscope/main.py --cfg scripts/example_configs/femnist.yaml
# Or with custom configurations
python federatedscope/main.py --cfg scripts/example_configs/femnist.yaml federated.total_round_num 50 data.batch_size 128
```

Then users can observe some monitored metrics during the training process as:

```
INFO: Server has been set up ...
INFO: Model meta-info: <class 'federatedscope.cv.model.cnn.ConvNet2'>.
... ...
INFO: Client has been set up ...
INFO: Model meta-info: <class 'federatedscope.cv.model.cnn.ConvNet2'>.
... ...
INFO: {'Role': 'Client #5', 'Round': 0, 'Results_raw': {'train_loss': 207.6341676712036, 'train_acc': 0.02, 'train_total': 50, 'train_loss_regular': 0.0, 'train_avg_loss': 4.152683353424072}}
INFO: {'Role': 'Client #1', 'Round': 0, 'Results_raw': {'train_loss': 209.0940284729004, 'train_acc': 0.02, 'train_total': 50, 'train_loss_regular': 0.0, 'train_avg_loss': 4.1818805694580075}}
INFO: {'Role': 'Client #8', 'Round': 0, 'Results_raw': {'train_loss': 202.24929332733154, 'train_acc': 0.04, 'train_total': 50, 'train_loss_regular': 0.0, 'train_avg_loss': 4.0449858665466305}}
INFO: {'Role': 'Client #6', 'Round': 0, 'Results_raw': {'train_loss': 209.43883895874023, 'train_acc': 0.06, 'train_total': 50, 'train_loss_regular': 0.0, 'train_avg_loss': 4.1887767791748045}}
INFO: {'Role': 'Client #9', 'Round': 0, 'Results_raw': {'train_loss': 208.83140087127686, 'train_acc': 0.0, 'train_total': 50, 'train_loss_regular': 0.0, 'train_avg_loss': 4.1766280174255375}}
INFO: ----------- Starting a new training round (Round #1) -------------
... ...
INFO: Server: Training is finished! Starting evaluation.
INFO: Client #1: (Evaluation (test set) at Round #20) test_loss is 163.029045
... ...
INFO: Server: Final evaluation is finished! Starting merging results.
... ...
```

### Distributed mode
The distributed mode in FederatedScope denotes running multiple procedures to build up an FL course, where each procedure plays as a participant (server or client) that instantiates its model and loads its data. The communication between participants is already provided by the communication module of FederatedScope.

To run with distributed mode, you only need to:

- Prepare isolated data file and set up `cfg.distribute.data_file = PATH/TO/DATA` for each participant;
- Change `cfg.federate.model = 'distributed'`, and specify the role of each participant  by `cfg.distributed.role = 'server'/'client'`.
- Set up a valid address by `cfg.distribute.server_host/client_host = x.x.x.x` and `cfg.distribute.server_port/client_port = xxxx`. (Note that for a server, you need to set up `server_host` and `server_port` for listening messages, while for a client, you need to set up `client_host` and `client_port` for listening as well as `server_host` and `server_port` for joining in an FL course)

We prepare a synthetic example for running with distributed mode:

```bash
# For server
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_server.yaml distribute.data_file 'PATH/TO/DATA' distribute.server_host x.x.x.x distribute.server_port xxxx

# For clients
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_client_1.yaml distribute.data_file 'PATH/TO/DATA' distribute.server_host x.x.x.x distribute.server_port xxxx distribute.client_host x.x.x.x distribute.client_port xxxx
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_client_2.yaml distribute.data_file 'PATH/TO/DATA' distribute.server_host x.x.x.x distribute.server_port xxxx distribute.client_host x.x.x.x distribute.client_port xxxx
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_client_3.yaml distribute.data_file 'PATH/TO/DATA' distribute.server_host x.x.x.x distribute.server_port xxxx distribute.client_host x.x.x.x distribute.client_port xxxx
```

An executable example with generated toy data can be run with (a script can be found in `scripts/distributed_scripts/run_distributed_lr.sh`):
```bash
# Generate the toy data
python scripts/gen_data.py

# Firstly start the server that is waiting for clients to join in
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_server.yaml distribute.data_file toy_data/server_data distribute.server_host 127.0.0.1 distribute.server_port 50051

# Start the client #1 (with another process)
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_client_1.yaml distribute.data_file toy_data/client_1_data distribute.server_host 127.0.0.1 distribute.server_port 50051 distribute.client_host 127.0.0.1 distribute.client_port 50052
# Start the client #2 (with another process)
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_client_2.yaml distribute.data_file toy_data/client_2_data distribute.server_host 127.0.0.1 distribute.server_port 50051 distribute.client_host 127.0.0.1 distribute.client_port 50053
# Start the client #3 (with another process)
python federatedscope/main.py --cfg scripts/distributed_scripts/distributed_configs/distributed_client_3.yaml distribute.data_file toy_data/client_3_data distribute.server_host 127.0.0.1 distribute.server_port 50051 distribute.client_host 127.0.0.1 distribute.client_port 50054
```

And you can observe the results as (the IP addresses are anonymized with 'x.x.x.x'):

```
INFO: Server: Listen to x.x.x.x:xxxx...
INFO: Server has been set up ...
Model meta-info: <class 'federatedscope.core.lr.LogisticRegression'>.
... ...
INFO: Client: Listen to x.x.x.x:xxxx...
INFO: Client (address x.x.x.x:xxxx) has been set up ...
Client (address x.x.x.x:xxxx) is assigned with #1.
INFO: Model meta-info: <class 'federatedscope.core.lr.LogisticRegression'>.
... ...
{'Role': 'Client #2', 'Round': 0, 'Results_raw': {'train_avg_loss': 5.215108394622803, 'train_loss': 333.7669372558594, 'train_total': 64}}
{'Role': 'Client #1', 'Round': 0, 'Results_raw': {'train_total': 64, 'train_loss': 290.9668884277344, 'train_avg_loss': 4.54635763168335}}
----------- Starting a new training round (Round #1) -------------
... ...
INFO: Server: Training is finished! Starting evaluation.
INFO: Client #1: (Evaluation (test set) at Round #20) test_loss is 30.387419
... ...
INFO: Server: Final evaluation is finished! Starting merging results.
... ...
```

## Run FS with configured scripts

We provide some scripts for reproducing existing algorithms with FederatedScope, which are constantly being updated in the [scripts](https://github.com/alibaba/FederatedScope/tree/master/scripts) folder. You learn how to configure FS and reproduce the results with them.

- [Distribute Mode](https://github.com/alibaba/FederatedScope/tree/master/scripts#distribute-mode)

- [Asynchronous Training Strategy](https://github.com/alibaba/FederatedScope/tree/master/scripts#asynchronous-training-strategy)

- [Graph Federated Learning](https://github.com/alibaba/FederatedScope/tree/master/scripts#graph-federated-learning)

- [Attacks in Federated Learning](https://github.com/alibaba/FederatedScope/tree/master/scripts#attacks-in-federated-learning)

- [Federated Optimization Algorithm](https://github.com/alibaba/FederatedScope/tree/master/scripts#federated-optimization-algorithm)

- [Personalized Federated Learning](https://github.com/alibaba/FederatedScope/tree/master/scripts#personalized-federated-learning)

- [Differential Privacy in Federated Learning](https://github.com/alibaba/FederatedScope/tree/master/scripts#differential-privacy-in-federated-learning)

- [Matrix Factorization in Federated Learning](https://github.com/alibaba/FederatedScope/tree/master/scripts#matrix-factorization-in-federated-learning)

## References

[1] McMahan B, Moore E, Ramage D, et al. "Communication-efficient learning of deep networks from decentralized data". Artificial intelligence and statistics. PMLR, 2017: 1273-1282.  
[2] Konečný J, McMahan H B, Ramage D, et al. "Federated optimization: Distributed machine learning for on-device intelligence". arXiv preprint arXiv:1610.02527, 2016.  
[3] Yang Q, Liu Y, Cheng Y, et al. "Federated learning". Synthesis Lectures on Artificial Intelligence and Machine Learning, 2019, 13(3): 1-207.
