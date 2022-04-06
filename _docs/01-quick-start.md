---
title: "Quick Start"
permalink: /docs/quick-start/
excerpt: "Federated Learning (FL) is a learning paradigm for collaboratively learning models from isolated data without directly sharing privacy information, which helps to satisfy the requirements of privacy protection of the public."
last_modified_at: 2021-06-07T08:48:05-04:00
redirect_from:
  - /theme-setup/
toc: true
layout: tuto
---

Federated Learning (FL) [1,2,3] is a learning paradigm for collaboratively learning models from isolated data without directly sharing privacy information, which helps to satisfy the requirements of privacy protection of the public. FederatedScope, a comprehensive federated learning platform via message passing, aims to provide easy-to-use and flexible support for users who want to quickly get started and customize task-specific FL procedures.  

We first provide an end-to-end example to illustrate how to implement a standard FL course with FederatedScope.

## Clone source code and install the environment

```bash
git clone https://github.com/alibaba/FederatedScope.git
cd FederatedScope
python setup.py install
```

## Prepare datasets & models

To run an FL course, firstly you should prepare datasets for FL. The [DataZoo]({{ "/docs/datazoo/" | relative_url }}) provided in FederatedScope can help to automatically download and preprocess widely-used public datasets from various FL applications, including computer vision, natural language processing, graph learning, recommendation, etc. Users can conveniently conduct experiments on the provided dataset via specifying `cfg.data.type = DATASET_NAME`in the configuration.  
We also support users to adopt customized datasets, please refer to [DataZoo]({{ "/docs/datazoo/" | relative_url }}) for more details about the provided datasets, and refer to [Customized Datasets]({{ "/docs/own-case/#load-a-dataset" | relative_url }}) for introducing customized datasets in FederatedScope.

Secondly, you should specify the model architecture that will be federally trained, such as ConvNet or LSTM. FederatedScope includes the ModelZoo to provide the implementation of famous model architectures for various FL applications. Users can set up `cfg.model.type = MODEL_NAME` to apply a specific model architecture in FL tasks. We allow users to use custom models via registering without caring about the federated process. You can refer to [DataZoo]({{ "/docs/datazoo/" | relative_url }}) for more details about how to customize models.

For a standard FL course, all participants share the same model architecture and training configuration. And FederatedScope also supports adopting client-specific models and training configurations (known as personalized FL) to handle the non-IID issue in practical  FL applications, please refer to [Personalized FL]({{ "/docs/pdf/" | relative_url }}) for more details. 

## Run an FL task with basic configurations.

Note that FederatedScope provides a unified interface for both standalone simulation and distributed deployment, therefore users can easily run an FL course in standalone mode or distributed mode via configuring. 

### Standalone mode

The standalone mode in FederatedScope means that we simulate multiple participants (servers and clients) in a single device, where the data and models of each participant are isolated from each other and only be shared via message passing. The standalone mode is often used for quickly and efficiently verifying the correctness and effectiveness of the proposed FL algorithms.

Here we demonstrate how to run a standard FL course with FederatedScope as an example, with setting `cfg.data.type = 'FEMNIST'`and `cfg.model.type = 'ConvNet2'` to run vanilla FedAvg [1] for an image classification task.
Users can include more training configurations, such as `cfg.federated.total_round_num`, `cfg.data.batch_size`, and `cfg.optimizer.lr`, in the configuration (a .yaml file), and run a standard FL course as: 

```python
# Run with default configurations
python federatedscope/main.py --cfg federatedscope/example_configs/femnist.yaml
# Or with custom configurations
python federatedscope/main.py --cfg federatedscope/example_configs/femnist.yaml federated.total_round_num 50 data.batch_size 128
```

Then users can observe some monitored metrics during the training process as:

```python
INFO: Server #0 has been set up ...
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
INFO: Server #0: Training is finished! Starting evaluation.
INFO: Client #1: (Evaluation (test set) at Round #20) test_loss is 163.029045
... ...
INFO: Server #0: Final evaluation is finished! Starting merging results.
... ...
```

### Distributed mode

The distributed mode in FederatedScope denotes running multiple procedures to build up an FL course, where each procedure plays as a participant (a server or a client) that instantiates its model and loads its data. The communication between participants relies on the computer network and is already provided by the communication module of FederatedScope.

FederatedScope provides a unified interface for both standalone simulation and distributed deployment. Therefore, for running with distributed mode, users only need to:

- Prepare isolated data file and set up `cfg.distribute.data_file = PATH/TO/DATA` for each participant;
- Change `cfg.federate.model = 'distributed'`, and specify the role of each participant  by `cfg.distributed.role = 'server' or 'client'`.
- Set up a valid address by `cfg.distribute.host = x.x.x.x` and `cfg.distribute.host = xxxx`. (Note that for a server, users need to set up server_host/server_port for listening messge, while for a client, users need to set up client_host/client_port for listening and server_host/server_port for  sending join-in applications when building up an FL course)

We prepare a synthetic example for running with distributed mode:

```python
 # For server
 python main.py --cfg federatedscope/example_configs/distributed_server.yaml data_path 'PATH/TO/DATA' server.host x.x.x.x client.port xxxx
 
 # For client
 python main.py --cfg federatedscope/example_configs/distributed_client.yaml data_path 'PATH/TO/DATA' server.host x.x.x.x server.port xxxx client.host x.x.x.x client.port xxxx
```

And you can observe the results as (the IP addresses are replaced with 'x.x.x.x'):

```python
INFO: Server #0: Listen to x.x.x.x:xxxx...
INFO: Server #0 has been set up ...
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
INFO: Server #0: Training is finished! Starting evaluation.
INFO: Client #1: (Evaluation (test set) at Round #20) test_loss is 30.387419
... ...
INFO: Server #0: Final evaluation is finished! Starting merging results.
... ...
```

## References

[1] McMahan B, Moore E, Ramage D, et al. "Communication-efficient learning of deep networks from decentralized data". Artificial intelligence and statistics. PMLR, 2017: 1273-1282.  
[2] Konečný J, McMahan H B, Ramage D, et al. "Federated optimization: Distributed machine learning for on-device intelligence". arXiv preprint arXiv:1610.02527, 2016.  
[3] Yang Q, Liu Y, Cheng Y, et al. "Federated learning". Synthesis Lectures on Artificial Intelligence and Machine Learning, 2019, 13(3): 1-207.  
