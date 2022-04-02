---
title: "AlgoZoo"
permalink: /docs/algozoo/
excerpt: "About AlgoZoo."
last_modified_at: 2021-05-11T10:40:42-04:00
toc: true
---

FederatedScope has built in various advanced federated learning algorithms. All of them are implemented as plug-ins, which are detachable and combinable. 

- Detachable: only the activated code will participate in the operation, 
- Combinable: different algorithms can be combined.

In this tutorial, you will learn the buildin algorithms, and how to implement a new federated algorithm in FederatedScope. 

<a name="NA3Sd"></a>
## Buildin Methods
<a name="QTBV1"></a>
### Distributed Optimization Methods
To tackle the challenge of statistical heterogeneity, we implement the following distributed optimization methods: FedAvg (Default), FedProx and FedOpt. Set the parameter `cfg.{METHOD_NAME}.use`as `True` to call them.

<a name="xJfCS"></a>
#### FedAvg
FedAvg [1] is a basic distributed optimization method in federated learning. During federated training, it broadcasts the initialized model to all clients, and aggregates the updated weights collected from several clients. FederatedScope implements it with a fedavg aggregator. More details can be found in `flpackage/core/aggregator.py`.

We provide some evaluation results for fedavg on different tasks as follows. 

| Task | Data | Accuracy(%) |
| --- | --- | --- |
| Logistic regression | Synthetic | 68.36 |
| Image classification | FEMNIST | 84.93 |
| Next-character Prediction | Shakespeare | 43.80 |

To reproduce the results, the running scripts are listed as follows. 
```bash
# logistic regression
python flpackage/main.py --cfg flpackage/nlp/baseline/fedavg_lr_on_synthetic.yaml

# image classification on femnist
python flpackage/main.py --cfg flpackage/cv/baseline/fedavg_convnet2_on_femnist.yaml

# next-character prediction on Shackespeare
python flpackage/main.py --cfg flpackage/nlp/baseline/fedavg_lstm_on_shakespeare.yaml
```
<a name="r0M1W"></a>
#### FedOpt
FedOpt [2] is an advanced distributed optimization method in federated learning. Compare with FedAvg, it permits the server to update weights rather than simply averaging the collected weights. More details can be found in `flpackage/core/aggregator.py`.

Similar with FedAvg, we perform some evaluation of FedAvg on different tasks.

| Task | Data | Learning rate (Server) | Accuracy(%) |
| --- | --- | --- | --- |
| Logistic regression | Synthetic | 0.5 | 68.32 |
| Image classification | FEMNIST | 1.0 | 84.92 |
| Next-character Prediction | Shakespeare | 0.5 | 47.39 |


<a name="VrjwR"></a>
#### FedProx
FedProx [3] is designed to solve the problem of heterogeneity, which updates model with a proximal regularizer. FederatedScope provides build-in FedProx implementation and it can easily be combined with other algorithms. More details can be found in `flpackage/core/trainer/flpackage/core/trainers/trainer_fedprox.py`.

The evaluation results are presented as follows. 

| Task | Data | $\mu$ | Accuracy(%) |
| --- | --- | --- | --- |
| Logistic regression | Synthetic | 0.1 | 68.36 |
| Image classification | FEMNIST | 0.01 | 84.77 |
| Next-character Prediction | Shakespeare | 0.01 | 47.85 |


<a name="gNMeP"></a>
### Personalization Methods

<a name="yppqG"></a>
#### FedBN

[FedBN](https://arxiv.org/abs/2102.07623) [4] is a simple yet effective approach to address feature shift non-iid challenge, in which the client BN parameters are trained locally, without communication and aggregation via server. **FederatedScope** provides simple configuration to implement FedBN and other variants that need to keep parameters of some model sub-modules local.

We provide some evaluation results for FedBN on different tasks as follows, in which the models contain batch normalization. Complete results, config files and running scripts can be found in `scripts/personalization_exp_scripts/fedbn`.

| Task | Data | Accuracy (%) |
| --- | --- | --- |
| Image classification | FEMNIST | 85.48 |
| Graph classification | multi-task-molecule | 72.90 |


<a name="If1FP"></a>
#### pFedMe

[pFedMe](https://arxiv.org/abs/2006.08848) [5]  is an effective pFL approach to address data heterogeneity, in which<br />the personalized model and global model are decoupled with Moreau envelops. **FederatedScope** implements pFedMe in `flpackage/core/trainers/trainer_pFedMe.py` and `ServerClientsInterpolateAggregator` in `flpackage/core/aggregator.py`.

We provide some evaluation results for pFedMe on different tasks as follows. Complete results, config files and running scripts can be found in `scripts/personalization_exp_scripts/pfedme`.

| Task | Data | Accuracy (%) |
| --- | --- | --- |
| Logistic regression | Synthetic | 68.73 |
| Image classification | FEMNIST | 87.65 |
| Next-character Prediction | Shakespeare | 37.40  |


<a name="FvPRi"></a>
#### Ditto

[Ditto](https://arxiv.org/abs/2012.04221) [6] is a SOTA pFL approach that improves fairness and robustness of FL via training local personalized model and global model simultaneously, in which the local model update is based on regularization to global model parameters. **FederatedScope** provides built-in Ditto implementation and users can easily extend to other pFL methods by re-using the model-para regularization. More details can be found in `flpackage/core/trainers/trainer_Ditto.py`.

We provide some evaluation results for Ditto on different tasks as follows. Complete results, config files and running scripts can be found in `scripts/personalization_exp_scripts/ditto`.

| Task | Data | Accuracy (%) |
| --- | --- | --- |
| Logistic regression | Synthetic | 69.67 |
| Image classification | FEMNIST | 86.61 |
| Next-character Prediction | Shakespeare | 45.14 |


<a name="XLiPv"></a>
#### FedEM

[FedEM](https://arxiv.org/abs/2108.10252) [7] is a SOTA pFL approach that assumes local data distribution is a mixture of unknown underlying distributions, and correspondingly learn a mixture of multiple internal models with Expectation-Maximization learning. **FederatedScope** provides built-in FedEM implementation and users can easily extends to other multi-model pFL methods based on this example. More details can be found in `flpackage/core/trainers/trainer_FedEM.py`.

We provide some evaluation results for FedBN on different tasks as follows. Complete results, config files and running scripts can be found in `scripts/personalization_exp_scripts/fedem`.

| Task | Data | Accuracy (%) |
| --- | --- | --- |
| Logistic regression | Synthetic | 68.80 |
| Image classification | FEMNIST | 84.79 |
| Next-character Prediction | Shakespeare | 48.06 |

<a name="QAqz4"></a>
## Implementation
<a name="EDJzS"></a>
### Preliminary
Before implementing a new federated algorithm, you need to realize the structure of [Trainer](https://yuque.antfin.com/gy2g1n/dcpcvz/zwicza) and [Context](https://yuque.antfin.com/gy2g1n/dcpcvz/zwicza). If you already have the knowledge about them, in this part we'll learn how to add an algorithm in FederatedScope.

In FederatedScope, there are three steps to implement a new federated algorithm: 

- Prepare parameters: figure out the parameters required by your algorithm, and fill them into the `Context`
- Prepare hook functions: split your algorithm into several functions according to their insert positions within Trainer/Server,
- Assemble algorithm: create a warp function to assemble your algorithm before create the trainer object. 

<a name="yVJzz"></a>
### Example (Fedprox)
Let's take FedProx as an example to show how to implement a new federated algorithm. 

<a name="M3wrE"></a>
#### Prepare parameters
First, FedProx requires to set proximal regularizer and its factor `ctx.regularizer.mu`. 
```python
# ------------------------------------------------------------------------ #
# Init variables for FedProx algorithm
# ------------------------------------------------------------------------ #
def init_fedprox_ctx(base_trainer):
    ctx = base_trainer.ctx
    cfg = base_trainer.cfg

    cfg.regularizer.type = 'proximal_regularizer'
    cfg.regularizer.mu = cfg.fedprox.mu

    from flpackage.core.auxiliaries.regularizer_builder import get_regularizer
    ctx.regularizer = get_regularizer(cfg.regularizer.type)
```

<a name="ATtAp"></a>
#### Prepare Hook Functions
During training,`FredProx`requires to record the initalized weights before local updating. Therefore, we create two hook functions to maintain the initialized weights. 

- `record_initialization`: record initialized weights, and
- `del_initialization`: delete initialized weights to avoid memory leakage
```python
# ------------------------------------------------------------------------ #
# Additional functions for FedProx algorithm
# ------------------------------------------------------------------------ #
def record_initialization(ctx):
    ctx.weight_init = deepcopy(
        [_.data.detach() for _ in ctx.model.parameters()])


def del_initialization(ctx):
    ctx.weight_init = None
```

<a name="WGCeG"></a>
#### Assemble algorithm
After preparing parameters and hook functions, we assemble FedProx within the function `wrap_fedprox_trainer` in two steps:

- initialize parameters (call function`init_fedprox_ctx`)
- register hook functions for the given trainer
```python
def wrap_fedprox_trainer(
        base_trainer: Type[GeneralTrainer]) -> Type[GeneralTrainer]:
    """Implementation of fedprox refer to `Federated Optimization in Heterogeneous Networks` [Tian Li, et al., 2020]
        (https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf)

    """

    # ---------------- attribute-level plug-in -----------------------
    init_fedprox_ctx(base_trainer)

    # ---------------- action-level plug-in -----------------------
    base_trainer.register_hook_in_train(new_hook=record_initialization,
                                        trigger='on_fit_start',
                                        insert_pos=-1)

    base_trainer.register_hook_in_eval(new_hook=record_initialization,
                                       trigger='on_fit_start',
                                       insert_pos=-1)

    base_trainer.register_hook_in_train(new_hook=del_initialization,
                                        trigger='on_fit_end',
                                        insert_pos=-1)

    base_trainer.register_hook_in_eval(new_hook=del_initialization,
                                       trigger='on_fit_end',
                                       insert_pos=-1)

    return base_trainer
```

Finally, add FedProx into the function `get_trainer`(`flpackage/core/auxiliaries/trainer_builder.py`). 
```python
def get_trainer(model=None,
                data=None,
                device=None,
                config=None,
                only_for_eval=False,
                is_attacker=False):
    
    trainer = ...
    
    # fed algorithm plug-in
    if config.fedprox.use:
        from flpackage.core.trainers.trainer_fedprox import wrap_fedprox_trainer
        trainer = wrap_fedprox_trainer(trainer)
```

<a name="G1lQ7"></a>
## Run an Example
Generally, build-in algorithms are called by setting the parameter `$cfg.{METHOD_NAME}.use` as True. For more infomation about their parameters, you can refer to `flpackage/config.py`. Similarily, taking FedProx as an exmple, its parameters in `flpackage/core/config.py` are
```python
...

# ------------------------------------------------------------------------ #
# fedprox related options
# ------------------------------------------------------------------------ #
cfg.fedprox = CN()

cfg.fedprox.use = True		# Whether to use fedprox
cfg.fedprox.mu = 0. 		# The regularizer factor within fedprox

...
```
You can call FedProx by the following command in the terminal
```bash
python flpackage/main.py --cfg {YOUR_CONFIG_FILE} fedprox.use True fedprox.mu 0.1
```
More example scripts are refer to`flpackage/example_configs/.`

<a name="rL9rC"></a>
### Note
Most combinations of the buildin methods have been tested. When implementing your own methods, it is suggested to carefully check the code to avoid conflicts (e.g. duplication of variables).

---

<a name="Gy7PK"></a>
# References
[1] Brendan McMahan, Eider Moore, Daniel Ramage, et al. Communication-Efficient Learning of Deep Networks from Decentralized Data. Proceedings of the International Conference on Artificial Intelligence and Statistics, 2017. <br />[2] Sashank J. Reddi, Zachary Charles, Manzil Zaheer, et al. Adaptive Federated Optimization. Proceedings of Intertional Conference on Learning Representations, 2021.<br />[3] Tian Li, Anit Kumar Sahu, Manzil Zaheer, et al. Federated Optimization in Heterogeneous Networks. Proceedings of Machine Learning and Systems, 2020. <br />[4] Li, Xiaoxiao, et al. "Fedbn: Federated learning on non-iid features via local batch normalization." arXiv preprint arXiv:2102.07623 (2021).<br />[5] T Dinh, Canh, Nguyen Tran, and Josh Nguyen. "Personalized federated learning with moreau envelopes." Advances in Neural Information Processing Systems 33 (2020): 21394-21405.<br />[6] Li, Tian, et al. "Ditto: Fair and robust federated learning through personalization." International Conference on Machine Learning. PMLR, 2021.<br />[7] Marfoq, Othmane, et al. "Federated multi-task learning under a mixture of distributions." Advances in Neural Information Processing Systems 34 (2021).<br /> 

