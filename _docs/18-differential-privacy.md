---
title: "Differential Privacy"
permalink: /docs/dp/
excerpt: "About Differential Privacy."
last_modified_at: 2016-11-03T11:13:12-04:00
toc: true
---

<a name="VfJFz"></a>
## Background
Differential privacy (DP) [1] is a powerful theoretical criterion metric for privacy preserving in database. Specifically, a randomized mechanism satisfying $(\epsilon-\delta)$-DP promises the privacy loss of all neighboring datasets is bounded by $\epsilon$ with the probability at least $1-\delta$[1].

**Differential Privacy**: <br />A randomized algorithm $\mathcal{M}$satisifies $(\epsilon-\delta)$-DP if for all $S\subseteq{Range(\mathcal{M})}$and all neigbhoring datasets $x,y$$(||x-y||_1\leq1)$:<br />$Pr(\mathcal{M}(x)\in{S})\leq{exp(\epsilon)Pr(\mathcal{M}(y)\in{S})}+\delta$

<a name="VZ5KN"></a>
## Support of DP
In federated learning scenario, noise injection and gradient clipping are foundational tools for differential privcay. In FederatedScope, DP algorithms are supported by preseting APIs in the core library, including 

- noise injection in download channel (server)
- noise injection in unload channel (client)
- gradient clipping before upload

<a name="kW3c6"></a>
### Noise Injection in Download
In server, a protected attribute `_noise_injector` is preset in the server class for noise injection with default value is `None`. 
```python
class Server(Worker):
    def __init__(**kwargs):
        ...
        # inject noise before broadcast
        self._noise_injector = None
    
    ...
```
Developers can achieve noise injection by calling `register_noise_injector`. Then the function `self._noise_injector`will be called before the server broadcasts parameters.  
```python
class Server(Worker):
    ...
    def register_noise_injector(self, func):
        self._noise_injector = func
    ...
```

<a name="ORt3o"></a>
### Noise Injection in Upload
For clients, the noise injection can be done by registering hook function at the end of training (before uploading parameters). 
```python
def noise_injection_function(ctx):
    pass

trainer.register_hook_in_train(new_hook=noise_injection_function,
                              trigger='on_fit_end',
                              insert_pos=-1)

```

<a name="fEDnf"></a>
### Gradient Clipping
Gradient clipping is preset in `flapackage/core/trainers/trainer.py`. When the function `_hook_on_batch_backward`is called, the gradient will be clipped by the parameter `cfg.optimizer.grad_clip`in the config.
```python
    ...
    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()
    ...
```
Threshold of gradient clipping in `flpackage/config.py`.
```python
# ------------------------------------------------------------------------ #
# Optimizer related options
# ------------------------------------------------------------------------ #
cfg.optimizer = CN()

cfg.optimizer.type = 'SGD'
cfg.optimizer.lr = 0.1
cfg.optimizer.weight_decay = .0
cfg.optimizer.grad_clip = -1.0  # negative numbers indicate we do not clip grad
```
<a name="HWuqz"></a>
# <br />
<a name="AOJIY"></a>
## Implementation of DP
NbAFL [2] is a DP algorithm designed for federated learning, which protects both the upload and download channels with $(\epsilon-\delta)$-DP. Taking NbAFL as an example, we show how to implement DP algorithm in FederatedScope. 

<a name="Ehipa"></a>
### Prepare DP Parameters
Add parameters into `flpackage/config.py`. Note FederatedScope supports at most two levels of config, e.g., `cfg.data.type`. 
```python
# ------------------------------------------------------------------------ #
# nbafl(dp) related options
# ------------------------------------------------------------------------ #
cfg.nbafl = CN()

# Params
cfg.nbafl.use = False
cfg.nbafl.mu = 0.
cfg.nbafl.epsilon = 100.
cfg.nbafl.w_clip = 1.
cfg.nbafl.constant = 30.
```

<a name="HjxGS"></a>
### Prepare DP Functions
Then developers should design their own DP functions. For NbAFL, the following three hook functions are required for client`Trainer`.  The functions`record_initialization`and `del_initialization`maintain the initialization received from the server, and `inject_noise_in_upload`injects noise into the model before upload.
```python
def record_initialization(ctx):
    ctx.weight_init = deepcopy(
        [_.data.detach() for _ in ctx.model.parameters()])


def del_initialization(ctx):
    ctx.weight_init = None


def inject_noise_in_upload(ctx):
    """Inject noise into weights before the client upload them to server

    """
    scale_u = ctx.nbafl_w_clip * ctx.nbafl_total_round_num * 2 * ctx.nbafl_constant / ctx.num_train_data / ctx.nbafl_epsilon
    # logging.info({"Role": "Client", "Noise": {"mean": 0, "scale": scale_u}})
    for p in ctx.model.parameters():
        noise = get_random("Normal", p.shape, {
            "loc": 0,
            "scale": scale_u
        }, p.device)
        p.data += noise
```

For the server, the following function is created for noise injection. 
```python
def inject_noise_in_broadcast(cfg, sample_client_num, model):
    """Inject noise into weights before the server broadcasts them

    """
    if len(sample_client_num) == 0:
        return

    # Clip weight
    for p in model.parameters():
        p.data = p.data / torch.max(torch.ones(size=p.shape),
                                    torch.abs(p.data) / cfg.nbafl.w_clip)

    if len(sample_client_num) > 0:
        # Inject noise
        L = cfg.federate.sample_client_num if cfg.federate.sample_client_num > 0 else cfg.federate.client_num
        if cfg.federate.total_round_num > np.sqrt(cfg.federate.client_num) * L:
            scale_d = 2 * cfg.nbafl.w_clip * cfg.nbafl.constant * np.sqrt(
                np.power(cfg.federate.total_round_num, 2) -
                np.power(L, 2) * cfg.federate.client_num) / (
                    min(sample_client_num.values()) * cfg.federate.client_num *
                    cfg.nbafl.epsilon)
            for p in model.parameters():
                p.data += get_random("Normal", p.shape, {
                    "loc": 0,
                    "scale": scale_d
                }, p.device)


```
<a name="w5vs3"></a>
## 
<a name="eOmOx"></a>
### Register DP Functions
The wrap function `wrap_nbafl_trainer` initializes parameters related to NbAFL and registers the above hook functions. 
```python
def wrap_nbafl_trainer(
        base_trainer: Type[GeneralTrainer]) -> Type[GeneralTrainer]:
    """Implementation of NbAFL refer to `Federated Learning with Differential Privacy: Algorithms and Performance Analysis` [et al., 2020]
        (https://ieeexplore.ieee.org/abstract/document/9069945/)

        Arguments:
            mu: the factor of the regularizer
            epsilon: the distinguishable bound
            w_clip: the threshold to clip weights

    """

    # ---------------- attribute-level plug-in -----------------------
    init_nbafl_ctx(base_trainer)

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

    base_trainer.register_hook_in_train(new_hook=inject_noise_in_upload,
                                        trigger='on_fit_end',
                                        insert_pos=-1)
    return base_trainer
```
Finally, in `flpackage/core/auxiliaries/trainer_builder.py`, the function `get_trainer`wraps the basic trainer with NbAFL variables and functions. 
```python
def get_trainer(model=None,
                data=None,
                device=None,
                config=None,
                only_for_eval=False,
                is_attacker=False):
    ...
    # differential privacy plug-in
    if config.nbafl.use:
        from flpackage.core.trainers.trainer_nbafl import wrap_nbafl_trainer
        trainer = wrap_nbafl_trainer(trainer)
    ...
```

<a name="KD2IU"></a>
## Run an Example
Run the  following command to call NbAFL on the dataset Femnist.
```bash
python flpackage/main.py --cfg flpackage/cv/baseline/fedavg_convnet2_on_femnist.yaml \
  nbafl.mu 0.01 \
  nbafl.constant 1 \
  nbafl.w_clip 0.1 \
  nbafl.epsilon 10
```

<a name="bNCuG"></a>
## Evaluation
Take the dataset Femnist as an example, the accuracy with different $(\epsilon-\delta)$-DP is shown as follows. 

| Task | $\epsilon$ | $\delta$ | Accuracy(%) |
| --- | --- | --- | --- |
| FEMNIST | 10 | 0.01 | 11.73 |
|  |  | 0.17 | 24.82 |
|  |  | 0.76 | 41.71 |
|  | 50 | 0.01 | 54.85 |
|  |  | 0.17 | 67.98 |
|  |  | 0.76 | 80.58 |
|  | 100 | 0.01 | 74.80 |
|  |  | 0.17 | 80.39 |
|  |  | 0.76 | 80.58 |


---

<a name="RpfyB"></a>
## Reference
[1] Cynthia Dwork, Aaron Roth. The Algorithmic Foundations of Differential Privacy. Foundations and Trends in Theoretical Computer Science. <br />[2] Kang Wei, Jun Li, Ming Ding, et al. Federated Learning With Differential Privacy: Algorithms and Performance Analysis. IEEE Transactions on Information Forensics and Security. 
