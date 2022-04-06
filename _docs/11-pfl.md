---
title: "Personalized FL"
permalink: /docs/pfl/
excerpt: "About personalized FL."
last_modified_at: 2018-03-20T15:59:57-04:00
toc: true
layout: tuto
---

FederatedScope  is a flexible FL framework, which enables users to implement complex FL algorithms simply and intuitively. In this tutorial, we will show how to implement diverse personalized FL algorithms.

## Background

In an FL course, multiple clients aim to cooperatively learn models without directly sharing their private data. As a result, these clients can be arbitrarily different in terms of their underlying **data distribution** and **system resources** such as computational power and communication width.

- On one hand, the data quantity skew, feature distribution skew,  label distribution skew, and temporal skew are pervasive in real-world applications as different users generate the data with different usage manners.
  Simply applying the shared global model for all participants might lead to sub-optimal performance.
- On the other hand, the participation degrees of different FL participants can be diverse due to their different hardware capabilities and network conditions.

It is challenging to make full use of local data considering such systematical heterogeneity. As a natural and effective approach to address these challenges, personalization gains increasing attention in recent years. Personalized FL (pFL) raises strong demand for various customized FL implementation, e.g., the personalization may exist in

- Model objects, optimizers and hyper-parameters
- Model sub-modules
- Client-end behaviors such as regularization and multi-model interaction
- Server-end behaviors such as model interpolation

We will demonstrate several implementations for state-of-the-art (SOTA) pFL methods  to meet the above requirements and show how  powerful and flexible the FederatedScope framework to implement pFL extensions.



## Demonstration

### Personalized model sub-modules - FedBN

[FedBN](https://arxiv.org/abs/2102.07623) [1] is a simple yet effective approach to address feature shift non-iid, in which the client BN parameters are trained locally, without communication and aggregation via server. FederatedScope provides simple configuration to implement FedBN and other variants that need to keep parameters of some model sub-modules local.

- By specifying the local parameter names as follows, the clients and server will filter out the sub-modules contains the given names in the model parameter `update` function.
  ```python
  cfg.personalization.local_param = [] 
  # e.g., ['pre', 'post', 'bn']
  ```
- We provide auxiliary logging function `print_trainer_meta_info()` to show the model type, local and filtered model parameter names in trainer instantiation


```python
# trainer.print_trainer_meta_info()

Model meta-info: <class 'federatedscope.cv.model.cnn.ConvNet2'>.
Num of original para names: 18.
Num of original trainable para names: 12.
Num of preserved para names in local update: 8.
Preserved para names in local update: {'fc2.bias', 'conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight', 'conv1.bias', 'fc1.bias', 'conv2.bias'}.
Num of filtered para names in local update: 10.
Filtered para names in local update: {'bn2.weight', 'bn2.num_batches_tracked', 'bn1.num_batches_tracked', 'bn1.running_var', 'bn2.running_mean', 'bn1.weight', 'bn2.running_var', 'bn1.running_mean', 'bn1.bias', 'bn2.bias'}.
```

### Personalized regularization - Ditto

[Ditto](https://arxiv.org/abs/2012.04221) [2] is a SOTA pFL approach that improves fairness and robustness of FL via training local personalized model and global model simultaneously, in which the local model update is based on regularization to global model parameters. FederatedScope provides built-in Ditto implementation and users can easily extends to other pFL methods by re-using the model-para regularization. More details can be found in `federatedscope/core/trainers/trainer_Ditto.py`.

- To preserve distinct local models in trainer, we can simply use another model object in trainer's context

  ```
  ctx.local_model = copy.deepcopy(ctx.model)  # the personalized model
  ctx.global_model = ctx.model
  ```

- To train local models with global-model regularization, we implement a new hook on  run_routine fit start and register the global model parameters into the new optimizer.

  ```
  def hook_on_fit_start_set_regularized_para(ctx):
      # set the compared model data for local personalized model
      ctx.global_model.to(ctx.device)
      ctx.local_model.to(ctx.device)
      ctx.global_model.train()
      ctx.local_model.train()
      compared_global_model_para = [{
          "params": list(ctx.global_model.parameters())
      }]
      ctx.optimizer_for_local_model.set_compared_para_group(
          compared_global_model_para)
  
  def regularize_by_para_diff(self):
      """
         before optim.step(), regularize the gradients based on para diff
      """
      for group, compared_group in zip(self.param_groups, self.compared_para_groups):
          for p, compared_weight in zip(group['params'], compared_group['params']):
              if p.grad is not None:
                 if compared_weight.device != p.device:
                      compared_weight = compared_weight.to(p.device)
                      p.grad.data = p.grad.data + self.regular_weight * (p.data - compared_weight.data)
  ```

- We implement Ditto with a pluggable manner, some Ditto specific attributes (*contexts*) and behaviors (*hooks*) can be added into an existing `base_trainer` as follows.

  ```python
  def wrap_DittoTrainer(
          base_trainer: Type[GeneralTrainer]) -> Type[GeneralTrainer]):
    
      # ---------------- attribute-level plug-in -----------------------
      init_Ditto_ctx(base_trainer)
  
      # ---------------- action-level plug-in -----------------------
      base_trainer.register_hook_in_train(
          new_hook=hook_on_fit_start_set_regularized_para,
          trigger="on_fit_start",
          insert_pos=0)
  ```

### Personalized multi-model interaction - FedEM

[FedEM](https://arxiv.org/abs/2108.10252) [3] is a SOTA pFL approach that assumes local data distribution is a mixture of unknown underlying distributions, and correspondingly learn a mixture of multiple internal models with Expectation-Maximization learning. FederatedScope provides built-in FedEM implementation and users can easily extends to other multi-model pFL methods based on this example. More details can be found in `federatedscope/core/trainers/trainer_FedEM.py`.

- The `FedEMTrainer` is derived from `GeneralMultiModelTrainer`. We can easily add FedEM-specific attributes and behaviors via context and hooks register functions

  ```python
  # ---------------- attribute-level modifications -----------------------
  # used to mixture the internal models
  self.weights_internal_models = (torch.ones(self.model_nums) /
                                  self.model_nums).to(device)
  self.weights_data_sample = (
      torch.ones(self.model_nums, self.ctx.num_train_batch) /
      self.model_nums).to(device)
  
  self.ctx.all_losses_model_batch = torch.zeros(
      self.model_nums, self.ctx.num_train_batch).to(device)
  self.ctx.cur_batch_idx = -1
  
  # ---------------- action-level modifications -----------------------
  # see customized register_multiple_model_hooks(), which is called in the __init__ of `GeneralMultiModelTrainer`
  ```

- We can simply extend  `GeneralMultiModelTrainer` with the default sequential interaction mode, and add some training behaviors such as `mixture_weights_update`, `weighted_loss_adjustment` and `track_batch_idx`

  ```python
  # hooks example, for only train
  def hook_on_batch_forward_weighted_loss(self, ctx):
      ctx.loss_batch *= self.weights_internal_models[ctx.cur_model_idx]
  
  def register_multiple_model_hooks(self):
      # First register hooks for model 0
      # ---------------- train hooks -----------------------
      self.register_hook_in_train(
          new_hook=self.hook_on_fit_start_mixture_weights_update,
          trigger="on_fit_start",
          insert_pos=0)  # insert at the front
      self.register_hook_in_train(
          new_hook=self.hook_on_batch_forward_weighted_loss,
          trigger="on_batch_forward",
          insert_pos=-1)
      self.register_hook_in_train(
          new_hook=self.hook_on_batch_start_track_batch_idx,
          trigger="on_batch_start",
          insert_pos=0)  # insert at the front
  ```

- We also need to add some evaluation behavior modifications such as `model_ensemble` and `loss_gather`

  ```python
      # ---------------- eval hooks -----------------------
      self.register_hook_in_eval(
          new_hook=self.hook_on_batch_end_gather_loss,
          trigger="on_batch_end",
          insert_pos=0
      )  # insert at the front, (we need gather the loss before clean it)
      self.register_hook_in_eval(
          new_hook=self.hook_on_batch_start_track_batch_idx,
          trigger="on_batch_start",
          insert_pos=0)  # insert at the front
      # replace the original evaluation into the ensemble one
      self.replace_hook_in_eval(
          new_hook=self._hook_on_fit_end_ensemble_eval,
          target_trigger="on_fit_end",
          target_hook_name="_hook_on_fit_end")
      
  # hooks example, for only eval
  def hook_on_batch_end_gather_loss(self, ctx):
      # before clean the loss_batch; we record it for further weights_data_sample update
      ctx.all_losses_model_batch[ctx.cur_model_idx][
              ctx.cur_batch_idx] = ctx.loss_batch.item()
  ```

- Note that the `GeneralMultiModelTrainer` will switch the model states automatically, we can differentiate different internal models in the new hooks with `ctx.cur_model_idx` and ` self.model_nums` attributes.

FedEM can be generalized to many **clustering ** based methods &  **multi-task modeling** based methods (see details inSection 2.3 in [3]) and we can extend `FedEMTrainer` to more multi-model based pFL methods.

## Evaluation Results
To facilitate rapid and reproducible pFL research, we provide the experimental results and corresponding scripts to benchmark  pFL performance for several SOTA pFL methods via FederatedScope.  We will continue to add more algorithm implementations and experimental results in different scenarios.

### FedBN

We provide some evaluation results for FedBN on different tasks as follows, in which the models contain batch normalization. Complete results, config files and running scripts can be found in `scripts/personalization_exp_scripts/fedbn`.


| Task                      | Data        | Accuracy (%) |
| ------------------------- | ----------- | ------------ |
| Image classification      | FEMNIST     | 85.48 |
| Graph classification      | multi-task-molecule   |  72.90   |


### pFedMe

[pFedMe](https://arxiv.org/abs/2006.08848) [4] is an effective pFL approach to address data heterogeneity, in which
the personalized model and global model are decoupled with Moreau envelops. FederatedScope implements pFedMe in `federatedscope/core/trainers/trainer_pFedMe.py` and `ServerClientsInterpolateAggregator` in `federatedscope/core/aggregator.py`.

We provide some evaluation results for pFedMe on different tasks as follows. Complete results, config files and running scripts can be found in `scripts/personalization_exp_scripts/pfedme`.

| Task                      | Data        | Accuracy (%)            |
| ------------------------- | ----------- | ----------------------- |
| Logistic regression       | Synthetic   | 68.73                   |
| Image classification      | FEMNIST     | 87.65                   |
| Next-character Prediction | Shakespeare | 37.40 |


### Ditto

We provide some evaluation results for Ditto on different tasks as follows. Complete results, config files and running scripts can be found in `scripts/personalization_exp_scripts/ditto`.


| Task                      | Data        | Accuracy (%) |
| ------------------------- | ----------- | ------------ |
| Logistic regression       | Synthetic   | 69.67        |
| Image classification      | FEMNIST     | 86.61        |
| Next-character Prediction | Shakespeare |   45.14   |
### FedEM

[FedEM](https://arxiv.org/abs/2108.10252) is a SOTA pFL approach that assumes local data distribution is a mixture of unknown underlying distributions, and correspondingly learn a mixture of multiple internal models with Expectation-Maximization learning. FederatedScope provides built-in FedEM implementation and users can easily extends to other multi-model pFL methods based on this example. More details can be found in `federatedscope/core/trainers/trainer_FedEM.py`.

We provide some evaluation results for FedBN on different tasks as follows. Complete results, config files and running scripts can be found in `scripts/personalization_exp_scripts/fedem`.

| Task                      | Data        | Accuracy (%) |
| ------------------------- | ----------- | ------------ |
| Logistic regression       | Synthetic   | 68.80        |
| Image classification      | FEMNIST     | 84.79             |
| Next-character Prediction | Shakespeare |       48.06    |

## Reference
[1] Li, Xiaoxiao, et al. "Fedbn: Federated learning on non-iid features via local batch normalization." arXiv preprint arXiv:2102.07623 (2021).

[2] Li, Tian, et al. "Ditto: Fair and robust federated learning through personalization." International Conference on Machine Learning. PMLR, 2021.

[3] Marfoq, Othmane, et al. "Federated multi-task learning under a mixture of distributions." Advances in Neural Information Processing Systems 34 (2021).

[4] T Dinh, Canh, Nguyen Tran, and Josh Nguyen. "Personalized federated learning with moreau envelopes." Advances in Neural Information Processing Systems 33 (2020): 21394-21405.
