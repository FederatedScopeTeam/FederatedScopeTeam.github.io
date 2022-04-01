---
title: "Local Learning Abstraction: Trainer"
permalink: /docs/trainer/
excerpt: "About trainer."
last_modified_at: 2020-08-30T21:27:40-04:00
toc: true
---

FederatedScope decouples the local learning process and details of FL communication and schedule, allowing users to freely customize local learning algorithm via the `trainer`. Each worker holds a `trainer` object to manage the details of local learning, such as the loss function, optimizer, training step, evaluation, etc. 

In this tutorial, you will learn:

- The structure of `Trainer` used in FederatedScope;
- How the `Trainer` maintains attributes and how to extend new attributes?
- How the `Trainer` maintains learning behaviors and how to extend new behaviors?
- How to extend  `Trainer` to learn with more than one internal model?

## `Trainer` Structure
A typical machine learning process consists of the following procedures:

1. Preparing datasets and pre-extracting data mini-batches
2. Iterations over training datasets to update the model parameters
3. Evaluation the quality of learned model on validation/evaluation datasets
4. Saving, loading, and monitoring the model and intermediate results

![undefined](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/226570/1647846107018-85df1a74-b53f-4b21-98d4-1d4eb15c3655.png) 

As the figure shows, in FederatedScope `Trainer`,  these above procedures are provided with high-level `routines` abstraction, which are made up of `Context` class and several pluggable `Hooks`.

- The `Context` class is used to holds learning-related attributes, including data, model, optimizer and etc. We will introduce more details in [next Section](#trainer-context).
```python
self.ctx = Context(model,
                   self.cfg,
                   data,
                   device,
                   init_dict=self.parse_data(data))
```
- The `Hooks` represent fine-grained learning behaviors at different point-in-times, which provides a simple yet powerful way to customize learning behaviors with a few modifications and easy re-use of fruitful default hooks. More details about the behavior customization are in [following Section](#trainer-behaviors).
```python
HOOK_TRIGGER = [
        "on_fit_start", "on_epoch_start", "on_batch_start", "on_batch_forward",
        "on_batch_backward", "on_batch_end", "on_epoch_end", "on_fit_end"
    ]
self.hooks_in_train = collections.defaultdict(list)
# By default, use the same trigger keys
self.hooks_in_eval = copy.deepcopy(self.hooks_in_train)
if not only_for_eval:
    self.register_default_hooks_train()
self.register_default_hooks_eval()
```

## Trainer Behaviors

### Routines

- Besides the common I/O procedures `save_model` and `load_model`, FederatedScope trainer uses the `update` function to load the model from FL clients.

- For the train/eval/validate procedures, FederatedScope implements them via calling a general `_run_routine` with different datasets, hooks_set and running mode.

  ```
  def _run_routine(self, mode, hooks_set, dataset_name=None)
  ```

  - We decouple the learning process with several fine-grained point-in-time and calling all registered hooks at specific point-in-times as follows

    ```python
    for hook in hooks_set["on_fit_start"]:
        hook(self.ctx)
    
    for epoch_i in range(self.ctx.get(
            "num_{}_epoch".format(dataset_name))):
        self.ctx.cur_epoch_i = epoch_i
        for hook in hooks_set["on_epoch_start"]:
            hook(self.ctx)
    
        for batch_i in range(
                self.ctx.get("num_{}_batch".format(dataset_name))):
            self.ctx.cur_batch_i = batch_i
            for hook in hooks_set["on_batch_start"]:
                hook(self.ctx)
            for hook in hooks_set["on_batch_forward"]:
                hook(self.ctx)
            if self.ctx.cur_mode == 'train':
                for hook in hooks_set["on_batch_backward"]:
                    hook(self.ctx)
            for hook in hooks_set["on_batch_end"]:
                hook(self.ctx)
    
            # Break in the final epoch
            if self.ctx.cur_mode == 'train' and epoch_i == self.ctx.num_train_epoch - 1:
                if batch_i >= self.ctx.num_train_batch_last_epoch - 1:
                    break
    
        for hook in hooks_set["on_epoch_end"]:
            hook(self.ctx)
    for hook in hooks_set["on_fit_end"]:
        hook(self.ctx)
    ```

### Hooks 
  - We implement fruitful default hooks to support various training/evaluation processes, such as [personalized FL behaviors](), [graph-task related behaviors](), [privacy-preserving behaviors](). 

  - Each hook takes the learning `context` as input and performs the learning actions such as 

      - prepare model and statistics
      
    ```python
    def _hook_on_fit_start_init(ctx):
        # prepare model
        ctx.model.to(ctx.device)
    
        # prepare statistics
        setattr(ctx, "loss_batch_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "loss_regular_total_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "num_samples_{}".format(ctx.cur_data_split), 0)
        setattr(ctx, "{}_y_true".format(ctx.cur_data_split), [])
        setattr(ctx, "{}_y_prob".format(ctx.cur_data_split), [])
    
    ```

    - calculate loss in forward stage
    
    ```python
    def _hook_on_batch_forward(ctx):
        x, label = [_.to(ctx.device) for _ in ctx.data_batch]
        pred = ctx.model(x)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch = ctx.criterion(pred, label)
        ctx.y_true = label
        ctx.y_prob = pred
    
        ctx.batch_size = len(label)
    ```
    
     - update model parameters in backward stage
    
    ```python
    def _hook_on_batch_backward(ctx):
        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
              ctx.model.parameters(), ctx.grad_clip)
        ctx.optimizer.step()
    ```
    
- To customize more trainer behaviors, users can reset and replace existing hooks, or register new hooks

  - Users can freely check the current hook set via print `trainer.hooks_in_train` and `trainer.hooks_in_eval`.

  - For delete case, users can either 1) reset all the hooks at a target point-in-time trigger; or 2) a specific hook by passing the target function name `hook_name `  in  train/eval hook set.

    ```python 
    def reset_hook_in_train(self, target_trigger, target_hook_name=None)
    def reset_hook_in_eval(self, target_trigger, target_hook_name=None)

  - For create case,  we allows registering a new hook at a target point-in-time trigger, and support 1) specifying a specific  positions (i.e., the order a hook called within the trigger set); or 2) inserting before or after a base hook

  ```python
  def register_hook_in_train(self,
                             new_hook,
                             trigger,
                             insert_pos=None,
                             base_hook=None,
                             insert_mode="before")
  ```

  - For update case, we provide functions to replace existing hook (by name) with a new_hook (function)

    ```python
    def replace_hook_in_train(self, new_hook, target_trigger, target_hook_name)
    ```

### Customized Data Preparation
- We provide the data pre-processing operations in `parse_data` function, which parses the dataset and initializes the variables `{}_data`, `{}_loader`,  and the counter `num_{MODE}_data` according to the types of datasets within `data` as follows. 

```python
    def parse_data(self, data):
        """Populate "{}_data", "{}_loader" and "num_{}_data" for different modes

        """
        # TODO: more robust for different data
        init_dict = dict()
        if isinstance(data, dict):
            for mode in ["train", "val", "test"]:
                init_dict["{}_data".format(mode)] = None
                init_dict["{}_loader".format(mode)] = None
                init_dict["num_{}_data".format(mode)] = 0
                if data.get(mode, None) is not None:
                    if isinstance(data.get(mode), Dataset):
                        init_dict["{}_data".format(mode)] = data.get(mode)
                        init_dict["num_{}_data".format(mode)] = len(
                            data.get(mode))
                    elif isinstance(data.get(mode), DataLoader):
                        init_dict["{}_loader".format(mode)] = data.get(mode)
                        init_dict["num_{}_data".format(mode)] = len(
                            data.get(mode).dataset)
                    elif isinstance(data.get(mode), dict):
                        init_dict["{}_data".format(mode)] = data.get(mode)
                        init_dict["num_{}_data".format(mode)] = len(
                            data.get(mode)['y'])
                    else:
                        raise TypeError("Type {} is not supported.".format(
                            type(data.get(mode))))
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict
```
- To support customized dataset, please implement the function `parse_data` in the new trainer and initialize the following variables.
	- `{train/test/val}_data`: the data object,
	- `{train/test/val}_loader`: the data loader object, 
	- `num_{train/test/val}_data`: the number of samples within the dataset.

## Trainer Context

`Context` class is an implementation of messager within the trainer. All variables within it can be called by `ctx.{VARIABLE_NAME}`. 

As stated above, both the training and evluation processes are consisted of independent hook functions, which only receive an instance of `Context` as the sole parameter. Therefore, the parameter `ctx` should 
- maintain the references of objects (e.g. model, data, optimizer), 
- provide running parameters (e.g. number of training epochs), 
- indicate the current operating status (e.g. train/test/validate), and 
- record statistical variables (e.g. loss, output, accuracy). 

### Attributes
To satisfy the above requirements, an instance of `Context` contains two types of variables: 
- Static variables: provide the basic references, and most of the time remain unchanged. 
	- The reference of running model, dataset, dataloader, optimizer, criterion function, regularizer, and so on.
	
	  ```python
	  NAME		TYPE		MEANING
	  model		Module		Reference to the model
	  data 		Dict		A dict contains train/val/test dataset or dataloader
	  device 		Device		The running device, e.g. cpu/gpu
	  criterion	-			Specific loss function
	  optimizer	-			Reference to the optimzier
	  data_batch	-			Current batch data from train/test/val data loader
	  ```

	- The running parameters.
	
	  ```python
	  NAME						TYPE		MEANING
	  num_train_epoch				Int			The number of training epochs
	  num_train_batch				Int			The number of training batchs within one epoch
	  num_train_batch_last_epoch	Int			The number of training batchs within the last training epoch
	  grad_clip					Float		The threshold of gradient clipping
	  ```

- Dynamic variables: 
	- Indicators of current dataset and running mode

	  ```python
	  NAME			TYPE		MEANING
	  cur_mode		-			The current running mode, used to distinguish the statiscal variables, e.g. loss_train/loss_test/loss_val
	  cur_dataset		-			The current dataset
	  ```

	- Statistical variables to monitor the running status. 
	
	  ```python
	  NAME					TYPE		MEANING
	  loss_batch				Float		The loss of current batch		
	  loss_regular			Float		The loss of current regular term
	  loss_task				Float		The sum of loss_batch and loss_regular, used to compute the gradients
	  loss_total_train		Float		The sum of loss_batch during local training
	  pred					Tensor		The predict tensor output by the model
	  label					Tensor		The labels of current batch_data
	  num_samples_train		Int			The count of samples during local training
	  ```

### Note
Developers can add any variables to `Context` as they want. 

```python
ctx.{VARIABLE_NAME} = {value}
```

However, you must check the lifecycle of the record varibales carefully, and release them once they are not used. An unreleased variable may cause memory leakage during federated learning. 

## Multi-model Trainer 

Several learning methods may leverage multiple models in each client such as clustering based method [1] and multi-task learning based method [2], FederatedScope implements the `MultiModelTrainer` class to meet this requirement.

- We instantiate multiple models, optimizer objects & hook_sets as lists for `MultiModelTrainer`. Different internal models can have different hook_sets and optimizers to support diverse multi-model based methods
  
  ```python
  self.init_multiple_models()   
  # -> self.ctx.models = [...]  
  # -> self.ctx.optimizers = [...]
  self.init_multiple_model_hooks()  
  # -> self.hooks_in_train_multiple_models = [...]
  # -> self.hooks_in_eval_multiple_models = [...]
  ```
  
- To enable easy extension, we support copy initialization from a single-model trainer.

  ```python
  # By default, the internal models & optimizers are the same type
  additional_models = [
      copy.deepcopy(self.ctx.model) for _ in range(self.model_nums - 1)
  ]
  self.ctx.models = [self.ctx.model] + additional_models
  ```

- We can customized hooks and optimizers for multi-model interaction. Specifically,  two types of internal model interaction mode are built in `MultiModelTrainer` .

  - ```python
    # assert models_interact_mode in ["sequential", "parallel"]
    self.models_interact_mode = models_interact_mode
    ```

  - The `sequential` interaction mode indicates the interaction are conducted at run_routine level

    ```python
    [one model runs its whole routine, then do sth. for interaction, then next model runs its whole routine]
    ... -> run_routine_model_i
    		-> _switch_model_ctx
        -> (on_fit_end, _interact_to_other_models)   
        -> run_routine_model_i+1
        -> ...
    ```

  - The `parallel`  interaction mode indicates the interaction are conducted at point-in-time level

    ```python
    [At a specific point-in-time, one model call hooks (including interaction), then next model call hooks]
    ... ->  (on_xxx_point, hook_xxx_model_i)
        ->  (on_xxx_point, _interact_to_other_models)
        ->  (on_xxx_point, _switch_model_ctx)
        ->  (on_xxx_point, hook_xxx_model_i+1)
        -> ...
    ```

  - Note that these two modes call `_switch_model_ctx` at different positions. By default, we will switch cur_model, and optimizer, and users can override this function to support customized switch logic

    ```python
    def _switch_model_ctx(self, next_model_idx=None):
        if next_model_idx is None:
            next_model_idx = (self.ctx.cur_model_idx + 1) % len(
                self.ctx.models)
        self.ctx.cur_model_idx = next_model_idx
        self.ctx.model = self.ctx.models[next_model_idx]
        self.ctx.optimizer = self.ctx.optimizers[next_model_idx]
    ```
	
	
	
## Reference
[1] Felix Sattler, Klaus-Robert Müller, and Wojciech Samek. “Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization Under Privacy Constraints”. In: IEEE Transactions on Neural Networks and Learning Systems (2020).

[2] Marfoq, Othmane, et al. "Federated multi-task learning under a mixture of distributions." Advances in Neural Information Processing Systems 34 (2021).
