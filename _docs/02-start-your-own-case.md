---
title: "Start Your Own Case"
permalink: /docs/own-case/
excerpt: "Start your own cases."
last_modified_at: 2018-03-20T15:19:22-04:00
toc: true
---

In addition to the rich collcetion of **datasets**, **models** and **evaluation metrics**, FederatedScope also allows to create your own or introduce more to our package.

We provide `register` function to help build your own federated learning workflow.  This introduction will help you to start with your own case:

1. [Load a dataset](#data)
1. [Build a model](#model)
1. [Create a trainer](#trainer)
1. [Introduce more evaluation metrics](#metric)
1. [Specify your own configuration](#config)

<a name="SmCQW"></a>
## Load a dataset

We provide a function `flpackage.register.register_data` to make your dataset available with three steps:

-  Step1: set up your data in the following format (standalone):<br />**Note**: This function returns a `dict`, where the `key` is the client's id, and the `value` is the data `dict` of each client with 'train', 'test' or 'val'.  You can also modify the config here. 
```python
def load_my_data(config):
		r"""
    		NOTE: client_id 0 is SERVER for global evaluation!
    
    Returns:
          dict: {
                    'client_id': {
                        'train': DataLoader or Data,
                        'test': DataLoader or Data,
                        'val': DataLoader or Data,
                    }
                }
    """
    ...
    return data_dict, config
```


We take `torchvision.datasets.MNIST`, which is split and assigned to two clients, as an example: 
```python
def load_my_data(config):
    import numpy as np
    from torchvision import transforms
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader

    # Build data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    data_train = MNIST(root='data', train=True, transform=transform)
    data_test = MNIST(root='data', train=False, transform=transform)

    # Split data into dict
    data_dict = dict()
    train_per_client = len(data_train) // config.client_num
    test_per_client = len(data_test) // config.client_num

    for client_idx in range(1, config.client_num + 1):
        dataloader_dict = {
            'train':
            DataLoader([
                data_train[i]
                for i in range((client_idx - 1) *
                               train_per_client, client_idx * train_per_client)
            ],
                       config.data.batch_size,
                       shuffle=config.data.shuffle),
            'test':
            DataLoader([
                data_test[i]
                for i in range((client_idx - 1) * test_per_client, client_idx *
                               test_per_client)
            ],
                       config.data.batch_size,
                       shuffle=False)
        }
        data_dict[client_idx] = dataloader_dict

    return data_dict, config
```
 

-  Step2: register your data with a keyword, such as `"mydata"` 
```python
from flpackage.register import register_data

def call_my_data(config):
    if config.data.type == "mydata":
        data, modified_config = load_my_data(config)
        return data, modified_config

register_data("mydata", call_my_data)
```
 

-  Step3: put this `.py` file in the `flpackage/contrib/data/` folder, and set `cfg.data.type = "mydata"` to use it. 

Also,  you can modify the source code to make the **FederatedScope** support your dataset. Please see [flpackage.core.auxiliaries.data_builder](flpackage/core/auxiliaries/data_builder.py) , and you can add an `elif` to skip `Step2` and `Step3` above.

<a name="AgRA6"></a>
## Build a model

We provide a function `flpackage.register.register_model` to make your model available with three steps: (we take `ConvNet2` as an example)

-  Step1: build your model with Pytorch or Tensorflow and instantiate your model class with config and data. 
```python
import torch

class MyNet(torch.nn.Module):
    def __init__(self, in_channels, h=32, w=32, hidden=2048, class_num=10, use_bn=True):
        super(MyNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, 5, padding=2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = torch.nn.Linear((h // 2 // 2) * (w // 2 // 2) * 64, hidden)
        self.fc2 = torch.nn.Linear(hidden, class_num)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(self.relu(x))
        x = self.conv2(x)
        x = self.maxpool(self.relu(x))
        x = torch.nn.Flatten()(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
      
def load_my_net(model_config, local_data):
  	# You can also build models without local_data
    model = MyNet(in_channels=x.shape[1],
                  h=x.shape[2],
                  w=x.shape[3],
                  hidden=model_config.hidden,
                  class_num=model_config.out_channels)
    return model
```
 

-  Step2: register your model with a keyword, such as `"mynet"` 
```python
from flpackage.register import register_model

def call_my_net(model_config, local_data):
    if model_config.type == "mynet":
        model = load_my_net(model_config, local_data)
        return model

register_model("mynet", call_my_net)
```
 

-  Step3: put this `.py` file in the `flpackage/contrib/model/` folder, and set `cfg.model.type = "mynet"` to use it. 

Also,  you can modify the source code to make the **FederatedScope** support your model. Please see [flpackage.core.auxiliaries.model_builder](/flpackage/core/auxiliaries/model_builder.py) , and you can add an `elif` to skip `Step2` and `Step3` above.

<a name="l3cbN"></a>
## Create a trainer

FederatedScope decouples the local learning process and details of FL communication and schedule, allowing users to freely customize the local learning algorithms via the `Trainer`. We recommend user build trainer by inheriting `flpackage.core.trainers.trainer.GeneralTrainer`, for more details, please see [Trainer](xxxxxx). Similarly, we provide `flpackage.register.register_trainer` to make your customized trainer available:

-  Step1: build your trainer by inheriting `GeneralTrainer`. Our `GeneralTrainer` already supports many different usages, for the advanced user, please see flpackage.core.trainers.trainer.GeneralTrainer for more details. 
```python
from flpackage.core.trainers.trainer import GeneralTrainer

class MyTrainer(GeneralTrainer):
    pass
```
 

-  Step2: register your trainer with a keyword, such as `"mytrainer"` 
```python
from flpackage.register import register_trainer

def call_my_trainer(trainer_type):
    if trainer_type == 'mytrainer':
        trainer_builder = MyTrainer
        return trainer_builder

register_trainer('mytrainer', call_my_trainer)
```
 

-  Step3: put this `.py` file in the `flpackage/contrib/trainer/` folder, and set `cfg.trainer.type = "mytrainer"` to use it. 

Also,  you can modify the source code to make the **FederatedScope** support your model. Please see `flpackage/core/auxiliaries/trainer_builder.py` , and you can add an `elif` to skip `Step2` and `Step3` above.

<a name="Z43Cb"></a>
## Introduce more evaluation metrics

We provide a number of metrics to monitor the entire federal learning process. You just need to list the name of the metric you want in `cfg.eval.metrics`. We currently support metrics such as loss, accuracy, etc. (See [flpackage.core.evaluator](flpackage/core/evaluator.py) for more details).

We also provide a function `flpackage.register.register_metric` to make your evaluation metrics available with three steps:

-  Step1: build your metric (see [flpackage.core.context](flpackage/core/context.py) for more about `ctx`) 
```python
def cal_my_metric(ctx, **kwargs):
  	...
    return MY_METRIC_VALUE
```
 

-  Step2: register your metric with a keyword, such as `"mymetric"` 
```python
from flpackage.register import register_metric

def call_my_metric(types):
    if "mymetric" in types:
        metric_builder = cal_my_metric
        return "mymetric", metric_builder

register_metric("mymetric", call_my_metric)
```
 

-  Step3: put this `.py` file in the `flpackage/contrib/metircs/` folder, and add `"mymetric"` to `cfg.eval.metric` activate it. 


<a name="config"></a>
## Introduce your own configuration
### Basic usage
FederatedScope provides an extended configuration system based on [yacs](https://github.com/rbgirshick/yacs). We leverage a two-level tree structure that consist of several internal dict-like containers to allow simple key-value access and management. For example,
```
cfg.backend = 'torch'  # level-1 configuration

cfg.federate = CN()  # level-2 configuration
cfg.federate.client_num = 0
```
The frequently-used APIs include
- `merge_from_file`, `merge_from_other_cfg` and `merge_from_list` that load configs from a yaml file, another `cfg` instance or a list stores the keys and values.
- Besides, we can use `freeze` to make the configs immutable and save the configs in a yaml file under the specified `cfg.outdir`.
- Both these functions will trigger the configuration validness checking.
- To modify a config node after calling `freeze`, we can call `defrost`.

### Built-in configurations
We divide the configuration could be used in the FL process into several sub files such as `cfg_fl_setting`, `cfg_fl_algo`, `cfg_model`, `cfg_training`, `cfg_evaluation`, see more details in `federatedscope/core/configs` directory.

### Customized configuration
To add new configuration, you need 
1. implement your own extend function `extend_my_cfg(cfg):`, e.g.,
   ```
   def extend_training_cfg(cfg):
       # ------------------------------------------------------------------------ #
       # Trainer related options
       # ------------------------------------------------------------------------ #
       cfg.trainer = CN()

       cfg.trainer.type = 'general'
       cfg.trainer.finetune = CN()
       cfg.trainer.finetune.steps = 0
       cfg.trainer.finetune.only_psn = True
       cfg.trainer.finetune.stepsize = 0.01
        
       # --------------- register corresponding check function ----------
       cfg.register_cfg_check_fun(assert_training_cfg)
   ```
2. and implement your own config validation check function `assert_my_cfg`, e.g.,
   ```
   def assert_training_cfg(cfg):
       if cfg.backend not in ['torch', 'tensorflow']:
           raise ValueError(
                "Value of 'cfg.backend' must be chosen from ['torch', 'tensorflow']."
           )
   ```
    
3. finally, register your own extended function, e.g.,
   ```
   from federatedscope.register import register_config
   register_config("fl_training", extend_training_cfg)
   ```
   
   
We recommend users put the new customized configuration in `federatedscope/contrib/configs` directory
  
