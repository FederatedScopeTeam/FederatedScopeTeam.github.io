---
title: "FS Data"
permalink: /docs/fs-data/
excerpt: "The FS Data module used in FederatedScope."
last_modified_at: 2022-10-10T10:40:42-04:00
toc: true
layout: tuto
---

## FS Data Module

In the tutorial, you will learn how to use FS Data Module and the code structure is shown below:

```bash
federatedscope/core
├── auxiliaries
│   ├── data_builder
│   ├── dataloader_builder
│   ├── ...
├── data
│   ├── base_data
│   |   ├── ClientData
│   |   ├── StandaloneDataDict
│   ├── base_translator
│   ├── ...
```

We will discuss all the concepts of our FS data module from top to bottom.

### The main entrance of FS data

`federatedscope.core.auxiliaries.data_builder.get_data` is the entrance functions to access built-in FS data. With the `config` appropriately set, you can easily get FS Data.

```python
fs_data, modified_cfg = get_data(config=init_cfg.clone())
```

`get_data` consists of three steps:

* **Load Dataset**
  * `federatedscope.core.data.utils.load_dataset`
  * Load local file to torch dataset
* **Translate data**
  * `federatedscope.core.data.BaseDataTranslator`
  * Dataset -> ML split -> FL split ->  FS Dataloader
* **Convert mode**
  * `federatedscope.core.data.utils.convert_data_mode`
  * To adapt simulation mode and distributed mode

### Data Translator

In FederatedScope, the input to `Runner` is `ClientData` (in distributed mode) or `StandaloneDataDict` (in standalone mode), which are both subclasses of python `dict`. So FederatedScope provides Data Translator to help you convert `torch.utils.data.Dataset` to our data format. Data Translator contains four steps, two of which are optional.

**Dataset** -> **(ML split)** -> **(FL split)** -> **FS Dataloader**

* **ML split**(`split_train_val_test`): 
  * Build train/val/test data split
* **FL split**(`split_to_client`) (please see `splitter` for details): 
  * Split global data into local data for each client.

### ClientData

In FederatedScope, each client will obtain a `ClientData`, which has the following attributes:

* A subclass of `dict` with `train`, `val` and `test`.

* Convert `dataset`/`list`/`array` to DataLoader.

  Example:

  * ```python
    # Instantiate client_data for each Client
    client_data = ClientData(PyGDataLoader, 
                             cfg, 
                             train=train_data, 
                             val=None, 
                             test=test_data)
    # other_cfg with different batch size
    client_data.setup(other_cfg)
    print(client_data)
    
    >> {'train': PyGDataLoader(train_data), 'test': PyGDataLoader(test_data)}
    ```

### StandaloneDataDict

In standalone mode,  the input to `Runner` is `StandaloneDataDict`,  which is the return value of calling Data Translator. `StandaloneDataDict` has the following attributes:

* A subclass of `dict` with **client_id** as keys: 
  * `{1: ClientData, 2: ClientData, ...}`
* Responsible for some pre-process for FS data:
  * Global evaluation: merge test data
  * Global training: merge all data into one client
  * Injected data attacks
  * ...

