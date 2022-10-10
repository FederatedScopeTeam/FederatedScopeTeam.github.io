---
title: "Workers"
permalink: /docs/workers/
excerpt: "The Worker class used in FederatedScope."
last_modified_at: 2022-10-10T10:40:42-04:00
toc: true
layout: tuto
---

`Worker` class encapsulates the behaviors of FL participants (i.e., server and clients) and can be described via a collection of (message, handler) pairs. This tutorial focuses on helping you develop your own `Worker` subclasses. More details about the event-driven programming paradigm can be found in [Event-driven Architecture]({{ "/docs/event-driven-architecture/" | relative_url }}).

## Base Class of Workers

We show the class hierarchy for FS workers as below, and you can develop your own workers by inheriting the appropriate base class and making specialization.

```bash
Worker # Abstract class for both client and server
├── BaseClient # Abstract class for client
│   ├── Client # An implemented client with BaseClient
│   │   ├── XXXClient # Client for a specific algorithm with minor modifications
│   │   ├── ... ...
├── BaseServer # Abstract class for server
│   ├── Server # An implemented server with BaseServer
│   │   ├── XXXServer # Server for a specific algorithm with minor modifications
│   │   ├── ... ...
```

You can either develop your client (or server) from `federatedscope.core.workers.base_client.BaseClient`  (or `federatedscope.core.workers.base_server.BaseServer`) for a brand new client or  from `federatedscope.core.workers.client.Client` (or `federatedscope.core.workers.server.Server`) for minor modifications. 

We have two mechanisms below to help you keep your `Worker` subclasses complete, which are all essential for the usage of **Event-driven Architecture**.

## Completeness of implementation

In this section, we give a hands-on tutorial for developing a new client, which can be adapted to developing a new server. If you want to develop your own client with `BaseClient` as the base class, you must implement your own `callback_funcs_for_xxx`, which are `abc.abstractmethod` in `BaseClient` as shown below:

```python
class BaseClient(Worker):
    def __init__(self, ID, state, config, model, strategy):
        super(BaseClient, self).__init__(ID, state, config, model, strategy)
        self.msg_handlers = dict()
        self.msg_handlers_str = dict()
    def register_handlers(self, msg_type, callback_func, send_msg=[None]):
        if msg_type in self.msg_handlers.keys():
            logger.warning(f"Overwriting msg_handlers {msg_type}.")
        self.msg_handlers[msg_type] = callback_func
        self.msg_handlers_str[msg_type] = (callback_func.__name__, send_msg)

    def _register_default_handlers(self):
    		...

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_model_para(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_assign_id(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_join_in_info(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_address(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_evaluate(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_finish(self, message):
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_converged(self, message):
        raise NotImplementedError
```

These messages and handlers are necessary for making up a complete FL course. With them properly implemented, you can run FS with your own client!

## Completeness of messages and handlers

In this section, we will introduce the mechanism that would somewhat help you ensure the completeness of messages and handlers. This mechanism is to check the situation below:

![completeness](https://user-images.githubusercontent.com/39145382/192993254-32700f8e-9b71-4f2a-9737-1ffb76deddd1.png)

If the messages and handlers are not set properly, the FL course might fail due to missing handlers for some specific message, hander never being used, etc. Thus, this mechanism will help you debug.

### How to enable this check

Following our `config` rules, you can enable this feature by setting `cfg.check_completeness=True`, or running in the command line with `check_completeness True`:

```bash
python federatedscope/main.py --cfg scripts/example_configs/femnist.yaml check_completeness True
```

### Status for the completeness check

There are three statuses for the check: `Pass`, `WARNING,` and `Error`.

`Pass`: Everything goes fine with workers.

`WARNING`: The FL course goes well, but some handers are never used.

![warning](https://user-images.githubusercontent.com/39145382/193240024-d4a02949-6ac6-4a3c-8eac-d53d6d2e1cfd.png)

`Error`: The FL course fails as there is no path from start (i.e., the initial state) to end (i.e., the state corresponding to the end of an FL course).

### Debug information of the completeness check

With the help of the `networkx`, we can quickly visualize and find out whether there is a potential failure. A directed graph will be plotted and saved into `cfg.exp_dir` folder to remind you something might go wrong:

![msg_handler](https://user-images.githubusercontent.com/39145382/192993044-a44db17b-6295-4a21-a5d2-e2bca39aea52.png)
