---
title: "Contributing to FederatedScope"
permalink: /docs/contributor/
excerpt: "About contributing to FederatedScope."
last_modified_at: 2022-04-13T19:47:43-05:00
toc: true
layout: tuto
---

As an open-sourced project, we categorize our users into two classes:

- Running our code with different configurations, or re-organizing the training courses with their own entry scripts. These users may have no interest in understanding our implementation details.
- Customizing some modules of FederatedScope by themselves, e.g., customized data loader, novel function approximators, and even novel federated learning algorithms. If you believe that other users could benefit from your ideas, we always welcome your pull-requests.

This post targets for the latter. We give an overview of our design and clarify the requirements any pull-request should obey.

## Develop new modules

At first, we elaborate on how the various modules of our package FederatedScope are organized together. Our package is implemented by Python language, and thus the submodules are all located in the folder `federatedscope/`:

```
federatedscope/
    |- attack/
    |- autotune/
    |- contrib/
    |- core/
    |- cv/
    |- gfl/
    |- methods/
    |- mf/
    |- nlp/
    |- vertical_fl/
    |- __init__.py
    |- config.py
    |- hpo.py
    |- main.py
    |- parse_exp_results.py
    |- register.py
```

The .py files `main.py` and `hpo.py` are entry files to conduct a single FL course and an HPO procedure consisting a series of FL courses, respectively. The .py file `config.py` maintains all the involved hyperparameters. As for the folders, each corresponds to a specific submodule, where we discuss several of them as follow:

- `federatedscope.core`: The infrastructure for conducting a FL course, including the definition of key concepts of runner, worker, and trainer. For more details about the design of FederatedScope, we refer our readers to our another post ["Event-driven Architecture"]({{ "/docs/event-driven architecture/" | relative_url }}).
- `federatedscope.gfl`, `federatedscope.cv`, `federatedscope.nlp`, and `federatedscope.mf`: Data, models, and trainers dedicated to the corresponding application domain, which has led to many successful applications.
- `federatedscope.autotune`: The AutoML-related functionalities, which has provided a rich collection of hyperparameter optimization methods (see the posts about [HPO usage]({{ "/docs/use-hpo/" | relative_url }}) and [HPO development]({{ "/docs/improve-hpo/" | relative_url }}) for more details), yet stuff about automatic feature engineering and neural architecture search will come later.
- `federatedscope.attack`: The privacy attack and defense related functionalities ([more details]({{ "/docs/privacy-attacks/" | relative_url }})), which enable our users to further validate the privacy-preserving property of their FL instances.

Although FederatedScope has provided these out-of-the-box FL capabilities, we highly encourage our users to express their novel ideas via FederatedScope and contribute the developed new modules into our package. For the ease of seeminglessly integrating contributed modules, we follow a popular design among open-sourced machine learning packages that allows any external class/function/variable to be registered as internal stuff of the package:

- `register.py`: There are several dict objects, e.g., `trainer_dict`, `metric_dict`, `criterion_dict`, etc. Each such dict contains the objects designated for the corresponding purpose and can be further augmented with new objects.
- `contrib/` folder: Users are expected to put their new modules there. We have provided a new metric as an example.

We refer our readers to the post ["Start your own case"]({{ "/docs/own-case/" | relative_url }}) for more examples about this register mechanism.

## Ready to submit your pull-request

Please run `scripts/format.sh` and apply the changes. Otherwise, our linter won't let your pull-request pass.

Please run `scripts/ci_test.sh` and ensure all test cases are passed. Otherwise, the test automatically triggered by your pull-requests would fail.
