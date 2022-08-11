---
title: "Tuning Federated Learning"
permalink: /docs/use-hpo/
excerpt: "About using Federated HPO."
last_modified_at: 2022-07-21
toc: true
layout: tuto
---

It is well-known that deep neural networks are often sensitive to their hyperparameters, which need to be tuned carefully. When it comes to federated learning (FL), there are additional hyperparameters concerning the FL behaviors, which include the number of steps to make local update (`federate.local_update_steps`), the ratio of clients sampled at each round (`federate.sample_client_rate`), the coefficient of proximal regularization in FedProx (`fedprox.mu`), etc. Any poor choices for these hyperparameters would lead to unsatisfactory performances or even divergence.

Therefore, FederatedScope has provided the functionality of hyperparameter optimization (HPO), which saves our users from the tedious trial-and-error procedure.

## An hands-on exercise

We encourage our users to try the provided toy example of HPO by:

```bash
python demo/bbo.py
```

This toy example shows how to use Gaussian Process (GP) algorithm to seek the appropriate learning rate (`optimizer.lr`) for a logistic regression model.
There are more such scripts in this `demo/` folder to show how to use popular HPO packages (e.g., [SMAC3](https://github.com/automl/SMAC3) and [HpBandSter](https://github.com/automl/HpBandSter)) to interact with FederatedScope. After reading these script, users would be able to conduct HPO for their own FL cases with FederatedScope.

Meanwhile, we have implemented several popular HPO methods in the `autotune` module of FederatedScope, including random search [1], Successive Halving Algorithm (SHA) [2], etc. Users could try the rando search algorithm by:

```bash
python federatedscope/hpo.py --cfg scripts/example_configs/toy_rs.yaml
```

### How to declare the search space?
At first, any HPO procedure starts with declaring the search space, say that, which hyperparameters need to be determined and what are the candidate choices for them. FederatedScope allows users to specify the search space via the argument `hpo.ss` (in the above example, it is specified as 'scripts/example_configs/toy_hpo_ss.yaml'). As you can see from this .yaml file, the search space of each hyperparameter is described as a dict:

```yaml
train.optimizer.lr:
  type: float
  lower: 0.001
  upper: 0.5
  log: True
train.optimizer.weight_decay:
  type: cate
  choices: [0.0, 0.0005, 0.005]
```

where the key `type` specifies whether the search space of this hyperparameter is continuous or categorical. Then other keys and values correspond to the argument names and values for instantiating `UniformFloatHyperparameter` or `CategoricalHyperparameter` objects (see [ConfigureSpace package](https://automl.github.io/ConfigSpace/master/index.html)).

### How the scheduler interact with the FL runner?
Up to now, we have declared the search spaces. As random search algorithm has been adopted (`hpo.init_strategy == 'random'`  and default BruteForce scheduler is adopted), FederatedScope will randomly sample a specified number of candidate configurations from the declared search spaces. Specifically, to sample each configuration, the considered hyperparameters are enumerated, and choice for each hyperparameter is sampled from its search space uniformly:
> `optimizer.weight_decay` ~ Uniform({0.0, 0.0005, 0.005}),

It is worth mentioning that users are allowed to apply the uniform distribution to any continuous search space with log scale (by setting `hpo.log_scale` to be `True`), which is a convention in machine learning, e.g., considering learning rates 0.001, 0.01, and 0.1. After acquiring the specified number of candidate configurations, the scheduler, as an HPO agent, is going to interact with the FL runner, where each configuration is attempted (i.e., evaluated). Such interactions are repeated again and again till the agent has determined which configuration is the optimal, as the following figure illustrats.

![](https://img.alicdn.com/imgextra/i1/O1CN01lHkWop1XE7luBkfF8_!!6000000002891-0-tps-402-146.jpg#crop=0&crop=0&crop=1&crop=1&id=qzMAl&originHeight=146&originWidth=402&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

### Results of HPO
Executing this example eventually produces some outputs like the following:

![](https://img.alicdn.com/imgextra/i2/O1CN01ZKBQ5F1Vu8sK8Ko9F_!!6000000002712-2-tps-387-290.png#crop=0&crop=0&crop=1&crop=1&id=UUDzR&originHeight=290&originWidth=387&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

where, as you can see, the scheduler has evaluated 16 configurations each of which takes different choices of the learning rate and the weight decay coefficient. Since we have (by default) chosen the test loss as the evaluation metric (`hpo.metric == 'client_summarized_weighted_avg.test_loss`), the optimal configuration is the one that has the smallest value in the 'performance' column (by default, `hpo.larger_better == False`). Obvioiusly, taking `optimizer.lr=0.228037` and `optimizer.weight_decay=.0` is the best.

## More HPO algorithms

In addition to the simple search strategies including random search and grid search, FederatedScope has provided more sophisticated HPO algorithms, e.g., Hyperband (variant of Successive Halving Algorithm (SHA)) [2] and Population-Based Training (PBT) [3]. Users can specify which HPO algorithm to be used by setting `hpo.scheduler`.

Taking SHA as an example, we can try it by:

```bash
python federatedscope/hpo.py --cfg federatedscope/example_configs/toy_sha.yaml
```

As the default values for SHA are `hpo.sha.elim_round_num=3` and `hpo.sha.elim_rate=3`, the scheduler begins with $3 \times 3=27$ randomly sampled configurations. Then the search procedure continues iteratively. At the first round, all these 27 configurations are evaluated, and then only the top 1/3 candidates are reserved for the next round. In the next round, the reserved 9 configurations are evaluated, where each FL course is restored from the checkpoint resulted from the last round. The scheduler repeat such iterations untill there is only one configuration remaining. We just show the outputs of the final round:

![](https://img.alicdn.com/imgextra/i4/O1CN01UrrEC51vVZddBmQlY_!!6000000006178-2-tps-375-71.png#crop=0&crop=0&crop=1&crop=1&id=CA8y2&originHeight=71&originWidth=375&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

Obviously, the configuration to be reserved as the winner consists of the choices `optimizer.lr=0.223291` and `optimizer.weight_decay=0.0005`.

## Try it yourself

In this post, we have introduced the HPO functionalities of FederatedScope via two examples, i.e., random search and SHA. We encourage users to try these HPO algorithms on your own dataset (simply modify the data and model related fields in the .yaml file). For more details about the HPO functionalities of FederatedScope, please look up the API references.


## References

[1] Bergstra, James, and Yoshua Bengio. "Random search for hyper-parameter optimization." Journal of machine learning research 13.2 (2012).

[2] Li, Lisha, et al. "Hyperband: A novel bandit-based approach to hyperparameter optimization." The Journal of Machine Learning Research 18.1 (2017): 6765-6816.

[3] Jaderberg, Max, et al. "Population based training of neural networks." arXiv preprint arXiv:1711.09846 (2017).
