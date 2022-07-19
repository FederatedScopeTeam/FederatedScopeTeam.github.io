---
layout: single
permalink: /competition/
title: "CIKM 2022 AnalytiCup Competition"
author_profile: false
toc: true
---

# Federated Hetero-Task Learning


<h2>Introduction</h2>

We propose a new task, federated hetero-task learning, which meets the requirements of a wide range of real-world scenarios, while also promoting the interdisciplinary research of Federated Learning with Multi-task Learning, Model Pre-training, and AutoML. We have prepared an easy-to-use toolkit based on **FederatedScope [1,2]** to help participants easily explore this challenging yet manageable task from several perspectives, and also set up a fair testbed and different formats of awards for participants.

**We are running this competition on Tianchi competition platform. Please visit this [link](https://tianchi.aliyun.com/competition/entrance/532008/introduction).**


## Awards

- Prizes: 
  - 1st place: 5000 USD
  - 2nd place: 3000 USD
  - 3rd place: 1500 USD
  - 4th ~ 10th place: 500 USD each

- Certification: 
  - 1st ~ 20th: Certification with rank
  - Others: Certification with participation


## Schedule

- July 15, 2022: Competition launch. Sample dataset releases and simulation environment opens. Participants can register, join the discussion forum, upload the code for training and get feedback from leadboard.
- Sept 1, 2022: Registration ends.
- Sept 11, 2022: Submission ends. 
- Sept 12, 2022: Checking phase starts. Codes of top 30 teams will automatically be migrated into a checking phase. 
- Sept 18, 2022: Notification of checking results. 
- Sept 21, 2022: Announcement of the CIKM 2022 AnalytiCup Winner.
- Oct 17, 2022: Beginning of CIKM 2022.

All deadlines are at **11:59 PM UTC** on the corresponding day. The organizers reserve the right to update the contest timeline if necessary.


<h2>Problem description</h2>

In federated hetero-task learning, the learning goals of different clients are different. In practice, this setting is often observed due to personalized requirements of different clients, or the difficulty in aligning goals among multiple clients. Specifically, the problem is defined as follows:

  - Input: Several clients, each one is associated with a different dataset (feature space can be different) and a different learning objective.
  - Output: A learned model for each client.
  - Evaluation metric: The averaged improvement ratio (against the provided "isolated training" baseline) across all the clients.

More details can be found in this [page](https://tianchi.aliyun.com/competition/entrance/532008/information).

We provide the dataset for this competition via Tianchi. At the same time, we encourage you to see the exemplary federated hetero-task learning datasets defined in **B-FHTL [3]**, where the design and construction of these datasets are illustrated in the following picture:
<img src="https://img.alicdn.com/imgextra/i3/O1CN01yVaEBB25d2Gnu9mnh_!!6000000007548-0-tps-3422-1888.jpg" width="480" class="align-center">


<h2>References</h2>

[1] FederatedScope: A Flexible Federated Learning Platform for Heterogeneity. arXiv preprint 2022. [pdf](https://arxiv.org/pdf/2204.05011.pdf)

[2] FederatedScope-GNN: Towards a Unified, Comprehensive and Efficient Package for Federated Graph Learning. KDD 2022. [pdf](https://arxiv.org/pdf/2204.05562.pdf)

[3] A Benchmark for Federated Hetero-Task Learning. arXiv preprint 2022. [pdf](https://arxiv.org/pdf/2206.03436v2.pdf)


# Step-by-step Guidance for CIKM22 AnalytiCup Competition

## Step1. Install FederatedScope

Download FederatedScope and switch to the stable branch `cikm22competition` for this competition:

```shell
git clone https://github.com/alibaba/FederatedScope.git

cd FederatedScope

git checkout cikm22competition
```

## Step 2. Setup the running environment

You can use Docker or Conda to setup your running environment

- Use Docker

  - Check your cuda version via the command
  
      ```shell
      nvidia-smi
      ```
  
  - Build the corresponding docker image
    - If your CUDA Version >= 11:
  
      ```shell
      docker build -f enviroment/docker_files/federatedscope-torch1.10-application.Dockerfile -t alibaba/federatedscope:base-env-torch1.10 .
      
      docker run --gpus device=all --rm -it --name "fedscope" -v $(pwd):$(pwd) -w $(pwd) alibaba/federatedscope:base-env-torch1.10 /bin/bash
      
      pip install -e .
      ```

    - If your CUDA Version >= 10 but <11:

      ```shell
      docker build -f enviroment/docker_files/federatedscope-torch1.8-application.Dockerfile -t alibaba/federatedscope:base-env-torch1.8 .
      
      docker run --gpus device=all --rm -it --name "fedscope" -v $(pwd):$(pwd) -w $(pwd) alibaba/federatedscope:base-env-torch1.8 /bin/bash
      
      pip install -e .
      ```

- Use Conda

  - We recommend using a new virtual environment to install FederatedScope:

      ```shell
      conda create -n fs python=3.9
      conda activate fs
      ```
  
  - If you are using torch, please install it in advance ([torch-get-started](https://pytorch.org/get-started/locally/)). For example, if your cuda version is 11.3 please execute the following command:
  
      ```shell
      conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
      ```
  
  - Install the required packages as follows
  
      ```shell
      python setup.py install
      ```
  
  - Finally, install packages required by graph tasks as follows
  
      ```shell
      bash environment/extra_dependencies_torch1.10-application.sh
      ```

## Step 3. Download contest data

- Click sign up in [Tianchi](https://tianchi.aliyun.com/competition/entrance/532008/introduction), and register your account as follows if you don't have one.

  <figure>
  <img src="https://gw.alicdn.com/imgextra/i3/O1CN018QES7i1HpeOQk2BW6_!!6000000000807-2-tps-5120-2610.png" width=480>
  <figcaption>Click sign up in Tianchi.</figcaption>
  </figure>

  <figure>
  <img src="https://gw.alicdn.com/imgextra/i1/O1CN01ZnsXhT1RUG3ArkI1U_!!6000000002114-2-tps-5120-2606.png" width=480>
  <figcaption>Register your account.</figcaption>
  </figure>

- Login and download the [contest data](https://tianchi.aliyun.com/competition/entrance/532008/information)

  <figure>
  <img src="https://img.alicdn.com/imgextra/i4/O1CN015Po1qF24yH1ZKWJ86_!!6000000007459-2-tps-5120-2608.png" width=480>
  <figcaption>Download the contest data.</figcaption>
  </figure>

- Suppose the contest data is placed in `${YOUR_OWN_PATH}/CIKM22Competition.zip`, unzip the contest data as follows 

    ```shell
    mkdir data
    unzip -d ./data/ ${YOUR_OWN_PATH}/CIKM22Competition.zip
    ```

- Then you can access the contest data with the directory `FederatedScope/data/CIKM22Competition`. The contest data is organized by the index of the client `CIKM22Competition/${client_id}` (counts from 1), and the data of each client contains the train, test and validate splitted parts. You can load it by `torch.load` as follows:

    ```python
    import torch
    # The train split of client 1
    train_data_client1 = torch.load('./data/CIKM22Competition/1/train.pt')
    # Check the first sample
    print(train_data_client1[0])
    # Check the label of the first sample
    print(train_data_client1[0].y)
    # Check the index of the first sample as ${sample_id}
    print(train_data_client1[0].data_index)
    ```

## Step 4. Execute baselines on the contest data

Within FederatedScope, we build in two baselines for the contest data, "isolated training" and "FedAvg". Suppose you have successfully built the running environment, and downloaded the contest data

  - Run the following command to execute the isolated training

    ```shell
    python federatedscope/main.py --cfg federatedscope/gfl/baseline/isolated_gin_minibatch_on_cikmcup.yaml --client_cfg federatedscope/gfl/baseline/isolated_gin_minibatch_on_cikmcup_per_client.yaml
    ```

  - Run the following command to execute the FedAvg solution

    ```shell
    python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gin_minibatch_on_cikmcup.yaml --client_cfg federatedscope/gfl/baseline/fedavg_gin_minibatch_on_cikmcup_per_client.yaml
    ```

where the argument `--cfg xxxx.yaml` specifies the global configuration, and `--client_cfg xxx.yaml` specifies the client-wise hyper parameters.

## Step 5. Save and submit the prediction results

**Submission format**

- As stated in the [introduction of CIKM 2022 AnalytiCup Competition](https://tianchi.aliyun.com/competition/entrance/532008/information?lang=en-us), participants are required to submit the prediction results for all clients within one csv file. Within the file, each line records one prediction and is identified by `${client_id}` and `${sample_id}`. The `${client_id}` counts from 1 and `${sample_id}` should be consistent with the contest data (You can access it by the attribute `data_index`).
- The classification and multi-dimensional regression tasks follow different formats as follows:
  - For classification tasks, each line follows (`${category_id}` counts from 0)

    ```
    ${client_id},${sample_id},${category_id}
    ```

  - For N-dimensional regression task, each line follows

    ```
    ${client_id},${sample_id},${prediction_1st_dimension},…,${prediction_N-th_dimension}
    ```

**Saving prediction results**

- By FederatedScope

  - The "cikm22competition" branch in FederatedScope supports to save prediction results at the end of training. You can refer to code in  [federatedscope/gfl/trainer/graphtrainer.py](https://github.com/alibaba/FederatedScope/blob/cikm22competition/federatedscope/gfl/trainer/graphtrainer.py) and [federatedscope/core/trainers/torch_trainer.py](https://github.com/alibaba/FederatedScope/blob/cikm22competition/federatedscope/core/trainers/torch_trainer.py). 
  - You can specify the path of prediction results by modifying `config.eval.prediction_path`, which defaults to "prediction". Taking FedAvg as an example, at the end of training FederatedScope will report the path of prediction results as follows:
  
    <img src="https://img.alicdn.com/imgextra/i1/O1CN01eb3zR21QUsyjdXAM6_!!6000000001980-2-tps-4920-404.png" width=480>
  
  - Then you can refer to the directory for prediction results. 
  
    <img src="https://img.alicdn.com/imgextra/i1/O1CN01ndig5n1vNmvicmDBT_!!6000000006161-2-tps-1988-887.png" width=480>

- By Yourself

  Also, you can save the prediction results by yourself. Within a test routine in FederatedScope, the output of the model are all saved in the context as follows,  and you can access it by `ctx.test_y_prob`

  ```python
      def _hook_on_fit_start_init(self, ctx):
          ...
          setattr(ctx, "{}_y_true".format(ctx.cur_data_split), [])
          setattr(ctx, "{}_y_prob".format(ctx.cur_data_split), [])
  
      ...
      
      def _hook_on_batch_forward(self, ctx):
          batch = ctx.data_batch.to(ctx.device)
          pred = ctx.model(batch)
          # TODO: deal with the type of data within the dataloader or dataset
          if 'regression' in ctx.cfg.model.task.lower():
              label = batch.y
          else:
              label = batch.y.squeeze(-1).long()
          if len(label.size()) == 0:
              label = label.unsqueeze(0)
          ctx.loss_batch = ctx.criterion(pred, label)
  
          ctx.batch_size = len(label)
          ctx.y_true = label
          ctx.y_prob = pred
  
          # record the index of the ${MODE} samples
          if hasattr(ctx.data_batch, 'data_index'):
              setattr(
                  ctx,
                  f'{ctx.cur_data_split}_y_inds',
                  ctx.get(f'{ctx.cur_data_split}_y_inds') + ctx.data_batch.data_index.detach().cpu().numpy().tolist()
              )
      
      ...
      
      def _hook_on_batch_end(self, ctx):
          ...
          # cache label for evaluate
          ctx.get("{}_y_true".format(ctx.cur_data_split)).append(
              ctx.y_true.detach().cpu().numpy())
  
          ctx.get("{}_y_prob".format(ctx.cur_data_split)).append(
              ctx.y_prob.detach().cpu().numpy())
      
      ...
      
      def _hook_on_fit_end(self, ctx):
          """Evaluate metrics.
  
          """
          setattr(
              ctx, "{}_y_true".format(ctx.cur_data_split),
              np.concatenate(ctx.get("{}_y_true".format(ctx.cur_data_split))))
          setattr(
              ctx, "{}_y_prob".format(ctx.cur_data_split),
              np.concatenate(ctx.get("{}_y_prob".format(ctx.cur_data_split))))
          ...
    ```
        
**Submit prediction results**

Finally, you can submit your prediction results and get your score in [Tianchi](https://tianchi.aliyun.com/competition/entrance/532008):

<figure>
<img src="https://img.alicdn.com/imgextra/i3/O1CN01oAtui11asNRzLPxkP_!!6000000003385-2-tps-1990-820.png" width=480>
<figcaption>Submit the prediction results.</figcaption>
</figure>

**Get the evaluation feedback**
In Tianchi, the submitted prediction results will be evaluated by the metric of "average improve ratio", which is calculate as:

$$\text{averaged improvement ratio}=\frac{1}{n}\sum_{i=1}^{n}(\frac{b_i-m_i}{b_i}\times 100\%),$$

where $n$ is the total number of clients; when client $i$ owns the classification task, $m_i$ and $b_i$ are the error rate of the developed method and "isolated training" baseline, respectively; when client $i$ has a regression task, $m_i$ and $b_i$ correspond to their mean squared error (MSE). 

To ensure a fair competition, we will use the following $b_i$for the 13 clients to calculate the averaged improvement ratio.

| Client ID | Task type | Metric | $b_i$ |
| --- | --- | --- | --- |
| 1 | cls | Error rate | 0.263789 |
| 2 | cls | Error rate | 0.289617 |
| 3 | cls | Error rate | 0.355404 |
| 4 | cls | Error rate | 0.176471 |
| 5 | cls | Error rate | 0.396825 |
| 6 | cls | Error rate | 0.261580 | 
| 7 | cls | Error rate | 0.302378 | 
| 8 | cls | Error rate | 0.211538 | 
| 9 | reg | MSE | 0.059199 |
| 10 | reg | MSE | 0.007083 | 
| 11 | reg | MSE | 0.734011 | 
| 12 | reg | MSE | 1.361326 | 
| 13 | reg | MSE | 0.004389 |

After submit the prediction results, you can check the evaluation results as follows, and the Leaderboard will update at *UTC time 00:00, 04:00, 08:00, 12:00 and 16:00*.

<img src="https://img.alicdn.com/imgextra/i3/O1CN01iZueTJ1Di5OADICIn_!!6000000000249-2-tps-1988-1064.png" width=480>


# Advanced Guidance for Participants

## About FederatedScope

FederatedScope is a well-modularized federated learning platform. Participants are welcome and encouraged to develop their own federated solutions with FederatedScope. The following documents will help you better understand the organization of FederatedScope and how it works

- [Tutorial of FederatedScope](https://federatedscope.io/docs/quick-start/)
- [Paper of FederatedScope](https://arxiv.org/abs/2204.05011)
- [API Document of FederatedScope](https://federatedscope.io/refs/index)

## Develop your own solution
You can develop your own algorithm based on FederatedScope as follows:

- If you want to improve the performance of the baseline, you can adjust the global hyper parameters (specified by `--cfg`) or adjust the hyper parameters for each client (specified by `--client_cfg`). Taking FedAvg as an example:
  - The global configuration `federatedscope/gfl/baseline/isolated_gin_minibatch_on_cikmcup.yaml` specifies the global settings, such as the total training round (`federate.total_round_num`), dataset (`data.type` and `data.root`)  and evaluation metric (`eval.metrics`)

    ```yaml
    use_gpu: True
    device: 0
    early_stop:
      patience: 20
      improve_indicator_mode: mean
      the_smaller_the_better: False
    federate:
      mode: 'standalone'
      make_global_eval: False
      total_round_num: 100
      share_local_model: False
    data:
      root: data/
      type: cikmcup
    model:
      type: gin
      hidden: 64
    personalization:
      local_param: ['encoder_atom', 'encoder', 'clf']
    train:
      batch_or_epoch: epoch
      local_update_steps: 1
      optimizer:
        weight_decay: 0.0005
        type: SGD
    trainer:
      type: graphminibatch_trainer
    eval:
      freq: 5
      metrics: ['imp_ratio']
      report: ['avg']
      best_res_update_round_wise_key: val_imp_ratio
      count_flops: False
      base: 0.
    ```

  - The client configuration `federatedscope/gfl/baseline/fedavg_gin_minibatch_on_cikmcup_per_client.yaml` allows you to set different hyper parameters for different clients by replacing the global configuration, and we use the argument `client_${client_id}` to specify the client. Here we set the loss function and the model according to the distributed tasks, and use `eval.base` to provide the basic performance for the metric `imp_ratio`. 

    ```yaml
    client_1:
      model:
        out_channels: 2
        task: graphClassification
      criterion:
        type: CrossEntropyLoss
      train:
        optimizer:
          lr: 0.1
      eval:
        base: 0.263789
    client_2:
      model:
        out_channels: 2
        task: graphClassification
      criterion:
        type: CrossEntropyLoss
      train:
        optimizer:
          lr: 0.01
      eval:
        base: 0.289617
    client_3:
      model:
        out_channels: 2
        task: graphClassification
      criterion:
        type: CrossEntropyLoss
      train:
        optimizer:
          lr: 0.001
      eval:
        base: 0.355404
    client_4:
      model:
        out_channels: 2
        task: graphClassification
      criterion:
        type: CrossEntropyLoss
      train:
        optimizer:
          lr: 0.01
      eval:
        base: 0.176471
    client_5:
      model:
        out_channels: 2
        task: graphClassification
      criterion:
        type: CrossEntropyLoss
      train:
        optimizer:
          lr: 0.0001
      eval:
        base: 0.396825
    client_6:
      model:
        out_channels: 2
        task: graphClassification
      criterion:
        type: CrossEntropyLoss
      train:
        optimizer:
          lr: 0.0005
      eval:
        base: 0.261580
    client_7:
      model:
        out_channels: 2
        task: graphClassification
      criterion:
        type: CrossEntropyLoss
      train:
        optimizer:
          lr: 0.01
      eval:
        base: 0.302378
    client_8:
      model:
        out_channels: 2
        task: graphClassification
      criterion:
        type: CrossEntropyLoss
      train:
        optimizer:
          lr: 0.05
      eval:
        base: 0.211538
    client_9:
      model:
        out_channels: 1
        task: graphRegression
      criterion:
        type: MSELoss
      train:
        optimizer:
          lr: 0.1
      eval:
        base: 0.059199
    client_10:
      model:
        out_channels: 10
        task: graphRegression
      criterion:
        type: MSELoss
      train:
        optimizer:
          lr: 0.05
      grad:
        grad_clip: 1.0
      eval:
        base: 0.007083
    client_11:
      model:
        out_channels: 1
        task: graphRegression
      criterion:
        type: MSELoss
      train:
        optimizer:
          lr: 0.05
      eval:
        base: 0.734011
    client_12:
      model:
        out_channels: 1
        task: graphRegression
      criterion:
        type: MSELoss
      train:
        optimizer:
          lr: 0.01
      eval:
        base: 1.361326
    client_13:
      model:
        out_channels: 12
        task: graphRegression
      criterion:
        type: MSELoss
      train:
        optimizer:
          lr: 0.05
      grad:
        grad_clip: 1.0
      eval:
        base: 0.004389
      ```

- If you want to modify the clients, it is suggested to create a new trainer that inherits the basic trainer as follows, and you can replace the original hook functions with yours. For example, you can implement your trainer with new `_hook_on_batch_forward` function, and you need to set `trainer.type` as "new_trainer" to use it.

  ```python
  from federatedscope.register import register_trainer
  from federatedscope.core.trainers import GeneralTorchTrainer
  
  
  class NewTrainer(GeneralTorchTrainer):
      def _hook_on_batch_forward(self, ctx):
          pass
          
  
  def call_new_trainer(trainer_type):
      if trainer_type == 'new_trainer':
          trainer_builder = NewTrainer
          return trainer_builder
  
  
  register_trainer('new_trainer', call_new_trainer)
  ```

- If you want to modify the server for federated learning, it is suggested to create a new aggregator and register it to FederatedScope. 
  - First, you can inherit the following abstract class `Aggregator` in `federatedscope/core/aggregator.py` and implement your own `aggregate` function:

    ```python
    class Aggregator(ABC):
        def __init__(self):
            pass
    
        @abstractmethod
        def aggregate(self, agg_info):
            pass
    ```

  - Then you can register your aggregator within the `federate/core/auxiliaries/aggregator_builder.py`.


# For More Help

We encourage participants to join our Q&A [slack channel](https://join.slack.com/t/federatedscopeteam/shared_invite/zt-1apmfjqmc-hvpYbsWJdm7D93wPNXbqww) or join our DingGroup by scanning the QR code:

<img src="https://gw.alicdn.com/imgextra/i4/O1CN01heXHpf1zuXhcOCgGF_!!6000000006774-2-tps-860-861.png" width="300" />
