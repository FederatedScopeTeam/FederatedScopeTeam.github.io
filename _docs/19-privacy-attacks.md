---
title: "Privacy Attacks"
permalink: /docs/privacy-attacks/
excerpt: "We provide the implementation of privacy attacker in FederateScope for developers to conveniently demonstrate privacy-preserving strength of the design systems and algorithms, since applying privacy attacks directly on the algorithm is an effective way to detect the vulnerability of FL."
last_modified_at: 2018-03-20T16:00:02-04:00
toc: true
layout: tuto
---

We provide the implementation of privacy attacker in FederateScope for developers to conveniently demonstrate the privacy-preserving strength of the design systems and algorithms, since applying privacy attacks directly on the algorithm is an effective way to detect the vulnerability of FL.

In this part, we will introduce:
1. [Types of Privacy attacks.](##1.-Background-of-Privacy-Attack-in-FL)
2. Usages of attack module in FederatedScope
     * [Example of Optimization based Training Data/Label Inference Attack](#2.2-Example-of-Optimization-based-Training-Data/Label-Inference-Attack)
     * [Example of Class Representative Attack](#2.3-Example-of-Class-Representative-Attack)
     * [Example of Membership Inference Attack](#2.4-Example-of-Membership-Inference-Attack)
     * [Example of Property Inference Attack](#2.5-Example-of-Property-Inference-Attack)
4. [Develop your own attacker.](#3.-Develop-Your-Own-Attack)
5. [Contribute your attackers.](#4.-Contribute-Your-Attacker-to-FederatedScope)

## 1. Background of Privacy Attack in FL
This part briefly introduces the types of privacy attacks in FL. For more detailed information about the privacy attacks in FL, please refer [[7]](#7).

### 1.1 Different Attack Targets
According to the types of attacker's target, typical attacks include membership inference attack, property inference attack, class representative attack, and training data/label inference attack. 

1. **Membership Inference Attack (MIA)**: In membership inference attack, an attacker can be a server or a client, and the objective is to infer whether the specific given data exists in another client's private dataset.

2. **Property Inference Attack (PIA)**: Property inference attack aims to infer the dataset property (may be the sensitive property) other than the class label. For example, in the facial image dataset, the original task is to infer whether wearing glass, an attacker may also be curious about the gender and age unrelated to the original task. 

3.  **Class Representative Attack (CRA)**: Class representative attack aims to infer the representative sample of a specific class. This type of attack often exists in the case where a client only owns a part of the class label, and is curious about the information related to other classes.

4.  **Training Data/Label Inference Attack (TIA)**: Training data/label inference attack aims to reconstruct the privately owned training samples through the intermediate information transmitted during the FL. 

### 1.2 Active Attack v.s. Passive Attack
According to the attacker's actions, the privacy attacks can be divided into passive and active attacks.

1. **Passive Attack**: In a passive attack, the attacker follows the FL protocols, and only saves the intermediate results or the received information for local attack computation. Due to the characteristics of a passive attack, it is very hard to be detected by the FL monitor.


2. **Active Attack**: Different from the passive attack, in the active attack, the attacker often injects malicious information into FL to induce other clients to reveal more private information into the global model. The attacker performing active attacks are also named malicious. Compared with the passive attack, due to the malicious information injection, this type of attack is easier to be detected with additional information checking.  



## 2. Usage of Attack Module
### 2.1 Preliminary
Before calling the attack module, the user should make sure that the FL has been set up. For more details of setting up FL, please refer [quick start]({{ "/docs/quick-start/" | relative_url }}) and [start your own case]({{ "/docs/own-case/" | relative_url }}). 

The attack module provides several attack methods for directly using. The users only need to set the corresponding hyper-parameters in the configuration to call the corresponding method and add the prior knowledge to the attacker.

In order to make it easier for users to check the privacy protection strength of FL, FederatedScope provides implementations of SOTA privacy attack algorithms. The users only need to add the configuration of the attack module into FL's configuration, and provide the additional information required by the attack method. 

The implemented privacy algorithms include:
(1) Membership inference attack: Gradient Ascent (Active attack) [[5]](#5) .
(2) property inference attack: BPC (Passive attack) [[6]](#6).
(3) Class representative attack: DCGAN (Passive attack) [[2]](#2) .
(4) training data/label inference attack: DLG [[3]](#3), iDLG [[1]](#1), InvertGradient [[4]](#4).

In the next, we will use the example of attacking fedAvg to show the four kinds of attacks. 

### 2.2 Example of Optimization based Training Data/Label Inference Attack. 
In this attack, the server is set as the attacker performing the passive attack, i.e., when the server receives the model parameter updating information from the target client, it will perform the reconstruction procedure to find the data that generates the same parameter updating as received. Specifically, DLG method optimizes the Eqn. (4) in [[1]](#1) and InvertingGradient (IG) method optimize the Eqn. (4) in [[4]](#4) 

The knowledge that the attacker needs to obtain includes the prior knowledge provided by the users, and the knowledge (e.g., model parameter updates/gradients) obtained from FL training procedure.  

**attacker's prior knowledge:** 
1. The feature dimension of the dataset.

**Attcker's knowledge obtained from FL:**
1. The parameter updates;
2. The number of samples corresponding the received parameter updates.

#### 2.2.1 Running the attack on Femnist Example

Step 1: Set the configuration of fedavg;

Step 2:  Add the configurations of attack to the configuration in step 1;

**configuration:**

```python
attack:
  attack_method: DLG
  max_ite: 500
  reconstruct_lr: 0.1
```

The alternatives of _attack_method_ are: "DLG", "IG" which correspond to method DLG and InvertingGradient (IG) respectively. `max_ite`, `reconstruct_lr` denote the maximum iteration and the learning rate of the optimization for reconstruction, separately. 

Step 3: Run the FL with the modified configuration;

The command to run the example attack: 

```console
python flpackage/main.py --cfg flpackage/attack/example_attack_config/reconstruct_fedavg_opt_on_femnist.yaml
```

**Results on FedAvg on Femnist example:**
The recovered training images are plotted in the directory of `cfg.outdir`.

The reconstructed results at round 31 & 32 is:
![](https://img.alicdn.com/imgextra/i2/O1CN01LU3XM51aIeXHdssZg_!!6000000003307-2-tps-1820-462.png)




#### 2.2.1  Running the attack on customized dataset

Step 1: The users should make sure the FL is set up correctly on the customized dataset. 

Step 2: Add the prior knowledge about the dataset to function  `get_data_info` in `flpackage/attack/auxiliary/utils.py`. An example of femnist dataset is: 

```python
def get_data_info(dataset_name):
    if dataset_name.lower() == 'femnist':
        return [1, 28, 28], 36, False
    else:
        ValueError(
            'Please provide the data info of {}: data_feature_dim, num_class'.
            format(dataset_name))
```
The function takes the `dataset_name` as the input, and it should return `data_feature_dim`, `num_class`, `is_one_hot_label`, denoting: feature dimension, number of total classes, whether the label is represented in one-hot version, separately. 

Step 3: Run the FL with the modified configuration. 

```python
python flpackage/main.py --cfg your_own_config.yaml 
```

### 2.3 Example of Class Representative Attack

The attack module provides the class representative attack with GAN, which is the implementation of the method in [[2]](#2).

#### 2.3.1 Example attack on Femnist dataset

**Configuration:** The configuration that is added to the existing fedAve configurationn is: 
```python
attack:
  attack_method: gan_attack
  attacker_id: 5
  target_label_ind: 3
```
`attack_method` is the method's name, `attacker_id` is the id the client performs the class representative attack. `target_label_ind` is the index of the target class that the attacker wants to infer its representative samples. 

The command to run the example attack: 
```console
python flpackage/main.py --cfg flpackage/attack/example_attack_config/CRA_fedavg_convnet2_on_femnist.yaml
```


**Results:**
The results of the recovered class representatives are plotted in the directory of `cfg.outdir`.

Results on Femnist dataset with target label class "3": 
The representative samples is: 
![](https://img.alicdn.com/imgextra/i2/O1CN01OXyyFT1xbBj0sOfRw_!!6000000006461-2-tps-91-85.png)

#### 2.3.2  Running the attack on the customized dataset
To run the customized dataset, the users should define the dataset's corresponding generator function, and add it to function 'get_generator' in 'flpackage/attack/auxiliary/utils.py'.

In the example of Femnist, the generator is defined in 'flpackage/attack/models/gan_based_model.py', and the `get_generator` is: 
```python
def get_generator(dataset_name):
    if dataset_name == 'femnist':
        from flpackage.attack.models.gan_based_model import GeneratorFemnist
        return GeneratorFemnist
    else:
        ValueError("The generator to generate data like {} is not defined!".format(dataset_name))
```
It takes the `dataset_name` as the input and returns the dataset's corresponding generator. 


### 2.4 Example of Membership Inference Attack
In the Membership Inference Attack, the attacker's object is to infer whether the given target data exists in another client's private datasets. FederatedScope provides the method GradAscent, which is an active attack proposed in [[5]](#5). Specifically, GradAscent runs the gradient ascent to the gradient corresponding to the target data: 
$$W  \leftarrow W + \gamma \frac{\partial L_x}{\partial W},$$ 
where $W$ is the model parameter, $L_x$ is the loss of the target data.

If the target data $x$ in another client's training dataset, it would have a large update on the model updates, since its optimizer will abruptly reduce the gradient of $L_x$. 

#### 2.4.1 Example of MIA on Femnist dataset
The configuration of GradAscent is: 
```python
attack:
  attack_method: GradAscent
  attacker_id: 5
  inject_round: 0
```
Where `attack_method` is the attack method name, `attacker_id` is the id of the client that performs the membership inference attack, `inject_round` is the round to run the gradient ascent on the target dataset. 

The command to run the example attack: 
```console
python flpackage/main.py --cfg flpackage/attack/example_attack_config/gradient_ascent_MIA_on_femnist.yaml
```

The results of loss changes on the target data are plotted in the directory of `cfg.outdir`.

#### 2.4.2 Running the attack on a customized dataset
The users should add the way to get the target data in function `get_target_data` in `flpackage/attack/auxiliary/MIA_get_target_data.py`


### 2.5 Example of Property Inference Attack
In the property inference attack, the attacker aims to infer the property of the sample that is irrelevant to the classification task. In the server-client FL settings, the attacker is usually the server, and based on the received parameter updates, and it wants to infer whether the property exists in the batch based on which the batch owner client sends the parameter updates to the server.

#### 2.5.1 Running the attack on the synthetic dataset
Since the Femnist dataset doesn't have the additional property information, we use the synthetic dataset as the example dataset and implement the BPC, which is algorithm 3 in [[6]](#6). 

The synthetic dataset: 
* The feature x with 5 dimension is generation by $N(0,0.5)$;
* Label related:  $w_x\sim N(0, 1)$ and $b_x\sim N(0, 1)$ ; 
* Property related:  $w_p\sim N(0, 1)$ and $b_p\sim N(0, 1)$;
* Label: $y = w_xx + b_x$
* Property: $\text{prop} = \text{sigmoid}(w_px + b_p)$

The configureation of PIA: 
```python
attack:
  attack_method: PassivePIA
  classifier_PIA: svm
```
where `attack_method` is the attack method name; `classifier_PIA` is the method name of the classifer that infer the property. 

The command to run the example attack: 
```console
python flpackage/main.py --cfg flpackage/attack/example_attack_config/PIA_toy.yaml
```


## 3. Develop Your Own Attack
When developing your own attack method, it only needs to overload the class of server, client, or trainer according to the role of the attacker and the actions the attacker made in FL. Before developing your own attack, please make sure that your target FL algorithm is already set up. 

### 3.1 Add New Configuration

If your own attack method requires new hyper-parameters, they should be registered in the configuration. The details of registering your own hyper-parameter can be found in [Link to add]({{}}).


### 3.2 Server as the attacker
When the attacker is the server, it requires an overload of the Server class of the target FL algorithm.  
* If the attack happens in the FL training process, the users should rewrite the `callback_funcs_model_para`, which is the function that called when the server receives the model parameter updates. 
* If the attack happens at the end of the FL training, the users should check whether it is the last round, and if yes, it will execute the attack actions. 

After overloading the server class, you should add it the function `get_server_cls` in `flpackage/core/auxiliaries/worker_builder.py` .

The following is an example in a property inference attack where the attacker (server) performs attack actions both during and after the FL training. The attacker collects the received parameter updates during the FL training and generates the training data for PIA classifier based on the current model. After the FL training, the attacker trains the PIA classifier, and then infers the property based on the collected parameter updates. 

```python
    def callback_funcs_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        # For a new round
        if round not in self.msg_buffer['train'].keys():
            self.msg_buffer['train'][round] = dict()

        self.msg_buffer['train'][round][sender] = content

        # collect the updates
        self.pia_attacker.collect_updates(previous_para= self.model.state_dict(),
                                          updated_parameter=content[1],
                                          round=round,
                                          client_id=sender)
        self.pia_attacker.get_data_for_dataset_prop_classifier(model=self.model)

        if self._cfg.federate.online_aggr:
            # TODO: put this line to `check_and_move_on`
            # currently, no way to know the latest `sender`
            self.aggregator.inc(content)
        self.check_and_move_on()

        if self.state == self.total_round_num:
            self.pia_attacker.train_property_classifier()
            self.pia_results = self.pia_attacker.infer_collected()
            print(self.pia_results)
```

### 3.3 Client as the attacker
When the attacker is one of the clients:
* If the attack actions only happen in the local training procedure, users only need to define the wrapping function to wrap the trainer and add the attack actions. 
* If the attack actions also happen at the end of the FL training, users need to overload the client class and modify its `callback_funcs_for_finish` function. 

After setting the trainer wrapping function, it should be added to the function `get_trainer` in `flpackage/core/auxiliaries/trainer_builder.py`. Similarly, after overloading the client class, it should be added to function `get_client_cls` in `flpackage/core/auxiliaries/worker_builder.py`.

The following is an example of wrapping the trainer in the implementation of the class representative attack in [[2]](#2). In this method, the attacker holds a local GAN, and at each FL training round, it will first update the GAN's discriminator with the received paramters, and then local trains GAN's generator so that its generated data can be classified as the target class. After that, the client labels the generated data as the class other than the target class and injects them into the training batch to perform the regular local training. 

The following code shows the wrapping function that adds the above procedures to the trainer. The wrapping function takes the trainer instance as the input. 
It first adds an instance of `GANCRA` class which is the GAN attack class defined in `flpackage/attack/models/gan_based_model.py` to the context of the trainer, i.e., `base_trainer.ctx` . Then it registers different hooks into its corresponding phase: 
* Register the hook `hood_on_fit_start_generator` to the phase'on_fit_start`, so that the trainer will update the round number at the beginning of local training in each round. 
* Register the hook `hook_on_gan_cra_train` and `hook_on_batch_injected_data_generation` to the phase `on_batch_start`, so that the trainer will train the GAN and inject the generated data to the training batch when preparing the train batch for local training. 
* Register the hook `hook_on_data_injection_sav_data` to the phase `on_fit_end` , so that at the end of local training, it will save the data generated by GAN. 

```python
def wrap_GANTrainer(
        base_trainer: Type[GeneralTrainer]) -> Type[GeneralTrainer]:

    # ---------------- attribute-level plug-in -----------------------

    base_trainer.ctx.target_label_ind = base_trainer.cfg.attack.target_label_ind
    base_trainer.ctx.gan_cra = GANCRA(base_trainer.cfg.attack.target_label_ind,
                                      base_trainer.ctx.model,
                                      dataset_name = base_trainer.cfg.data.type,
                                      device=base_trainer.ctx.device,
                                      sav_pth=base_trainer.cfg.outdir
                                      )

    # ---- action-level plug-in -------

    base_trainer.register_hook_in_train(new_hook=hood_on_fit_start_generator,
                                        trigger='on_fit_start',
                                        insert_mode=-1)
    base_trainer.register_hook_in_train(new_hook=hook_on_gan_cra_train,
                                        trigger='on_batch_start',
                                        insert_mode=-1)
    base_trainer.register_hook_in_train(
        new_hook=hook_on_batch_injected_data_generation,
        trigger='on_batch_start',
        insert_mode=-1)
    base_trainer.register_hook_in_train(
        new_hook=hook_on_batch_forward_injected_data,
        trigger='on_batch_forward',
        insert_mode=-1)

    base_trainer.register_hook_in_train(
        new_hook=hook_on_data_injection_sav_data,
        trigger='on_fit_end',
        insert_mode=-1)

    return base_trainer


def hood_on_fit_start_generator(ctx):
  # update the round number
    ctx.gan_cra.round_num += 1
    print('----- Round {}: GAN training ............'.format(
        ctx.gan_cra.round_num))


def hook_on_batch_forward_injected_data(ctx):
    # inject the generated data into training batch loss
    x, label = [_.to(ctx.device) for _ in ctx.data_batch]
    pred = ctx.model(x)
    if len(label.size()) == 0:
        label = label.unsqueeze(0)
    ctx.loss_batch += ctx.criterion(pred, label)
    ctx.y_true_injected = label
    ctx.y_prob_injected = pred


def hook_on_batch_injected_data_generation(ctx):
    # generate the injected data
    ctx.injected_data = ctx.gan_cra.generate_fake_data()


def hook_on_gan_cra_train(ctx):
   # update the GAN's discriminator with the broadcasted parameter
    ctx.gan_cra.update_discriminator(ctx.model)
   # train the GAN's generator
    ctx.gan_cra.generator_train()


def hook_on_data_injection_sav_data(ctx):
    ctx.gan_cra.generate_and_save_images()
```



## 4. Contribute Your Attacker to FederatedScope
Users are welcome to contribute their own attack methods to FederatedScope. Please refer [Contributing to FederatedScope]({{ "/docs/contributor/" | relative_url }}) for more details. 






***
## Reference
<a id="1">[1]</a>  Zhao B, Mopuri K R, Bilen H. idlg: Improved deep leakage from gradients[J]. arXiv preprint arXiv:2001.02610, 2020.

<a id="2">[2]</a>  Hitaj, Briland, Giuseppe Ateniese, and Fernando Perez-Cruz. "Deep models under the GAN: information leakage from collaborative deep learning." Proceedings of the 2017 ACM SIGSAC conference on computer and communications security. 2017.

<a id="3">[3]</a> Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in Neural Information Processing Systems 32 (2019).

<a id= "4">[4]</a> Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?." Advances in Neural Information Processing Systems 33 (2020): 16937-16947.

<a id="5">[5]</a> Nasr, Milad, R. Shokri and Amir Houmansadr. "Comprehensive Privacy Analysis of Deep Learning: Stand-alone and Federated Learning under Passive and Active White-box Inference Attacks." ArXiv abs/1812.00910 (2018): n. pag.

<a id= "6">[6]</a> Melis, Luca, Congzheng Song, Emiliano De Cristofaro and Vitaly Shmatikov. "Exploiting Unintended Feature Leakage in Collaborative Learning." 2019 IEEE Symposium on Security and Privacy (SP) (2019): 691-706.

<a id="7">[7]</a> Lyu, Lingjuan, Han Yu and Qiang Yang. "Threats to Federated Learning: A Survey." ArXiv abs/2003.02133 (2020): n. pag.
