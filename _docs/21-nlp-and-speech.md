---
title: "NLP and Speech"
permalink: /docs/nlp-and-speech/
excerpt: "About NLP and Speech"
last_modified_at: 2018-11-25T19:46:43-05:00
toc: true
layout: tuto
---

<a name="bJdWk"></a>
## Background

Recently, privacy-preserving methods gain increasing attentions in machine learning (ML) applications using linguistic data  including text and audio, due to the fact that linguistic data can involve a wealth of information relating to an identified or identifiable natural person, such as the physiological, psychological, economic, cultural or social identity.

Federated Learning (FL) methods show promising results for collaboratively training models from a large number of clients without sharing their private linguistic data. To facilitate FL research in linguistic data, FederatedScope provides several built-in linguistic datasets and supports various tasks such as language modeling and text classification with various FL algorithms.

<a name="Yzwrs"></a>
## Natural Language Processing (NLP)

<a name="ZhStB"></a>
### Datasets

We provide three popular text datasets for next-character prediction, next-word prediction, and sentiment analysis.

- Shakespeare: a federation text dataset of Shakespeare Dialogues from [LEAF](https://leaf.cmu.edu/) [1] for next-character prediction, which contains 422,615 sentences and about 1,100 clients.
- subReddit: a federation text dataset and subsampled of reddit from LEAF for next-word prediction, which contains 216,858 sentences and about 800 clients.
- Sentiment140: a federation text dataset of Twitter from LEAF for Sentiment Analysis, which contains 1,600,498 sentences, about 660,000  clients.

<a name="eHVuQ"></a>
### Models

We provide a LSTM model implementation in `federatedscope/nlp/model`

- **LSTM:** a type of RNN that solves the vanishing gradient problem through additional cells, input and output gates. (`cfg.model.type = 'lstm'`)

```python
class LSTM(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden,
                 out_channels,
                 n_layers=2,
                 embed_size=8):
        pass
```

- Currently, we are working on implement more interfaces to support more popular NLP Transformer models and more NLP tasks with [HuggingFace Transformers](https://github.com/huggingface/transformers) [2].

<a name="nOGSF"></a>
### Start an example

Next-character/word prediction is a classic NLP task as it can be applied in many consumer applications and appropriately be modeled by statistical language models, we show how to achieve next-character prediction in cross-device FL setting.

- Here we implement a simple LSTM model for next-character prediction: taking an English characters sequence as input, the model learns to predict the next possible character. After registering the model, we can use it by specifying `cfg.model.type=lstm` and  hyper-parameters such as  `cfg.model.in_channels=80, cfg.model.out_channels=80, cfg.model.emd_size=8`.  Complete codes are in `federatedscope/nlp/model/rnn.py` and `federatedscope/nlp/model/model_builder.py`.

```python
class LSTM(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden,
                 out_channels,
                 n_layers=2,
                 embed_size=8):
        super(LSTM, self).__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.embed_size = embed_size
        self.out_channels = out_channels
        self.n_layers = n_layers

        self.encoder = nn.Embedding(in_channels, embed_size)

        self.rnn =\
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden, out_channels)

    def forward(self, input_):
        encoded = self.encoder(input_)
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        final_word = output[:, :, -1]
        return final_word
```

- For the dataset, we use the Shakespeare dataset from [LEAF](https://leaf.cmu.edu/), which is built from _The Complete Works of William Shakespeare_,  and partitioned to ~1100 clients (speaking roles) from 422615.  We can specify the `cfg.dataset.type=shakespeare` and adjust the fraction of data subsample (`cfg.data.sub_sample=0.2`), and train/val/test ratio (``cfg.data.splits=[0.6,0.2,0.2]). Complete NLP data codes are in`federatedscope/nlp/dataset`.

```python
class LEAF_NLP(LEAF):
    """
    LEAF NLP dataset from
    
    leaf.cmu.edu
    
    self:
        root (str): root path.
        name (str): name of dataset, ‘shakespeare’ or ‘xxx’.
        s_frac (float): fraction of the dataset to be used; default=0.3.
        tr_frac (float): train set proportion for each task; default=0.8.
        val_frac (float): valid set proportion for each task; default=0.0.
    """
    def __init__(
            self,
            root,
            name,
            s_frac=0.3, 
            tr_frac=0.8,
            val_frac=0.0,
            seed=123,
            transform=None,
            target_transform=None):
        pass
```

- To enable large-scale clients simulation, we provide online aggregator in standalone mode to save the memory, which  maintains only three model objects for the FL server aggregation. We can use this feature by specifying `cfg.federate.online_aggr = True` and `federate.share_local_model=True` , more details about this feature can be found in [the post "Simulation and Deployment"]({{ "/docs/simulation-and-deployment" | relative_url }}).
- To handle the non-i.i.d. challenge, FederatedScope supports several SOTA [personalization]({{ "/docs/pfl" | relative_url }}) algorithms and easy extension.
- To enable partial clients participation in each FL round, we provide clients sampling feature with various configuration manners: 1) `cfg.federate.sample_client_rate`, which is in the range (0, 1] and indicates selecting partial clients using random sampling with replacement; 2) `cfg.federate.sample_client_num` , which is an integer to indicate sample client number at each round.

With these specification, we can run the experiment with

```python
 main.py --cfg federatedscope/nlp/baseline/fedavg_lstm_on_shakespeare.yaml
```

You will get the accuracy of FedAvg algorithm around `43.80%`.

Other NLP related scripts to run the next-character prediction experiments can be found in `federatedscope/nlp/baseline`.

<a name="MK5wA"></a>
### Customize your NLP task

FederatedScope enables users to easily implement and register more NLP datasets and models.

-  Implement and register your own NLP data 
```python
# federatedscope/contrib/data/my_nlp_data.py

import torch
import copy
import numpy as np

from federatedscope.register import register_data

def get_my_nlp_data(config):
    r"""
        This function returns a dictionary, where key is the client id and 
    value is the data dict of each client with 'train', 'test' or 'val'.
    		NOTE: client_id 0 is SERVER!
    
    Returns:
          dict: {
                    'client_id': {
                        'train': DataLoader or Data,
                        'test': DataLoader or Data,
                        'val': DataLoader or Data,
                    }
                }
    """
    import numpy as np
    from torch.utils.data import DataLoader

    # Build data
    dataset = LEAF_NLP(root=path,
                       name="twitter",
                       s_frac=config.data.subsample,
                       tr_frac=splits[0],
                       val_frac=splits[1],
                       seed=1234,
                       transform=transform)
    
    client_num = min(len(dataset), config.federate.client_num
                     ) if config.federate.client_num > 0 else len(dataset)
    config.merge_from_list(['federate.client_num', client_num])

    # get local dataset
    data_local_dict = dict()
    for client_idx in range(client_num):
        dataloader = {
            'train': DataLoader(dataset[client_idx]['train'],
                                batch_size,
                                shuffle=config.data.shuffle,
                                num_workers=config.data.num_workers),
            'test': DataLoader(dataset[client_idx]['test'],
                               batch_size,
                               shuffle=False,
                               num_workers=config.data.num_workers)
        }
        if 'val' in dataset[client_idx]:
            dataloader['val'] = DataLoader(dataset[client_idx]['val'],
                                           batch_size,
                                           shuffle=False,
                                           num_workers=config.data.num_workers)

        data_local_dict[client_idx + 1] = dataloader

    return data_local_dict, confi

def call_my_data(config):
    if config.data.type == "my_nlp_data":
        data, modified_config = get_my_nlp_data(config)
        return data, modified_config

register_data("my_nlp_data", call_my_data)
```
 

-  Implement and register your own NLP model 
```python
import torch
from federatedscope.register import register_model


class  KIM_CNN(nn.Module):
  """
  		ref to Kim's CNN text classification paper [3]
  		https://github.com/Shawn1993/cnn-text-classification-pytorch
  """
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        if self.args.static:
            self.embed.weight.requires_grad = False

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
    
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit

     
def call_my_net(model_config, local_data):
    if model_config.type == "my_nlp_model":
        model = KIM_CNN(args=model_config)
        return model

register_model("my_nlp_model", call_my_net)
```
 

-  Then with fruitful  built-in FL experiments scripts , users can run own FL experiments by _replacing_ the model type and dataset type in the provided  scripts. 

<a name="Txa84"></a>
## Speech (Coming soon)

We are working on implement more interfaces to support more Conformer [4] models and more speech-related tasks with [WeNet](https://github.com/wenet-e2e/wenet) [5], which is designed for various end-2-end speech recognition tasks and provides full stack solutions for production and real-world applications.

<a name="Reference"></a>
## Reference

[1] Caldas, Sebastian, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Konečný, H. Brendan McMahan, Virginia Smith, and Ameet Talwalkar. "Leaf: A benchmark for federated settings." _arXiv preprint arXiv:1812.01097_ (2018).

[2] Wolf, Thomas, et al. "Huggingface's transformers: State-of-the-art natural language processing." _arXiv preprint arXiv:1910.03771_ (2019).

[3] Yoon Kim. 2014. [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181). In _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)_, pages 1746–1751, Doha, Qatar.

[4] Gulati, Anmol, et al. "Conformer: Convolution-augmented transformer for speech recognition." _arXiv preprint arXiv:2005.08100_ (2020).

[5] Zhang, Binbin, et al. "Wenet: Production first and production ready end-to-end speech recognition toolkit." _arXiv e-prints_ (2021): arXiv-2102.
