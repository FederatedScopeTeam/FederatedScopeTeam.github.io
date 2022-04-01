---
title: "Recommendation"
permalink: /docs/recommendation/
excerpt: "Federated Recommendation."
last_modified_at: 2018-03-20T15:59:57-04:00
toc: true
---

<a name="U7cD5"></a>
## Matrix Factorization
FederatedScope has built in the matrix factorization (MF) task for recommendation, which provides flexible supports for MF models, datasets and federated settings. In this tutorial, we will introduce 

- the supports of MF tasks, 
- how to implement matrix factorization task with FederatedScope, and
- the privacy preserving techniques used in FederatedScope.
<a name="L9eHO"></a>
### Background
Matrix factorization (MF) [1-3] is a fundamental building block in recommendation system. For a matrix, a row corresponds to a user, while a column corresponds to an item. The target of matrix factorization is to approximate unobserved ratings by constructing user embedding $U\in{\mathbb{R}^{n\times{d}}}$and item embedding $V\in{\mathbb{R}^{m\times{d}}}$. <br /> ![mf_task.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/9556273/1648197184171-dca7f204-192f-424d-9794-52e8bf0d195c.png#clientId=u9a669a0e-153f-4&crop=0&crop=0&crop=1&crop=1&from=ui&height=200&id=udb15df61&margin=%5Bobject%20Object%5D&name=mf_task.png&originHeight=368&originWidth=1031&originalType=binary&ratio=1&rotation=0&showTitle=false&size=16721&status=done&style=none&taskId=udbf3570a-b126-41e7-bb07-0479a3c4cf7&title=&width=559)<br />Supposing$X\in{\mathbb{R}^{n\times{m}}}$is the target rating matrix, the target is formulized as minimizing the loss function $\mathcal{L}(X,U,V)$:<br />$\frac{1}{|\Omega|}\sum_{(i,j)\in\Omega}\mathcal{L}_{i,j}(X,U,V)=\frac{1}{|\Omega|}\sum_{(i,j)\in\Omega}(X_{i,j}-<u_i,v_j>)^2$<br />where$u_i\in{\mathbb{R}^{n\times1}}$and $v_j\in{\mathbb{R}^{m\times1}}$are the user and item vectors of $U$and $V$.
<a name="SCGUt"></a>
### MF in Federated Learning
In federated learning, the dataset is distributed in different clients. The vanilla federated matrix factorization algorithm runs as follows

- Step1: Server initializes shared parameters
- Step2: Server broadcasts shared parameters to all participators
- Step3: Each participator updates their parameters locally
- Step4: Participators upload their shared parameters to the server
- Step5: Server aggregates the received parameters and repeat Step2 until the training is finished

With different data partitions, matrix factorization has three FL settings: **Vertical FL**(VFL), **Horizontal FL**(HFL) and **Local FL**(LFL). 

<a name="AcF3D"></a>
#### Vertical FL
In VFL, the set of users is the same across different databases, and each participators only has partial items. In this setting, the user embedding is shared across all participators and each client maintains its own item embedding.  <br />![VFL setting [3]](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/9556273/1647842481181-dd8bb1bf-44f2-47a9-aac7-e7cc7226d258.png#crop=0&crop=0&crop=1&crop=1&from=url&height=262&id=Gb7zt&margin=%5Bobject%20Object%5D&originHeight=600&originWidth=732&originalType=binary&ratio=1&rotation=0&showTitle=true&status=done&style=none&title=VFL%20setting%20%5B3%5D&width=320 "VFL setting [3]")
<a name="R33VJ"></a>
#### Horizontal FL
In HFL, the set of items is the same across different participators, and they only share the item embedding with the coordination server. <br />![截屏2022-03-21 下午2.03.06.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/9556273/1648018555853-f80b0128-83c6-4acd-9d41-a63571053cbf.png#clientId=u2fd8d2dc-4f3e-4&crop=0&crop=0&crop=1&crop=0.8809&from=ui&height=314&id=uc6843663&margin=%5Bobject%20Object%5D&name=%E6%88%AA%E5%B1%8F2022-03-21%20%E4%B8%8B%E5%8D%882.03.06.png&originHeight=638&originWidth=650&originalType=binary&ratio=1&rotation=0&showTitle=true&size=139863&status=done&style=none&taskId=u058a337e-fc56-4087-863a-957ce056aeb&title=HFL%20setting%20%5B3%5D&width=320 "HFL setting [3]")
<a name="pgOAT"></a>
#### Local FL
LFL is a special case of HFL, where each user owns her/his own ratings. It's a common scenario on mobile devices. <br />![截屏2022-03-23 下午2.59.34.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/9556273/1648018785541-5623924c-ef1f-49f4-9ce5-025bb46cb970.png#clientId=u2fd8d2dc-4f3e-4&crop=0&crop=0&crop=1&crop=1&from=ui&height=266&id=u3e77b969&margin=%5Bobject%20Object%5D&name=%E6%88%AA%E5%B1%8F2022-03-23%20%E4%B8%8B%E5%8D%882.59.34.png&originHeight=652&originWidth=784&originalType=binary&ratio=1&rotation=0&showTitle=true&size=110371&status=done&style=none&taskId=uf480945a-2c26-4d93-912b-909acadcc62&title=LFL%20setting%20%5B3%5D&width=320 "LFL setting [3]")
<a name="vAW11"></a>
### Support of MF
To support federated MF, FederatedScope builds in MF models, datasets and trainer in different federated learning settings. 

<a name="b1IUy"></a>
#### MF models
MF model has two trainable parameters: user embedding and item embedding. Based on the given federated setting, they share different embedding with the other participators. FederatedScope achieves `VMFNet`and `HMFNet`to support the settings of VFL and HFL.
```python
class VMFNet(BasicMFNet):
    name_reserve = "embed_item"


class HMFNet(BasicMFNet):
    name_reserve = "embed_user"
```
The attribute `name_reserve`specifics the name of local embedding vector, and the parent class`BasicMFNet`defines the common actions, including 

- load/fetch parameters, and
- forward propagation.

Note the rating matrix is usually very sparse. To impove the efficiency, FederatedScope creates the predicted matrix and the target rating matrix as sparse tensors.
```python
class BasicMFNet(Module):
    ...
    def forward(self, indices, ratings):
        pred = torch.matmul(self.embed_user, self.embed_item.T)
        label = torch.sparse_coo_tensor(indices,
                                        ratings,
                                        size=pred.shape,
                                        device=pred.device,
                                        dtype=torch.float32).to_dense()
        mask = torch.sparse_coo_tensor(indices,
                                       np.ones(len(ratings)),
                                       size=pred.shape,
                                       device=pred.device,
                                       dtype=torch.float32).to_dense()

        return mask * pred, label, float(np.prod(pred.size())) / len(ratings)
    ...
```

<a name="RwWMo"></a>
#### MF Datasets
MovieLens is series of movie recommendation datasets collected from the website [MovieLens](https://movielens.org). <br />To satisify the requirement of different FL settings, FederatedScope splits the dataset into `VFLMoviesLens`and `HFLMovieLens`as follows. For example, if your want to use the dataset MovieLens1M in VFL settings, just set `cfg.data.type='VFLMovieLens1M'`.
```python
class VFLMovieLens1M(MovieLens1M, VMFDataset):
    """MovieLens1M dataset in VFL setting
    
    """
    pass


class HFLMovieLens1M(MovieLens1M, HMFDataset):
    """MovieLens1M dataset in HFL setting

    """
    pass


class VFLMovieLens10M(MovieLens10M, VMFDataset):
    """MovieLens10M dataset in VFL setting

    """
    pass


class HFLMovieLens10M(MovieLens10M, HMFDataset):
    """MovieLens10M dataset in HFL setting

    """
    pass
```

The parent classes of the above datasets define the data information and the FL setting respectively.
<a name="rFhUC"></a>
#### 
<a name="NGR3t"></a>
##### Data information
The first parent class `MovieLens1M` and `MovieLens10M`provide the details (e.g. url, md5, filename).
```python
class MovieLens1M(MovieLensData):
    """MoviesLens 1M Dataset
    (https://grouplens.org/datasets/movielens)

    Format:
        UserID::MovieID::Rating::Timestamp

    Arguments:
        root (str): Root directory of dataset where directory
            ``MoviesLen1M`` exists or will be saved to if download is set to True.
        config (callable): Parameters related to matrix factorization.
        train_size (float, optional): The proportion of training data.
        test_size (float, optional): The proportion of test data.
        download  (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'MovieLens1M'
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    filename = "ml-1m"
    zip_md5 = "c4d9eecfca2ab87c1945afe126590906"
    raw_file = "ratings.dat"
    raw_file_md5 = "a89aa3591bc97d6d4e0c89459ff39362"


class MovieLens10M(MovieLensData):
    """MoviesLens 10M Dataset
    (https://grouplens.org/datasets/movielens)

    Format:
        UserID::MovieID::Rating::Timestamp

    Arguments:
        root (str): Root directory of dataset where directory
            ``MoviesLen1M`` exists or will be saved to if download is set to True.
        config (callable): Parameters related to matrix factorization.
        train_size (float, optional): The proportion of training data.
        test_size (float, optional): The proportion of test data.
        download  (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'MovieLens10M'
    url = "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
    filename = "ml-10M100K"

    zip_md5 = "ce571fd55effeba0271552578f2648bd"
    raw_file = "ratings.dat"
    raw_file_md5 = "3f317698625386f66177629fa5c6b2dc"
```
<a name="PXyQL"></a>
#### 
<a name="sxniP"></a>
##### FL Setting
`VMFDataset`and `HMFDataset`specific the spliting of MF datasets (VFL or HFL).
```python
class VMFDataset:
    """Dataset of matrix factorization task in vertical federated learning.

    """
    def _split_n_clients_rating(self, ratings: csc_matrix, num_client: int,
                                test_portion: float):
        id_item = np.arange(self.n_item)
        shuffle(id_item)
        items_per_client = np.array_split(id_item, num_client)
        data = dict()
        for clientId, items in enumerate(items_per_client):
            client_ratings = ratings[:, items]
            train_ratings, test_ratings = self._split_train_test_ratings(
                client_ratings, test_portion)
            data[clientId + 1] = {"train": train_ratings, "test": test_ratings}
        self.data = data


class HMFDataset:
    """Dataset of matrix factorization task in horizontal federated learning.

    """
    def _split_n_clients_rating(self, ratings: csc_matrix, num_client: int,
                                test_portion: float):
        id_user = np.arange(self.n_user)
        shuffle(id_user)
        users_per_client = np.array_split(id_user, num_client)
        data = dict()
        for cliendId, users in enumerate(users_per_client):
            client_ratings = ratings[users, :]
            train_ratings, test_ratings = self._split_train_test_ratings(
                client_ratings, test_portion)
            data[cliendId + 1] = {"train": train_ratings, "test": test_ratings}
        self.data = data
```

<a name="XAlhf"></a>
#### MF Trainer
Considering the target rating matrix is large and sparse, FederatedScope achieves`MFTrainer`to support MF tasks during federated training.
```python
class MFTrainer(GeneralTrainer):
    """
    model (torch.nn.module): MF model.
    data (dict): input data
    device (str): device.
    """

    def _hook_on_fit_end(self, ctx):
        results = {
            "{}_avg_loss".format(ctx.cur_mode): ctx.get("loss_batch_total_{}".format(ctx.cur_mode)) /
            ctx.get("num_samples_{}".format(ctx.cur_mode)),
            "{}_total".format(ctx.cur_mode): ctx.get("num_samples_{}".format(ctx.cur_mode))
        }
        setattr(ctx, 'eval_metrics', results)

    def _hook_on_batch_end(self, ctx):
        # update statistics
        setattr(
            ctx, "loss_batch_total_{}".format(ctx.cur_mode),
            ctx.get("loss_batch_total_{}".format(ctx.cur_mode)) +
            ctx.loss_batch.item() * ctx.batch_size)

        if ctx.get("loss_regular", None) is None or ctx.loss_regular == 0:
            loss_regular = 0.
        else:
            loss_regular = ctx.loss_regular.item()
        setattr(
            ctx, "loss_regular_total_{}".format(ctx.cur_mode),
            ctx.get("loss_regular_total_{}".format(ctx.cur_mode)) +
            loss_regular)
        setattr(
            ctx, "num_samples_{}".format(ctx.cur_mode),
            ctx.get("num_samples_{}".format(ctx.cur_mode)) + ctx.batch_size)

        # clean temp ctx
        ctx.data_batch = None
        ctx.batch_size = None
        ctx.loss_task = None
        ctx.loss_batch = None
        ctx.loss_regular = None
        ctx.y_true = None
        ctx.y_prob = None

    def _hook_on_batch_forward(self, ctx):
        indices, ratings = ctx.data_batch
        pred, label, ratio = ctx.model(indices, ratings)
        ctx.loss_batch = ctx.criterion(pred, label) * ratio

        ctx.batch_size = len(ratings)
```

<a name="bd3A7"></a>
### Start an Example
Taking the combination of dataset `MovieLen1M`and VFL setting as an example, the running command is as follows.
```bash
python main.py --cfg flpackage/mf/baseline/fedavg_vfl_fedavg_standalone_on_movielens1m.yaml
```

More running scripts can be found in `flpackage/scripts`. Partial experimental results are shown as follows.

| Federated setting | Dataset | Number of clients | Loss |
| --- | --- | --- | --- |
| VFL | MovieLens1M | 5 | 1.16 |
| HFL | MovieLens1M | 5 | 1.13 |


<a name="Yb6U1"></a>
### Privacy Protection
To protect the user privacy, FederatedScope implements two differential privacy algorithms, VFL-SGDMF and HFL-SGDMF in [vldb22] as plug-ins. 
<a name="JdvEb"></a>
#### VFL-SGDMF
VFL-SGDMF is a DP based algorithm for privacy preserving in VFL setting. It satisifies $(\epsilon-\delta)$privacy by injecting noise into the embedding matrix. More details please refer to [3]. The related parameters are shown as follows.
```python
# ------------------------------------------------------------------------ #
# VFL-SGDMF(dp) related options
# ------------------------------------------------------------------------ #
cfg.sgdmf = CN()

cfg.sgdmf.use = False    # if use sgdmf
cfg.sgdmf.R = 5.         # The upper bound of rating
cfg.sgdmf.epsilon = 4.   # \epsilon in dp
cfg.sgdmf.delta = 0.5    # \delta in dp
cfg.sgdmf.constant = 1.  # constant
cfg.sgdmf.theta = -1     # -1 means per-rating privacy, otherwise per-user privacy
```

VFL-SGDMF is implemented as plug-in in `flpackage/mf/trainer/trainer_sgdmf.py`. Similar with the other plug-in algorithms, it initializes and registers hook functions in the function`wrap_MFTrainer`.
```python
def wrap_MFTrainer(base_trainer: Type[MFTrainer]) -> Type[MFTrainer]:
    """Build `SGDMFTrainer` with a plug-in manner, by registering new functions into specific `MFTrainer`

    """

    # ---------------- attribute-level plug-in -----------------------
    init_sgdmf_ctx(base_trainer)

    # ---------------- action-level plug-in -----------------------
    base_trainer.replace_hook_in_train(new_hook=hook_on_batch_backward,
                                       target_trigger="on_batch_backward",
                                       target_hook_name="_hook_on_batch_backward")

    return base_trainer
...
```
The embedding clipping and noise injection is finished in the new hook function `hook_on_batch_backward`.
```python
def hook_on_batch_backward(ctx):
    """Private local updates in SGDMF

    """
    ctx.optimizer.zero_grad()
    ctx.loss_task.backward()
    
    # Inject noise
    ctx.model.embed_user.grad.data += get_random(
        "Normal",
        sample_shape=ctx.model.embed_user.shape,
        params={
            "loc": 0,
            "scale": ctx.scale
        },
        device=ctx.model.embed_user.device)
    ctx.model.embed_item.grad.data += get_random(
        "Normal",
        sample_shape=ctx.model.embed_item.shape,
        params={
            "loc": 0,
            "scale": ctx.scale
        },
        device=ctx.model.embed_item.device)
    ctx.optimizer.step()

    # Embedding clipping
    with torch.no_grad():
        embedding_clip(ctx.model.embed_user, ctx.sgdmf_R)
        embedding_clip(ctx.model.embed_item, ctx.sgdmf_R)
```
<a name="LeypM"></a>
#### 
<a name="pZEVg"></a>
##### Start an Example
Similarly, taking `MovieLens1M` as an example, the running script is shown as follows. 
```bash
python flpackage/main.py --cfg flpackage/mf/baseline/vfl-sgdmf_fedavg_standalone_on_movielens1m.yaml
```
<a name="xwxxt"></a>
#### 
<a name="ow9gP"></a>
##### Evaluation
Take the dataset `MovieLens1M` as an example, the detailed settings are listed in `flpackage/mf/baseline/vfl_fedavg_standalone_on_movielens1m.yaml` and `flpackage/mf/baseline/vfl-sgdmf_fedavg_standalone_on_movielens1m.yaml`. VFL-SGDMF is evaluated as follows.

| Algo | $\epsilon$ | $\delta$ | Loss |
| --- | --- | --- | --- |
| VFL | - | - | 1.16 |
| VFL-SGDMF | 4 | 0.75 | 1.47 |
|  | 4 | 0.25 | 1.54 |
|  | 2 | 0.75 | 1.55 |
|  | 2 | 0.25 | 1.56 |
|  | 0.5 | 0.75 | 1.68 |
|  | 0.5 | 0.25 | 1.84 |



<a name="uVVYN"></a>
#### HFL-SGDMF
On the other side,  HFL-SGDMF protects privacy in HFL setting in the same way, and share the same parameters with VFL-SGDMF.
```python
# ------------------------------------------------------------------------ #
# VFL-SGDMF(dp) related options
# ------------------------------------------------------------------------ #
cfg.sgdmf = CN()

cfg.sgdmf.use = False    # if use sgdmf
cfg.sgdmf.R = 5.         # The upper bound of rating
cfg.sgdmf.epsilon = 4.   # \epsilon in dp
cfg.sgdmf.delta = 0.5    # \delta in dp
cfg.sgdmf.constant = 1.  # constant
cfg.sgdmf.theta = -1     # -1 means per-rating privacy, otherwise per-user privacy
```


<a name="uqrq4"></a>
##### Start and Example
Run an example of HFL-SGDMF by the following command. 
```bash
python flpackage/main.py --cfg flpackage/mf/baseline/hfl-sgdmf_fedavg_standalone_on_movielens1m.yaml
```

<a name="GjjQv"></a>
##### Evaluation
The evaluation results of HFL-SGDMF on the dataset MovieLens1M are shown as follows. 

| Algo | $\epsilon$ | $\delta$ | Loss |
| --- | --- | --- | --- |
| HFL | - | - | 1.13 |
| HFL-SGDMF | 4 | 0.75 | 1.56 |
|  | 4 | 0.25 | 1.62 |
|  | 2 | 0.75 | 1.60 |
|  | 2 | 0.25 | 1.64 |
|  | 0.5 | 0.75 | 1.66 |
|  | 0.5 | 0.25 | 1.73 |


---

<a name="DHOTy"></a>
### Reference
[1] Hao Ma, Haixuan Yang, Michael R. Lyu, et al. SoRec: social recommendation using probabilistic matrix factorization. Proceedings of the ACM Conference on Information and Knowledge Management, 2008.<br />[2] Mohsen Jamali, Martin Ester. A matrix factorization technique with trust propagation for recommendation in social networks. Proceedings of the ACM Conference on Recommender Systems, 2010.<br />[3] Zitao Li, Bolin Ding, Ce Zhang, et al. Federated Matrix Factorization with Privacy Guarantee. Proceedings of the VLDB Endowment, 2022.
