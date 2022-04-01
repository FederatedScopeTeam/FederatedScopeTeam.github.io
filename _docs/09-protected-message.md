---
title: "Privacy Protection for Message"
permalink: /docs/protected-msg/
excerpt: "About protecting messages."
last_modified_at: 2018-09-10T12:33:24-04:00
toc: true
---

In order to satisfy different requirements of message privacy protection,  several technologies, including Differential Privacy, Encryption, Multi-Party Computation, etc., are applied in an FL course to enhance the strength of protection.<br />Here we give a brief introduction of how to use these technologies in FederatedScope.
<a name="pHbdx"></a>
## Differential Privacy
Differential privacy (DP) is a powerful theoretical criterion for privacy. A mechanism satisfying $(\epsilon-\delta)$differential privacy promises the indistinguishability of similar information with the possbility $1-\delta$. <br />As a message-orented federated learning framework, FederatedScope usually exchanges messages at high frenquency. Differential privacy provides a powerful theoretical metric to evaluate the balance between training performance and privacy protection. 

To protect the privacy contained in the message, FederatedScope provides flexible supports for differential privacy:

- **Abundant benchmark datasets and tasks** are preset in FederatedScope, e.g., cv, graph, nlp and recommendation tasks. It's quite convenient and simple to conduct DP evaluations under different settings;
- **Build-in DP algorithms** are user-friendly for the beginners; 
- **Fexiable DP APIs** are preset for users to develop and implement their own DP algorithms; 
- **Rich attackers** are implemented in FederatedScope. It's convenient to test your DP algorithm.

<a name="DOR4F"></a>
## Encryption
One of the techniques to protect privacy for messages is encryption. Before sending messages to others, participants can apply encryption algorithms to transfer the plaintext to ciphertext, which prevents privacy leakage during the communication. 

When receiving the ciphertext, the receiver usually needs to recover the plaintext via decryption. However, if homomorphic encryption algorithms, such as Paillier [1], are adopted, the receiver is allowed to perform (limited) operations on the ciphertext, which ensures the correctness of operations as well as prevents the receiver from being aware of the plaintext. <br />Readers can refer to xx for an example of using homomorphic encryption algorithms in an FL course. Note that developers can extend more encryption algorithms in FederatedScope without caring about other modules since they have been decoupled with each other.

The biggest weakness of applying encryption in FL is bringing additional costs for both computation and communication. It might be intolerable in an FL course, especially for cross-device scenarios that involve IoT devices or smartphones that have limited computation resources and communication width.

<a name="EpuAp"></a>
## Secure Multi-Party Computation
Secure Multi-Party Computation (SMPC) aims to jointly compute a function by multiple participants while keeping the original inputs private. Formally, given $n$participants and each participant$i$owns the private data$x_i$. SMPC proposes to learn $y=f(x_1, x_2, ..., x_n)$without exposing the values of $x_1, x_2, ..., x_n$to other participants.<br />We show an example of additive secret sharing in the figure with instantiating $f$as MEAN operation, which can be used in vanilla FedAvg [2]. <br />![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/218841/1648211656332-ba763a7c-dec5-42b1-b235-22e00c1b17df.png#clientId=ufe8f6591-4d0f-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=328&id=uaca62bc5&margin=%5Bobject%20Object%5D&name=image.png&originHeight=656&originWidth=1940&originalType=binary&ratio=1&rotation=0&showTitle=false&size=398787&status=done&style=none&taskId=ufed2e182-5d8c-4b45-bd61-7bd8fdb1a85&title=&width=970)<br />To protect the privacy of the input$x_i$, the client splits it into several frames and makes sure that $x_{i,0} + x_{i,1}+ ... + x_{i,n}=x_i$($n-1$ frames can be randomly chosen from$[p]$where $p$is a large number). After that, the clients exchange the frames and sum the frames up, and send them to the server. In this way, what the server receives is a mixture of frames. However, when the server performs the mean operation on these received mixture frames, it happens to be the same result of performing the mean operation on the original inputs. 

In FederatedScope, we provide the fundamental implementation for such additive secret sharing.  Developers only need to use the high-level interface such as `secret_split` after specifying the number of the shared parties (i.e., the number of frames). <br />Note that the configuration `cfg.federate.use_ss` controls whether to apply additive secret sharing in the FL course. When it is set to `True`, the client will split each element in the message into several frames and sends one frame to every client, and at the same time receives frames from others. The results of applying additively secret sharing can be demonstrated as (we set a small$p=2^{15}+1=32769$for better understanding and preserve two decimals):
```
### Before secret_split:
tensor([[43.74, -6.00,  9.30, 55.81, 46.65]])

### After secret_split:
[[[21712. 31401. 11148.  8541. 24217.]]

 [[ 1075. 19873.  3733. 28288. 24990.]]

 [[14356. 13664. 18818.  1521. 20996.]]]
 
### Recover: (21712. + 1075. + 14356.) mod p / 1e2 = 43.74
```

Furthermore, more MPC algorithms can be implemented in FederatedScope to support various FL tasks. We aim to provide more support for MPC in the future.

---

- [1] Paillier P. "Public-key cryptosystems based on composite degree residuosity classes". International conference on the theory and applications of cryptographic techniques. 1999: 223-238.
- [2] McMahan B, Moore E, Ramage D, et al. "Communication-efficient learning of deep networks from decentralized data". Artificial intelligence and statistics. 2017: 1273-1282.
