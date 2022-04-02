---
title: "Privacy Protection for Message"
permalink: /docs/protected-msg/
excerpt: "About protecting messages."
last_modified_at: 2018-09-10T12:33:24-04:00
toc: true
layout: tuto
---

In order to satisfy different requirements of message privacy protection,  several technologies, including Differential Privacy, Encryption, Multi-Party Computation, etc., are applied in an FL course to enhance the strength of protection.

Here we give a brief introduction of how to use these technologies in FederatedScope.

## Differential Privacy

Differential privacy (DP) is a powerful theoretical criterion for privacy. A mechanism satisfying $(\epsilon-\delta)$ differential privacy promises the indistinguishability of similar information with the possibility $1-\delta$. 

As a message-oriented federated learning framework, FederatedScope usually exchanges messages at high frequency. Differential privacy provides a powerful theoretical metric to evaluate the balance between training performance and privacy protection. 

To protect the privacy contained in the message, FederatedScope provides flexible supports for differential privacy:

- **Abundant benchmark datasets and tasks** are preset in FederatedScope, e.g., CV, graph learning, NLP and recommendation tasks. It's quite convenient and simple to conduct DP evaluations under different settings;
- **Build-in DP algorithms** are user-friendly for the beginners; 
- **Flexible DP APIs** are preset for users to develop and implement their own DP algorithms; 
- **Rich attackers** are implemented in FederatedScope. It's convenient to test your DP algorithm.

## Encryption

One of the techniques to protect the privacy of messages is encryption. Before sending messages to others, participants can apply encryption algorithms to transfer the plaintext to ciphertext, which prevents privacy leakage during the communication. 

When receiving the ciphertext, the receiver usually needs to recover the plaintext via decryption. However, if homomorphic encryption algorithms, such as Paillier [1], are adopted, the receiver is allowed to perform (limited) operations on the ciphertext, which ensures the correctness of operations as well as prevents the receiver from being aware of the plaintext. 
Readers can refer to [Cross-silo FL]({{ "/docs/cross-silo/" | relative_url }}) for an example of using homomorphic encryption algorithms in an FL course. Note that users can extend more encryption algorithms in FederatedScope without caring about other modules since they have been decoupled with each other.

The biggest weakness of applying encryption in an FL course is bringing additional costs for both computation and communication. It might be intolerable in an FL course, especially for cross-device scenarios that involve IoT devices or smartphones that have limited computation resources and communication width.

## Secure Multi-Party Computation

Secure Multi-Party Computation (SMPC) aims to jointly compute a function by multiple participants while keeping the original inputs private. Formally, given$n$participants and each participant$i$owns the private data$x_i$. SMPC proposes to learn$y=f(x_1, x_2, ..., x_n)$without exposing the values of $x_1, x_2, ..., x_n$to other participants.
We show an example of additive secret sharing in the figure with instantiating$f$as MEAN operation, which can be used in vanilla FedAvg [2]. 
![](https://img.alicdn.com/imgextra/i4/O1CN01H7022d1UAVnc1wCt3_!!6000000002477-0-tps-2009-659.jpg)
To protect the privacy of input$x_i$, the client splits it into several frames and makes sure that $x_{i,1} + x_{i,2}+ ... + x_{i,n}=x_i$( where$n-1$frames are randomly chosen from$[p]$ and $p$is a large number). After that, the clients exchange the frames and sum the received frames up, and send the results to the server. In this way, what the server receives is a mixture of frames. Meanwhile, when the server performs the MEAN operation on these received mixture frames, it happens to be the same result as performing the MEAN operation on the original inputs. 

In FederatedScope, we provide the fundamental functionalities for the additive secret sharing.  Uers only need to utilize the high-level interface such as `secret_split` after specifying the number of the shared parties (i.e., the number of frames). 
Note that the configuration `cfg.federate.use_ss` controls whether to apply additive secret sharing in the FL course. When it is set to be `True`, the client splits each element of messages into several frames and sends one frame to every client. At the same time, the client receives$n-1$frames from others. The results of applying additively secret sharing can be demonstrated as (we use a small value of $p=2^{15}+1=32769$in this example for better understanding, and preserve two decimals):

```
### Before using secret_split:
tensor([43.74])

### After using secret_split:
[[21712.]

 [ 1075.]

 [14356.]]
 
### Recover: (21712. + 1075. + 14356.) mod p / 1e2 = 43.74
```

Furthermore, more SMPC algorithms can be implemented in FederatedScope to support various FL applications. We aim to provide more support for SMPC in the future.

## References

[1] Paillier P. "Public-key cryptosystems based on composite degree residuosity classes". International conference on the theory and applications of cryptographic techniques. 1999: 223-238.  
[2] McMahan B, Moore E, Ramage D, et al. "Communication-efficient learning of deep networks from decentralized data". Artificial intelligence and statistics. 2017: 1273-1282.