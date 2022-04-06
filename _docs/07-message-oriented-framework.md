---
title: "Message-oriented Framework"
permalink: /docs/msg-oriented-framework/
excerpt: "The message-oriented framework used in FederatedScope."
last_modified_at: 2021-05-11T10:40:42-04:00
toc: true
layout: tuto
---

Before introducing more details about how to develop customized FL procedures for various real-world applications with FederatedScope, it is necessary to present the message-oriented framework for you. On the one hand, our design philosophy can provide users with another perspective for federated learning, which is different from the widely-adopted sequential process and might bring benefits for customizing. On the other hand, the introduction of the message-oriented framework can help users to know how to implement using provided rich functional components or extending fancy features with FederatedScope conveniently and effortlessly.

## Infrastructure

FederatedScope consists of two basic modules for implementing an FL procedure, i.e., Worker Module and Communication Module.

### Worker Module

The worker module is used to describe the behaviors of servers and clients in an FL course, which is designed towards expressing rich behaviors conveniently.

We categorize the behaviors of servers and clients into federated behaviors and training behaviors. The federated behaviors include various message exchanges among participants, for example, a client sends a join-in application to the server, and a server broadcasts the model parameter to clients for starting a new training round. The training behaviors mainly denote the operations of updating models, including a client performing local training based on its data, a server aggregating the feedbacks (e.g., the updated models) from clients to generate the global model, and so on.

Considering both flexibility and convenience, we define the following basic member attributes of a worker (a server or a client) to decouple the federated behaviors and training behaviors:

- **ID**: The worker's ID is unique in an FL course for distinguishing. By default, we assign the server with "0" and number the clients according to the order they joined the FL course.
- **Data & Model**: Each worker holds its data and/or model locally. According to the settings of FL, the data are stored in the worker's private space and won't be directly shared directly because of privacy concerns. A client usually owns a training dataset and might have a local test dataset,  while a server might hold a dataset for global evaluation.

Besides, each worker maintains a model locally. In a standard FL course, all clients share the same model architecture, and the model parameters are synchronized through the server by aggregating. Since the local data are isolated and cannot be shared directly, the knowledge of these data is encoded into the models via local training and shared across participants in a privacy-preserving manner. Recent studies on personalized FL [1, 2] propose to control the extent of learning from the global models during an FL course, which might cause differences among the local models.

- **Communicator**: Each worker holds a communicator object for message exchange. The communicator hides the low-level details of the communication backends and only exposes high-level interfaces for workers, such as *send* and *receive*. 
- **Trainer/Aggregator**: A client/server holds a trainer/aggregator to encapsulate its training behaviors, which has the authority to access/update data and models. The trainer/aggregator manages the training details, such as loss function, optimizer, aggregation algorithm. The client/server only needs to call the high-level interfaces (e.g., *train*, *eval*, *aggregate*) without caring about the training details. In this way, we decouple the training behaviors and the federated behaviors of workers.


Based on the aforementioned attributes, users can conveniently express the rich behaviors of servers and clients for various FL applications. For example, users can customize the aggregation algorithms in the Aggregator without caring about federated behaviors and model architectures. 

### Communication Module

**Message**  
An FL course might need to exchange various types of data among participants, such as the model parameters, gradients, and also join-in applications, signal of finish, etc. In FederatedScope, we abstract all the exchanged data among servers and clients as "Message". 

The message is one of the key objects to describe and drive an FL course in the proposed message-oriented framework. To this end, the message is designed to contain *type*, *sender*, *receiver*, and *payload*. Different FL applications need different types of messages. For a standard FL course, servers and clients exchange model parameters during the training process, while for Graph Federated Learning tasks [3,4], node embeddings, adjacency tensors might be needed to share among participants.  

Users can define various types of messages according to the unique requirements of customized FL courses, and, at the same time, describe the handling functions of servers and clients to handle these received messages. Thus, a customized FL course can be implemented by adding new types of messages and the corresponding handling functions.

The detailed implementation of adding new types of exchanged messages and behaviors of servers and clients can be found in [New Types of Messages and Handlers]({{ "/docs/new-type/" | relative_url }}).

**Communicator**  
The communicator is used for participants to exchange messages with each other, which can be regarded as a black box viewed by servers and clients, since only high-level interfaces (such as *send* and *receive*) are exposed to them. The communicator hides the low-level implementation of backends so that it can provide a unified view for both standalone simulation and distributed deployment.

In FederatedScope, we implement a simulated communicator for standalone mode, and a gRPC [5] communicator for distributed mode. Users can implement more communicators based on various protocols according to the adopted environments.

## Message Passing

In FederatedScope, based on the message-oriented framework, an FL course is framed to multiple rounds of message passing and described in an event-driven manner. In general, users need to abstract the types of exchanged messages in the FL course, and then transform the behaviors of servers and clients into handling functions as subroutines to handle different types of received messages.

Next we will introduce two examples to better demonstrate the characteristics of the proposed message-oriented framework, compared to the widely-adopted procedural programming paradigm.

### The Standard FL Course

![](https://img.alicdn.com/imgextra/i4/O1CN01lzXxmU1C4PRPL1YQ4_!!6000000000027-0-tps-1969-879.jpg)
A standard FL course viewed from the perspective of message passing is shown in the figure. Using the procedural programming paradigm, developers need to carefully coordinate the participants. For example, firstly the clients send the join-in application to the server, and then the server receives these applications, and broadcasts the models to them, after that ... It can become too complicated for developers to implement a customized FL task.

On the other hand, when using the message-oriented framework, to perform the vanilla FedAvg, developers first need to abstract the types of exchanged messages and the corresponding handlers: The server needs to handle two types of message, i.e., handle *join* for admitting a new client to join in the FL course, and handle the updated *models* to perform aggregation. As for the clients, they train the model on the local data and return the updated model when receiving *model* from the server.  Finally, the FL course can be triggered by the instantiated clients sending *join* to the server.

The message-oriented framework allows users to focus on the behaviors as subroutines in an FL course and saves users' effort in coordinating the participants. And it also brings flexibility to support various FL applications that require heterogeneous message exchanging and rich behaviors, as shown in another example below.

### Heterogeneous Message Exchange and Rich Behaviors

![](https://img.alicdn.com/imgextra/i4/O1CN01k5veEB21uf205H7jr_!!6000000007045-0-tps-1984-1072.jpg)
We demonstrate another example in the figure to show how to describe a real-world FL application where heterogeneous messages are exchanged and handled. Compared to the standard FL course, here clients need to exchange intermediate results during each training round. To achieve this, developers who use the procedural programming paradigm are required to describe the complete FL course sequentially, add the new behaviors into the procedures after carefully positioning. 

For comparison, to express heterogeneous data change and rich behaviors,  developers who use the message-oriented framework only need to define the new types of messages and the corresponding handling function of servers and clients, eliminating the efforts for coordinating participants.

For example, developers only need to add a new type *intermediate results* for clients, and specify that clients continue training locally when receiving the intermediate results, without bothering by when the new behavior (i.e., exchanging intermediate results) happens or how many times it happens in an FL course.

In a nutshell, the message-oriented framework views an FL course from the perspective of message passing, and brings flexible support for various FL applications.  Also, it provides a unified view for both standalone mode and distributed mode, which helps users change from simulation to deployment effortlessly.

## References

[1] Tan A Z, Yu H, Cui L, et al. "Towards personalized federated learning". arXiv preprint arXiv:2103.00710, 2021.  
[2] Fallah A, Mokhtari A, Ozdaglar A. "Personalized federated learning: A meta-learning approach". arXiv preprint arXiv:2002.07948, 2020.  
[3] Wu C, Wu F, Cao Y, et al. "Fedgnn: Federated graph neural network for privacy-preserving recommendation". arXiv preprint arXiv:2102.04925, 2021.  
[4] Zhang K, Yang C, Li X, et al. "Subgraph federated learning with missing neighbor generation". Advances in Neural Information Processing Systems, 2021, 34.  
[5] https://grpc.io/  