---
title: "Message-oriented Framework"
permalink: /docs/msg-oriented-framework/
excerpt: "The message-oriented framework used in FederatedScope."
last_modified_at: 2021-05-11T10:40:42-04:00
toc: true
---

Before introducing more details about how to develop custom FL tasks for various real-world applications using provided functional components, it is necessary to present the message-oriented framework used in FederatedScope for you. On the one hand, our design paradigm can provide developers with another view for FL tasks, which is different from the widely-adopted sequential process and brings flexibility for customizing. On the other hand, understanding the message-oriented framework can help developers to know how to satisfy requirements using provided functional components or extend fancy features with FederatedScope conveniently and effortlessly.
<a name="gidYP"></a>
## Infrastructure
FederatedScope consists of three fundamental modules for implementing an FL procedure, i.e., Worker Module, Message Module, and Communicator Module.
<a name="xyc3E"></a>
### Worker Module
The worker module is used to describe the behaviors of servers and clients in an FL course, which is designed towards expressing rich behaviors conveniently.

We categorize the behaviors of servers and clients into federal behaviors and training behaviors. The federal behaviors include various message exchanges among participants, for example, a client sends a join-in application to the server, a server broadcasts the model parameter to clients for starting a federal training round. The training behaviors mainly denote the operations of updating models, including a client performing local training based on its data, a server aggregating the feedback (e.g., the updated models) from clients to generate the global model, and so on.

Considering both flexibility and convenience, we define the following fundamental member attributes of a worker (server/client) to decouple the federal behaviors and training behaviors:

- **ID**: The worker's ID is unique in an FL course for distinguishing. By default, we assign the server with ID:0 and number the clients according to the order they joined.
- **Data & Model: **Each worker holds its data and/or model locally. According to the settings of FL, the data are stored in the worker's private space and won't be directly shared directly because of privacy concerns. A client usually owns a training dataset and might have a local test dataset,  while a server might hold a dataset for global evaluation.

Besides, each worker maintains a model locally. In a standard FL procedure, all clients share the same model architecture, and the model parameters are synchronized through the server by aggregating. Since the local data are isolated and cannot be shared directly, the knowledge of these data is encoded into the models via local training and shared across participants in a privacy-preserving manner. Recent studies on personalized FL propose to control the extent of collaboratively learned during an FL course, which might cause the differences among the local models.

- **Communicator: **Each worker holds a communicator object for exchanging messages. The communicator will hide the low-level details of the communication backends and only exposes high-level interfaces for workers, such as _send_ or _receive_. 
- **Trainer/Aggregator: **A client/server holds a trainer/aggregator to encapsulate its training behaviors, which has the authority to access/update data&model and manages the training details, such as loss function, optimizer, aggregation algorithm. The client/server only needs to call the high-level interfaces (e.g., _train_, _eval_, _aggregate_) when necessary. In this way, we decouple the training behaviors with others for workers.

 <br />Based on the aforementioned attributes, developers can conveniently express the rich behaviors of servers and clients for various FL tasks. For example, developers can customize the aggregation algorithms in the Aggregator without caring about federal behaviors and model architectures (More practical examples and implementation can be found in other sections such as xxx).

<a name="tbPp3"></a>
### Message Module
An FL task might need to exchange various types of data among participants, such as the model parameters, gradients, and also join-in applications, signal of finish, etc. In FederatedScope, we abstract all the exchanged data among servers and clients as "Message". 

The message is one of the key objects to describe and drive an FL task in the proposed message-oriented framework. To this end, the message is designed to contain _type_, _sender_, _receiver_, and _payload_. Different FL tasks need different types of messages. For a standard FL course, server and client exchange model parameters during the training process, while for graph federated learning tasks, node embeddings, adjacency tensors might be need to share among participants.<br />Developers can define various types of messages according to the unique requirements of custom FL tasks, and, at the same time, describe the handled functions of server and client when receiving the new types of message. Thus, a custom FL task can be implemented via adding new types of exchanged messages and the corresponding handlers.

The detailed implementation of adding new types of exchanged messages and behaviors of servers and clients can be found in xxx.

<a name="zHcP6"></a>
### Communicator Module
The communicator is used for participants to exchange messages with each other, which can be regarded as a black box from server and client since only high-level interfaces (such as _send_ and _receive_) are exposed to them. The communicator hides the low-level implementation of backends so that it can provide a unified view for both standalone simulation and distributed deployment.

In FederatedScope, we implement a simulated communicator for standalone mode and gRPC communicator for distributed mode. Developers can implement more communicators based on various protocols according to the adopted environment.

<a name="UZIsQ"></a>
## Message Passing in FL
In FederatedScope, based on the message-oriented framework, an FL procedure is framed to multiple rounds of message passing and described in an event-driven way. In general, developers need to abstract the types of exchanged messages in the FL course, and transform the behaviors of servers and clients into handled functions as subroutines to handle different types of receiving messages.

Next we will introduce two examples to better demonstrate the characteristics of the proposed message-oriented framework, compared to the widely-adopted procedural programming paradigm.

<a name="hQIip"></a>
### The Standard FL Course
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/218841/1647843986366-780e0560-e9d0-4dce-991a-af6834b879e8.png#clientId=u75e704ca-ee03-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=532&id=ue900e2cb&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1064&originWidth=2340&originalType=binary&ratio=1&rotation=0&showTitle=false&size=922256&status=done&style=none&taskId=uaa3d3a7c-9057-47e4-bbe8-fe7db11cbc4&title=&width=1170)<br />A standard FL course viewed from message passing is shown in the figure. With procedural programming, developers need to carefully coordinate the participants. For example, firstly the clients send the join-in application to the server, and then the server receives these applications, and broadcasts the models to them, after that ... It can become too complicated for developers to implement a custom FL task.

On the other hand, when using message-oriented programming, to perform the vanilla FedAvg, developers first need to abstract the types of exchanged messages and the corresponding handlers: The server needs to handle two types of message, i.e., handle _join_ for admitting a new client to join in the FL course, and handle the updated _models_ to perform aggregation. As for the clients, they train the model on the local data and return the updated model when receiving _model_ from the server.  Finally, the FL course can be triggered by the instantiated clients sending _join_ to the server.

The message-oriented framework allows developers to focus on the behaviors as subroutines in an FL course and saves them from coordinating the participants, which also brings flexibility to support various FL tasks that require heterogeneous message exchanging and rich behaviors, as shown in the example below.

<a name="omjGL"></a>
### Heterogeneous Message Exchanging and Rich Behaviors
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2022/png/218841/1647844015096-48c6e6bb-46a6-4c4f-803b-2cab097c9e8b.png#clientId=u75e704ca-ee03-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=596&id=uc11e6b7a&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1192&originWidth=2344&originalType=binary&ratio=1&rotation=0&showTitle=false&size=1275875&status=done&style=none&taskId=u087b2344-745c-4711-b439-e1be87ab9bf&title=&width=1172)

We demonstrate another example in the figure to show how to describe a real-world FL application where heterogeneous messages are exchanged and handled. Compared to the standard FL course, here clients need to exchange intermediate results during each round of federal training. To achieve this, developers who use procedural frameworks are required to describe the complete FL course sequentially, add the new behaviors into the procedures after carefully positioning. 

For comparison, to express heterogeneous data changes and rich behaviors,  developers who use the message-oriented framework only need to define the new types of messages and the corresponding handled function of server and clients, eliminating the efforts for coordinating participants.<br />For example, developers only need to add a new type _intermediate results_ for clients and specify that client continues to local training when receiving the intermediate results, without bothering by when the new behavior (i.e., exchanging intermediate results) happens or how many times it happens in an FL course.

In a nutshell, the message-oriented framework views an FL course from message passing, and brings flexible support for various FL tasks.  Also, it provides a unified view for both standalone mode and distributed mode, which helps developers change from simulation to deployment effortlessly.
