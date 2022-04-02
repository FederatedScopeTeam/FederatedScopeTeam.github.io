---
title: "FederatedScope: A Comprehensive and Flexible Federated Learning Platform via Message Passing"
excerpt: "FederatedScope is a comprehensive and flexible federated learning platform proposed for providing fundamental support for various federated learning applications. Towards both convenient usage and flexible extension, FederatedScope exploits a message-oriented framework to describe an FL course, which frames the FL course into multiple rounds of message passing among participants."
excerptheader:
  image: "/assets/images/fed-core.jpg"
  caption: "The standard FL course viewed from the perspective of message passing"
categories:
  - "federated learning"
  - "system"
tags:
  - "federated learning"
  - "system"
---

Although remarkable progress has been made by the existing federated learning (FL) platforms to provide fundamental functionalities for application development, these FL platforms cannot well satisfy burgeoning demands from rapidly growing FL in academia and industry.  In this paper, we propose a novel and comprehensive federated learning platform based on a message-oriented framework, named FederatedScope. 
Towards more handy and flexible support for various FL tasks, FederatedScope frames an FL course into several rounds of message passing among participants, and allows developers to customize new types of exchanged messages and the corresponding handlers for various FL applications.
Compared to the procedural framework, the proposed message-oriented framework is more flexible to express heterogeneous message exchanging and rich behaviors of participants, and provides a unified view for both simulation and deployment.
Besides, we also include several functional components in FederatedScope, such as personalization, auto-tuning, and privacy protection, to satisfy the requirements of frontier studies in FL. 
We have public FederatedScope for developers to promote research and industrial deployment of federated learning in a variety of real-world applications.