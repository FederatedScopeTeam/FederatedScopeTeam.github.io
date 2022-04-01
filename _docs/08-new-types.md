---
title: "New Types of Messages and Handlers"
permalink: /docs/new-type/
excerpt: "About defining new stuffs."
last_modified_at: 2021-05-11T10:22:55-04:00
toc: true
---

Based on the [message-oriented framework](https://yuque.antfin-inc.com/gy2g1n/dcpcvz/ycrgg9), FederatedScope allows developers to customize FL tasks by introducing new types of exchanged messages and the corresponding handled functions. Here we provide the details of implementation.

<a name="lxRt0"></a>
## Define new message type
Firstly developers should define a new type of message that is exchanged in the custom FL course. The new message should include sender, receiver, msg_type, and payload. <br />For example, we define `Message(sender=server, receiver=client, msg_type='gradients', payload=MODEL_GRADIENTS)` to denote a new message containing gradients that is passed from server to the client.

<a name="tJPvs"></a>
## Add handled function
After that, developers should implement the handled function for the receiver (here is the client) to handle the newly defined message. The operations in the handled function can include parsing the payload, updating models, aggregating, triggering some events, returning feedback,  and so on. For example:
```python
class Client(object):

    ... ...
    
    # A handled function of client for 'gradients'
    def callback_for_messgae_gradients(self, message):
        # parse the payload
        sender, model_gradients = message.sender, message.content
        assert sender == self.server_ID
    
        # update model via trainer
        self.trainer.update_by_gradients(model_gradients)
    
        # trigger some events
        if	self.trainer.get_delta_of_model() > self.threshold:
            # local training
            updated_model = self.trainer.local_train()
    
        # return the feedback via communivator
        self.comm_manager.send(
            Message(sender=self.ID, 
                    receiver=sender, 
                    msg_type='updated_model', 
                    content=updated_model))
```
Note that in some cases, the newly added handled function includes returning a message, such as 'updated_model' in the example. Developers might need to define a new handled function for the returned message if it is also a new type, or accordingly modify the implemented handled functions if necessary.

<a name="Q9HSE"></a>
## Register the handled function
FederatedScope allows developers to add the new handled functions for server or client by registering:
```python
self.register_handlers(
    message_type='gradients', 
    callback_func=callback_for_messgae_gradients)
```
Thus, a new type of message can be exchanged and handled in a custom FL task. 
