---
title: "New Types of Messages and Handlers"
permalink: /docs/new-type/
excerpt: "About defining new stuffs."
last_modified_at: 2022-04-05T10:22:55-04:00
toc: true
layout: tuto
---

with the help of the [Event-driven Architecture]({{ "/docs/event-driven-architecture/" | relative_url }}), FederatedScope allows developers to customize FL applications via introducing new types of  messages and the corresponding handling functions. Here we provide the implementation details.

## Define new message type

Firstly developers should define a new type of message that is exchanged in the customized FL course. The new message should include sender, receiver, msg_type, and payload. 

For example, we define `Message(sender=server, receiver=client, msg_type='gradients', payload=MODEL_GRADIENTS)` to denote a new message containing gradients that is passed from the server to the client.

## Add handling function

After that, users should implement the handling function for the receiver (here is the client) to handle the newly defined message. The operations in the handling function might include parsing the payload, updating models, aggregating, triggering some events, returning feedback,  and so on. For example:

```python
class Client(object):

    ... ...
    
    # A handling function of client for 'gradients'
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
        else:
            updated_model = self.model
    
        # return the feedback via communivator
        self.comm_manager.send(
            Message(sender=self.ID, 
                    receiver=sender, 
                    msg_type='updated_model', 
                    content=updated_model))
```

Note that in some cases, the newly added handling function includes returning a message, such as _updated_model_ in the example. Users might need to define a new handling function for the returned message if it is also a new type, or accordingly modify the implemented handling functions if necessary.

## Register the handling function

FederatedScope allows users to add the new handling functions for servers or clients by registering:

```python
self.register_handlers(
    message_type='gradients', 
    callback_func=callback_for_messgae_gradients)
```

Thus, a new type of message can be exchanged and handled in a customized FL task. 
