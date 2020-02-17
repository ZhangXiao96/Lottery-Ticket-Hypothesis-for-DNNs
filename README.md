# Lottery Ticket Hypothesis for DNNs

This repo provides an easy-to-use interface for searching the lottery ticket for pytorch models.

Thanks for the help of this repo: [https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch](https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch)

## Related Papers

@InProceedings{Frankle2018,

author    = {Frankle, Jonathan and Carbin, Michael},

title     = {The lottery ticket hypothesis: Finding sparse, trainable neural networks},

booktitle = {Proc. Int'l Conf. on Learning Representations},

year      = {2018},

address   = {Vancouver, Canada},

month     = may,

}


@Article{Frankle2019,

author        = {Frankle, Jonathan and Dziugaite, G Karolina and Roy, DM and Carbin, M},

title         = {Stabilizing the Lottery Ticket Hypothesis},

journal       = {CoRR},

year          = {2019},

volume        = {abs/1903.01611},

archiveprefix = {arXiv},

url           = {http://arxiv.org/abs/1903.01611},

}

## Requirements
We only tested on:
- python==3.7.4
- pytorch==1.4.0

## How to use it?

You can see [example/get_masks.py](https://github.com/ZhangXiao96/Lottery-Ticket-Hypothesis-in-DNN/blob/master/example/get_mask.py) for more information.

We also provide an [example](https://github.com/ZhangXiao96/Lottery-Ticket-Hypothesis-for-DNNs/tree/master/example) on MNIST. Please see these files:

1. train_mnist.py
2. get_mask_mnist_single.py or get_mask_mnist_itr.py
3. prune_train_mnist.py
4. prune_train_mnist_random.py

### 1. Prune by one shot
```
from lib.LotteryTicket import LotteryTicket

# model: "your trained pytorch model"
# device: "cuda" or "cpu"
LT = LotteryTicket(model, device)

# get masks by single shot
# mode: "global" or "layer"
masks = LT.get_masks_by_single_shot(conv_prune_percent, fc_prune_percent, mode)

```
### 2. Prune by iterations
```
from lib.LotteryTicket import LotteryTicket

# model: "your trained pytorch model"
# device: "cuda" or "cpu"
# optimizer: torch.nn.optim._
# criterion: torch.nn.**Loss()
LT = LotteryTicket(model, device, optimizer, criterion)

# get masks by iteration
# init_weights: the init weights of the model
LT.set_init_weights(init_weights)

# train_loader: Data_Loader for the train set.
# train_itrs: number of trained batches for each pruning iteration
# mode: "global" or "layer"
masks = LT.get_masks_by_iteration(prune_itr, conv_prune_percent, fc_prune_percent, train_loader, train_itrs, mode)
```
### 3. Train the model with masks:

#### step 1
Initial the training procedure.
```
LT.set_init_weights(init_weights).set_masks(masks)
LT.train_init()
LT.train()    # change the mode of the model
```

#### step 2
Write your own train loop with
```
LT.train_on_batch(inputs, targets)
```

#### others
If you want to test your model.
```
LT.eval_all(Data_Loader)
```

### 4. Some Results
Here we show the testing curves with a fully-connected DNN on MNIST. You can see my [example](https://github.com/ZhangXiao96/Lottery-Ticket-Hypothesis-for-DNNs/tree/master/example) to reproduce the results.

![Results](https://github.com/ZhangXiao96/Lottery-Ticket-Hypothesis-for-DNNs/blob/master/assets/cover.png)

### 5. Contact
If you have any questions, please email xiao_zhang@hust.edu.cn