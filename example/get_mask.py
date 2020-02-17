from lib.LotteryTicket import LotteryTicket
from archs.mnist import LeNet
import torch
from torchvision import transforms, datasets
import os

# train
lr = 1.2e-3
train_batch_size = 100
train_itrs = 5000
# prune
mode = 'layer'
fc_prune_percent = 0.80
conv_prune_percent = 0.90
prune_itrs = 3
# weights
trained_weights_path = './save/trained_weights.pkl'
init_weights_path = './save/init_weights.pkl'


dataset = datasets.MNIST
train_transform = transforms.Compose([transforms.ToTensor()])

# load data
train_data = dataset('./Dataset', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                           drop_last=False)

# build model
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = LeNet.LeNet().to(device)

# load the trained_weights
trained_weights = torch.load(trained_weights_path)['net']
model.load_state_dict(trained_weights)

# get init weights
init_weights = torch.load(os.path.join(init_weights_path))['net']

# set lottery ticket
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
LT = LotteryTicket(model, device, optimizer, criterion)

# ======================= get masks =========================
# get masks by single shot
masks = LT.get_masks_by_single_shot(conv_prune_percent, fc_prune_percent, mode)

# get masks by iteration
LT.set_init_weights(init_weights)
masks = LT.get_masks_by_iteration(prune_itrs, conv_prune_percent, fc_prune_percent, train_loader, train_itrs, mode)


# ================== if you want to train the model with masks ===================
# step 1: initial
# LT.set_init_weights(init_weights).set_masks(masks)
# LT.train_init()
# LT.train()    # change the mode of the model
# step 2: write your train_loop with function train_on_batch(inputs, targets) in lib/LotteryTicket.py


# ================== if you want to test the model with the masks ======================
# test on test_loader
# LT.eval_all(test_loader)
