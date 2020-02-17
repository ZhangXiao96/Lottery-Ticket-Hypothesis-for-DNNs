from lib.LotteryTicket import LotteryTicket
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms, datasets
import os
from archs.mnist import fc, LeNet

lr = 1.2e-3
data_name = 'mnist'
model_name = 'fc'
# train
train_batch_size = 60
train_itrs = 5000
# eval
eval_batch_size = 200
eval_itrs = 200
# init recorder
init_itr = 1
# prune
mode = 'layer'
prune_percent = 0.8
prune_itr = 4

dataset = datasets.MNIST
train_transform = transforms.Compose([transforms.ToTensor()])
eval_transform = transforms.Compose([transforms.ToTensor()])
if model_name == 'fc':
    model = fc.FC()
elif model_name == 'LeNet':
    model = LeNet.LeNet()
else:
    raise Exception("No such model!")

# load data
train_data = dataset('D:/Datasets', train=True, download=True, transform=train_transform)
test_data = dataset('D:/Datasets', train=False, transform=eval_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=0,
                                           drop_last=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=0,
                                          drop_last=False)

# build model
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# load the model
save_path = os.path.join('runs', data_name, model_name)
trained_weights = torch.load(os.path.join(save_path, 'ckpt.pkl'))['net']
model.load_state_dict(trained_weights)
# get init weights
init_weights = torch.load(os.path.join(save_path, 'init_{}.pkl'.format(init_itr)))['net']
# set lottery ticket
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
LT = LotteryTicket(model, device, optimizer, criterion, init_weights)


masks = LT.get_masks_by_iteration(prune_itr, 0, prune_percent/prune_itr, train_loader, train_itrs, mode)

torch.save({'masks': masks}, os.path.join(save_path, "masks_{}_{}_{}.pkl".format(prune_percent, mode, prune_itr)))
