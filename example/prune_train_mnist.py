from lib.LotteryTicket import LotteryTicket
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms, datasets
from archs.mnist import fc, LeNet
import os

lr = 1.2e-3
data_name = 'mnist'
model_name = 'fc'
# train
train_batch_size = 60
train_itrs = 8000
# eval
eval_batch_size = 200
eval_itrs = 200
# init recorder
init_itr = 1
prune_percent = 0.8
mode = 'layer'
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
# get init weights and masks
init_weights = torch.load(os.path.join(save_path, 'init_{}.pkl'.format(init_itr)))['net']
masks = torch.load(os.path.join(save_path, "masks_{}_{}_{}.pkl".format(prune_percent, mode, prune_itr)))['masks']

# set lottery ticket
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
LT = LotteryTicket(model, device, optimizer, criterion, init_weights, masks)

LT.train_init()

writer = SummaryWriter(logdir=os.path.join(save_path, "log_{}_{}_{}".format(prune_percent, mode, prune_itr)))

itr_index = 1
LT.train()

while itr_index <= train_itrs:
    # train loop
    for (inputs, targets) in train_loader:
        loss, acc, _ = LT.train_on_batch(inputs, targets)
        writer.add_scalar("train acc", acc, itr_index)
        writer.add_scalar("train loss", loss, itr_index)
        print("itr: {}/{}, loss={}, acc={}".format(itr_index, train_itrs, loss, acc))
        if itr_index % eval_itrs == 0:
            LT.eval()
            test_loss, test_acc = LT.eval_all(test_loader)
            print("testing...")
            print("itr: {}/{}, loss={}, acc={}".format(itr_index, train_itrs, test_loss, test_acc))
            writer.add_scalar("test acc", test_acc, itr_index)
            writer.add_scalar("test loss", test_loss, itr_index)
            print('Saving...')
            state = {
                'net': model.state_dict(),
                'acc': test_acc,
                'itr': itr_index,
            }
            torch.save(state, os.path.join(save_path, "ckpt_{}_{}_{}.pkl".format(prune_percent, mode, prune_itr)))
            writer.flush()

            # return to train state.
            LT.train()

        itr_index += 1
writer.close()
