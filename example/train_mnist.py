from lib.ModelWrapper import ModelWrapper
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms, datasets
from archs.mnist import fc, LeNet
import os
from lib import utils

lr = 1.2e-3
data_name = 'mnist'
model_name = 'fc'
# train
train_batch_size = 60
train_itrs = 8000
# eval
eval_batch_size = 500
eval_itrs = 200
# init recorder
init_record_itr = 1


dataset = datasets.MNIST
train_transform = transforms.Compose([transforms.ToTensor()])
eval_transform = transforms.Compose([transforms.ToTensor()])
if model_name == 'fc':
    model = fc.FC()
elif model_name == 'lenet':
    model = LeNet.LeNet()
else:
    raise Exception("No such model!")

model.apply(utils.weight_init_method)

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

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
wrapper = ModelWrapper(model, optimizer, criterion, device)

# train the model
save_path = os.path.join('runs', data_name, model_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)
writer = SummaryWriter(logdir=os.path.join(save_path, "log"))

itr_index = 1
wrapper.train()

while itr_index <= train_itrs:

    if itr_index == init_record_itr:    # record the init weights.
        state = {
            'net': model.state_dict(),
            'itr': itr_index,
        }
        torch.save(state, os.path.join(save_path, "init_{}.pkl".format(itr_index)))

    # train loop
    for (inputs, targets) in train_loader:
        loss, acc, _ = wrapper.train_on_batch(inputs, targets)
        writer.add_scalar("train acc", acc, itr_index)
        writer.add_scalar("train loss", loss, itr_index)
        print("itr: {}/{}, loss={}, acc={}".format(itr_index, train_itrs, loss, acc))
        if itr_index % eval_itrs == 0:
            wrapper.eval()
            test_loss, test_acc = wrapper.eval_all(test_loader)
            print("testing...")
            print("itr: {}/{}, loss={}, acc={}".format(itr_index, train_itrs, test_loss, test_acc))
            writer.add_scalar("test acc", test_acc, itr_index)
            writer.add_scalar("test loss", test_loss, itr_index)
            state = {
                'net': model.state_dict(),
                'acc': test_acc,
                'itr': itr_index,
            }
            torch.save(state, os.path.join(save_path, "ckpt.pkl"))
            writer.flush()
            # return to train state.
            wrapper.train()
        itr_index += 1
        if itr_index > train_itrs:
            break

writer.close()
