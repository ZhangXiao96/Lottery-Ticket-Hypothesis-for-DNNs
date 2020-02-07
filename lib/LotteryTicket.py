import torch
import numpy as np


class LotteryTicket(object):
    # fixme: we shouldn't mask BN layer or others, hence I have to filter layers
    #  according to the shape of weights (which is not quite an ideal way). See _init_mask().

    def __init__(self, model, device):
        self.model = model
        self.init_weights = None
        self.optimizer = None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device
        self.masks = self._init_mask()

    def _init_mask(self):
        masks = {}
        for name, param in self.model.named_parameters():
            # we don't prune the bias.
            if 'weight' in name:
                if len(param.shape) >= 2:   # FC/Conv layer
                    tensor = param.data.cpu().numpy()
                    masks[name] = np.ones_like(tensor)
        return masks

    def get_mask_by_single_shot(self, prune_percent=0.8, mode="global"):
        masks = {}
        if mode == 'layer':     # layer-wised pruning
            for name, param in self.model.named_parameters():
                # we don't prune the bias.
                if 'weight' in name:
                    if len(param.shape) >= 2:   # FC/Conv layer
                        tensor = param.data.cpu().numpy()
                        flat_tensor = tensor.ravel()
                        threshold = np.percentile(np.abs(flat_tensor), 100 * prune_percent)
                        masks[name] = np.where(np.abs(tensor) <= threshold, 0., 1.)
        elif mode == 'global':  # global pruning
            # get the threshold
            all_weights = []
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    if len(param.shape) >= 2:   # FC/Conv layer
                        tensor = param.data.cpu().numpy()
                        all_weights.append(tensor.ravel())
            all_weights = np.concatenate(all_weights, axis=0)
            threshold = np.percentile(np.abs(all_weights), 100 * prune_percent)
            # get the mask
            for name, param in self.model.named_parameters():
                # we don't prune the bias.
                if 'weight' in name:
                    if len(param.shape) >= 2:   # FC/Conv layer
                        tensor = param.data.cpu().numpy()
                        masks[name] = np.where(np.abs(tensor) <= threshold, 0., 1.)
        else:
            raise Exception("The parameter mode must be \'layer\' or \'global\'!")
        return masks

    def train_on_batch(self, inputs, targets):
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()

        # prune the gradients
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                device = param.device
                grad_tensor = param.grad.data.cpu().numpy()
                grad_tensor = grad_tensor * self.masks[name]
                param.grad.data = torch.from_numpy(grad_tensor).to(device, dtype=torch.float32)
        self.optimizer.step()
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        return loss.item(), correct

    def eval_all(self, test_loader):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                loss, correct = self._eval_on_batch(inputs, targets)
                total += targets.size(0)
                test_loss += loss
                test_correct += correct
            test_loss /= (batch_idx + 1)
            test_acc = test_correct / total
        return test_loss, test_acc

    def _eval_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        return loss.item(), correct

    def train_init(self, masks, init_weights=None):
        self.set_mask(masks)
        self.set_init_weights(init_weights)
        if init_weights is not None:    # if None, use random initialization.
            self.model.load_state_dict(self.init_weights)
        # prune
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:  # FC/Conv layer
                    weights_deviece = param.device
                    tensor = param.data.cpu().numpy()
                    tensor = tensor * masks[name]
                    param.data = torch.from_numpy(tensor).to(weights_deviece, dtype=torch.float)
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def set_mask(self, masks):
        self.masks = masks
        return self

    def set_init_weights(self, init_weights):
        # the form of model.state_dict()
        self.init_weights = init_weights
        return self

    def set_criterion(self, criterion):
        self.criterion = criterion
        return self

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        return self

