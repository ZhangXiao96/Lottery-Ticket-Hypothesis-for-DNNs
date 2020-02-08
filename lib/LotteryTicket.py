import torch
import numpy as np
import copy


class LotteryTicket(object):
    # fixme: we shouldn't mask BN layer or others, hence I have to filter layers
    #  according to the shape of weights (which is not an ideal way). See _init_mask().

    def __init__(self, model, device, optimizer=None, criterion=None, init_weights=None):
        """
        NOTE: The model should be trained.
        :param model:  pytorch model
        :param device: device of inputs
        :param optimizer: Not needed if you only use get_masks_by_single_shot())
        :param criterion: Not needed if you only use get_masks_by_single_shot())
        :param init_weights: the form model.state_dict(). Not needed if you only use get_masks_by_single_shot())
        """
        self.model = model
        self.init_weights = copy.deepcopy(init_weights)
        self.optimizer = optimizer
        self._optimizer_init = copy.deepcopy(self.optimizer.state_dict())
        self.criterion = criterion
        self.device = device
        self.masks = self.init_masks()

    def init_masks(self):
        masks = {}
        for name, param in self.model.named_parameters():
            # we don't prune the bias.
            if 'weight' in name:
                if len(param.shape) >= 2:   # FC/Conv layer
                    tensor = param.data.cpu().numpy()
                    masks[name] = np.ones_like(tensor)
        return masks

    def get_masks_by_single_shot(self, prune_percent, mode="global"):
        """
        Get masks by single shot pruning.
        :param prune_percent: How many weights to prune. [0, 1]
        :param mode: "layer"/"global" for layer-wised/global pruning
        :return: the masks. Dict['param_name'] = 0/1 tensor
        """
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

    def get_masks_by_iteration(self, prune_percents, train_loader, train_iter, mode="global"):
        """
        Get masks by iterative pruning.
        :param prune_percents: lists. How many weights to prune for each iteration.
        :param train_loader: Data_Loader in pytorch.
        :param train_iter: Number of batches to train the model.
        :param mode: "layer"/"global" for layer-wised/global pruning
        :return: the masks. Dict['param_name'] = 0/1 tensor
        """
        re_mask = copy.deepcopy(self.masks)     # record init masks for reloading
        accumulate_prune_percent = prune_percents[0]
        masks = self.get_masks_by_single_shot(accumulate_prune_percent, mode=mode)
        for prune_percent in prune_percents[1:]:
            accumulate_prune_percent += prune_percent
            self.train_init()
            self.train_all(train_loader, train_iter)
            masks = self.get_masks_by_single_shot(accumulate_prune_percent, mode=mode)
        # reload self.masks so that this method won't influence self.masks.
        self.masks = re_mask
        return masks

    def train_on_batch(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
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
        acc = correct / targets.size(0)
        return loss.item(), acc, correct

    def train_all(self, train_loader, train_itrs=5000):
        self.model.train()
        itr_index = 1
        while itr_index <= train_itrs:
            for (inputs, targets) in train_loader:
                loss, acc, _ = self.train_on_batch(inputs, targets)
                print("itr: {}/{}, loss={}, acc={}".format(itr_index, train_itrs, loss, acc))
                itr_index += 1

    def eval_all(self, test_loader):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
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

    def train_init(self):
        self.model.load_state_dict(self.init_weights)
        self.optimizer.load_state_dict(self._optimizer_init)
        self.prune_weights()
        return self

    def prune_weights(self):
        # prune
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:  # FC/Conv layer
                    weights_device = param.device
                    tensor = param.data.cpu().numpy()
                    tensor = tensor * self.masks[name]
                    param.data = torch.from_numpy(tensor).to(weights_device, dtype=torch.float)
        return self

    def train(self):
        return self.model.train()

    def eval(self):
        return self.model.eval()

    def set_masks(self, masks):
        self.masks = copy.deepcopy(masks)
        return self

    def set_init_weights(self, init_weights):
        # the form of model.state_dict()
        self.init_weights = copy.deepcopy(init_weights)
        return self

    def set_criterion(self, criterion):
        self.criterion = criterion
        return self

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self._optimizer_init = copy.deepcopy(self.optimizer.state_dict())
        return self

