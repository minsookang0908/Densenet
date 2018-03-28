import os
import time
import shutil
from tqdm import trange

import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable

from model import DenseNet
from tensorboard_logger import configure, log_value

class Trainer(object):
    """
    The Trainer class encapsulates all the logic necessary for 
    training the DenseNet model. It use SGD to update the weights 
    of the model given hyperparameters constraints provided by the 
    user in the config file.
    """
    def __init__(self, config, data_loader1, data_loader2):
        """
        Construct a new Trainer instance.

        Params
        ------
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        
        
        self.train_loader = data_loader1[0]
        self.valid_loader = data_loader1[1]
        
        self.test_loader = data_loader2

        # network params
        self.num_blocks = config.num_blocks
        self.num_layers_total = config.num_layers_total
        self.growth_rate = config.growth_rate
        self.bottleneck = config.bottleneck
        self.theta = config.compression

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.best_valid_acc = 0.
        self.init_lr = config.init_lr
        self.lr = self.init_lr
        print (self.lr)
        self.is_decay = True
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.dropout_rate = config.dropout_rate
        if config.lr_decay == '':
            self.is_decay = False
        else:
            self.lr_decay = [float(x) for x in config.lr_decay.split(',')]
        print (self.lr_decay)
        # other params
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.num_gpu = config.num_gpu
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.dataset = config.dataset
        if self.dataset == 'cifar10':
            self.num_classes = 10
        elif self.dataset == 'cifar100':
            self.num_classes = 100
        else:
            self.num_classes = 1000

        # build densenet model
        self.model = DenseNet(self.num_blocks, self.num_layers_total,
            self.growth_rate, self.num_classes, self.bottleneck, 
                self.dropout_rate, self.theta)
       
       
        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # define loss and optimizer
#self.init_lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.init_lr,momentum=0.9,nesterov = True, weight_decay=1e-4)

        if self.num_gpu > 0:
            self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids = range(self.num_gpu))
           # self.criterion.cuda()

        # finally configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.get_model_name()
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

    def train(self):
        """
        Train the model on the training set. 

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # switch to train mode for dropout
        self.model.train()

        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        for epoch in trange(self.start_epoch, self.epochs):
            self.model.train()
            # decay learning rate
            if self.is_decay:
                self.anneal_learning_rate(epoch)

            # train for 1 epoch
            self.train_one_epoch(epoch)

            # evaluate on validation set
            self.model.eval()
            valid_acc = self.validate(epoch)

            is_best = valid_acc > self.best_valid_acc
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_valid_acc': self.best_valid_acc}, is_best)
            self.test()
    def test(self):
        """
        Test the model on the held-out test data. 

        This function should only be called at the very
        end once the model has finished training.
        """
        # switch to test mode for dropout
        self.model.eval()

        accs = AverageMeter()
        batch_time = AverageMeter()

        # load the best checkpoint
        self.load_checkpoint(best=True)

        tic = time.time()
        for i, (image, target) in enumerate(self.test_loader):
            if self.num_gpu > 0:
                image = image.cuda()
                target = target.cuda(async=True)
            input_var = torch.autograd.Variable(image)
            target_var = torch.autograd.Variable(target)

            # forward pass
            output = self.model(input_var)

            # compute loss & accuracy 
            acc = self.accuracy(output.data, target)
            accs.update(acc, image.size()[0])

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)

            # print to screen
            if i % self.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Test Acc: {acc.val:.3f} ({acc.avg:.3f})'.format(
                        i, len(self.test_loader), batch_time=batch_time,
                        acc=accs))

        print('[*] Test Acc: {acc.avg:.3f}'.format(acc=accs))

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set. 

        An epoch corresponds to one full pass through the entire 
        training set in successive mini-batches. 

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        self.model.train()       
        tic = time.time()
        for i, (image, target) in enumerate(self.train_loader):
            if self.num_gpu > 0:
                image = image.cuda()
                target = target.cuda()
            input_var = torch.autograd.Variable(image)
            target_var = torch.autograd.Variable(target)
            # forward pass
            output = self.model(input_var)
            loss = 0   
            loss = self.criterion(output, target_var)
            acc = self.accuracy(output.data, target)
            losses.update(loss.data[0], image.size()[0])
            accs.update(acc, image.size()[0])

            # compute gradients and update SGD
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)

            # print to screen

            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Train Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Train Acc: {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, i, len(self.train_loader), batch_time=batch_time,
                        loss=losses, acc=accs))        # log to tensorboard
        if self.use_tensorboard:
            log_value('train_loss', losses.avg, epoch)
            log_value('train_acc', accses.avg, epoch)


    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()

        tic = time.time()
        for i, (image, target) in enumerate(self.valid_loader):
            if self.num_gpu > 0:
                image = image.cuda()
                target = target.cuda(async=True)
            input_var = torch.autograd.Variable(image)
            target_var = torch.autograd.Variable(target)

            # forward pass
            output = self.model(input_var)

            # compute loss & accuracy 
            loss = self.criterion(output, target_var)
            acc = self.accuracy(output.data, target)
            losses.update(loss.data[0], image.size()[0])
            accs.update(acc, image.size()[0])

            # measure elapsed time
            toc = time.time()
            batch_time.update(toc-tic)

            # print to screen
            if i % self.print_freq == 0:
                print('Valid: [{0}/{1}]\t'
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Valid Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Valid Acc: {acc.val:.3f} ({acc.avg:.3f})'.format(
                        i, len(self.valid_loader), batch_time=batch_time,
                        loss=losses, acc=accs))

        print('[*] Valid Acc: {acc.avg:.3f}'.format(acc=accs))

        # log to tensorboard
        if self.use_tensorboard:
            log_value('val_loss', losses.avg, epoch)
            log_value('val_acc', accs.avg, epoch)

        return accs.avg

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated 
        on the test data.

        Furthermore, the model with the highest accuracy is saved as
        with a special name.
        """
        print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.get_model_name() + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.get_model_name() + '_model_best.pth.tar'
            shutil.copyfile(ckpt_path, 
                os.path.join(self.ckpt_dir, filename))
            print("[*] ==== Best Valid Acc Achieved ====")

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in 
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.get_model_name() + '_ckpt.pth.tar'
        if best:
            filename = self.get_model_name() + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['state_dict'])
        
        print("[*] Loaded {} checkpoint @ epoch {} with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc']))

    def anneal_learning_rate(self, epoch):
        """
        This function decays the learning rate at 2 instances.

        - The initial learning rate is divided by 10 at
          t1*epochs.
        - It is further divided by 10 at t2*epochs. 

        t1 and t2 are floats specified by the user. The default
        values used by the authors of the paper are 0.5 and 0.75.
        """
        sched1 = int(self.lr_decay[0] * self.epochs)
        sched2 = int(self.lr_decay[1] * self.epochs)

        self.lr = self.init_lr * (0.1 ** (epoch // sched1)) \
                               * (0.1 ** (epoch // sched2))

        # log to tensorboard
        if self.use_tensorboard:
            log_value('learning_rate', self.lr, epoch)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def get_model_name(self):
        """
        Returns the name of the model based on the configuration
        parameters.

        The name will take the form DenseNet-X-Y-Z where:

        - X: total number of layers specified by `config.total_num_layers`.
        - Y: can be BC or an empty string specified by `config.bottleneck`.
        - Z: name of the dataset specified by `config.dataset`.

        For example, given 169 layers with bottleneck on CIFAR-10, this 
        function will output `DenseNet-BC-169-cifar10`.
        """
        if self.bottleneck:
            return 'DenseNet-BC-{}-{}'.format(self.num_layers_total,
                self.dataset)
        return 'DenseNet-{}-{}'.format(self.num_layers_total,
            self.dataset)

    def accuracy(self, predicted, ground_truth):
        """
        Utility function for calculating the accuracy of the model.

        Params
        ------
        - predicted: (torch.FloatTensor)
        - ground_truth: (torch.LongTensor)

        Returns
        -------
        - acc: (float) % accuracy.
        """
        predicted = torch.max(predicted, 1)[1]
        total = len(ground_truth)
        correct = (predicted == ground_truth).sum()
        acc = 100 * (correct / total)
        return acc

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
