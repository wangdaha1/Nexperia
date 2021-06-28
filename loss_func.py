import torch.nn as nn
import  torch
from torch.nn import functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    """
    这个函数的写法应该要和nn.CrossEntropy是一致的吧
       References URL: https://zhuanlan.zhihu.com/p/28527749
       This criterion is a implementation of Focal Loss, which is proposed in
       Focal Loss for Dense Object Detection.
           Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
       The losses are averaged across observations for each minibatch.
       Args:
           alpha(1D Tensor, Variable) : the scalar factor for this criterion
           gamma(float, double) : gamma > 0; reduces the relative loss for well-classified examples (p > .5),
                                  putting more focus on hard, misclassified examples
           size_average(bool): By default, the losses are averaged over observations for each minibatch.
                               However, if the field size_average is set to False, the losses are
                               instead summed for each minibatch.
   """

    def __init__(self, num_classes=8, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()  # 一般class有继承的时候都要加这句话啦
        if alpha is None:
            self.alpha = Variable(torch.ones(num_classes, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = num_classes
        self.size_average = size_average

    def forward(self, inputs, targets): # 这里还没看
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)
        # print("Loss probs minimus is: {}".format(torch.min(probs)))
        probs = probs.clamp(min=0.0001, max=1.0)
        # prob = prob.clamp(min=0.0001, max=1.0)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, scale=1.0, gamma=2, num_classes=8, alpha=None, size_average=True):
        super(NormalizedFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(num_classes, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale  # 这里的scale指的是后面在结合APL的时候用到的那个参数

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()  # 看下scale有没有用
        else:
            return loss.sum()

class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes=8, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_classes = num_classes
        self.scale = scale   # scale指的是APL的参数

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()  # 这里就算有warning也可以运行的欸 应该也是要加入一个size_average的参数吧

class AdjustCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        prob = input.softmax(dim=1)
        entropy = (- prob*prob.log()).sum(dim=1).detach()
        cross_entropy = -prob[torch.arange(prob.size(0)), target].log()
        return (cross_entropy*entropy).mean()  # 先用mean

class NFLandRCE(torch.nn.Module):
    def __init__(self, alpha=1, beta=1, num_classes=8, gamma=2):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)

class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return self.scale * nce.mean()
