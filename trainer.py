
from bisect import bisect_right
import numpy as np
import torch
import torch.nn.functional as F




# 可变学习率
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1 / 3,
            warmup_iters=100,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]



# 训练模型的函数
class ModelTrainer(object):

    @staticmethod
    # static method就是在用的时候不需要将class实例化也ok 其他好像也没什么特殊的额
    def train(data_loader, model, loss_f, optimizer, epoch_id, gpu, max_epoch):
        '''
        :param data_loader: 就是用DataLoader函数导出来的batches
        :param model: 模型，这里的模型应该是要满足一些条件的吧  反正main函数里面用models.resnet34改造后得到的模型可以用
        :param loss_f: nn.CrossEntropyLoss 这种就可以 FocalLoss的写法应该和ce的是一致的咯
        :param optimizer: optim.SGD
        :param epoch_id: 第几个epoch
        :param gpu: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        :param max_epoch: total epoches
        :return: loss_train, acc_train, mat_train, y_true_train, y_outputs_train
                 训练损失     训练准确率   混淆矩阵   真实label     预测的概率（对于每个class都有，二维list）
        '''
        model.train()
        # Some models use modules which have different training and evaluation behavior, such as batch normalization.
        # To switch between these modes, use model.train() or model.eval() as appropriate.

        conf_mat = np.zeros((8, 8))  # confusion matrix 每次epoch都清空啦
        loss_sigma = []  # 损失函数  记录每一次epoch每一个batch的training loss
        label_append = []
        outputs_append = []

        for i, data in enumerate(data_loader):
            # data = batch_x, batch_y

            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

            outputs = model(inputs) # forward 了

            optimizer.zero_grad()
            # optimizer.module.zero_grad()
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)

            # 统计混淆矩阵记录
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss
            loss_sigma.append(loss.item())

            # save labels and outputs to calculate ROC_auc and PR_auc
            probs = F.softmax(outputs, dim=1)
            label_append.append(labels.detach())
            outputs_append.append(probs.detach())  # 这里放在outputs里面的是对应的概率哦

            # average accuracy of each batch
            acc_avg = conf_mat.trace() / conf_mat.sum()  # accuracy_average of each batch

            # 每300个batch 打印一次训练信息，loss为300个batch的平均  可以每300次batch iteration打印一次也可以每个epoch才打印一次啦
            if i % 300 == 300 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch_id + 1, max_epoch, i + 1, len(data_loader), np.mean(loss_sigma), acc_avg))

        return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append

    @staticmethod
    def valid(data_loader, model, loss_f, gpu):
        model.eval()

        conf_mat = np.zeros((8, 8))
        loss_sigma = []
        label_append = []
        outputs_append = []

        for i, data in enumerate(data_loader):

            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

            outputs = model(inputs)
            loss = loss_f(outputs, labels)

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 统计loss
            loss_sigma.append(loss.item())

            # save labels and outputs to calculate ROC_auc and PR_auc
            probs = F.softmax(outputs, dim=1)
            label_append.append(labels.detach())
            outputs_append.append(probs.detach())

        acc_avg = conf_mat.trace() / conf_mat.sum()
        # do not need to print anything in the validation process of each epoch

        return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append
