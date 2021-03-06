
from bisect import bisect_right
import numpy as np
import torch
import torch.nn.functional as F
from misc import mixup_data, mixup_criterion, cutmix_data, cutout
from torch.autograd import Variable
from apex import amp


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

    @staticmethod # static method就是在用的时候不需要将class实例化也ok 其他好像也没什么特殊的额
    # vanilla train
    def train(data_loader, model, loss_f, optimizer, epoch_id, gpu, max_epoch, num_classes):
        ''' 这个函数是训练单个epoch
        :param data_loader: 就是用DataLoader函数导出来的batches
        :param model:
        :param loss_f: nn.CrossEntropyLoss 这种就可以 FocalLoss的写法应该和ce的是一致的咯
        :param optimizer: optim.SGD
        :param epoch_id: 第几个epoch
        :param gpu: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        :param max_epoch: total epoches
        :param num_classes: class number
        :return: loss_train, acc_train, mat_train, y_true_train, y_outputs_train
                 训练损失     训练准确率   混淆矩阵   真实label     预测的概率（对于每个class都有，二维list）
        '''
        model.train()
        # Some models use modules which have different training and evaluation behavior, such as batch normalization.
        # To switch between these modes, use model.train() or model.eval() as appropriate.

        conf_mat = np.zeros((num_classes, num_classes))  # confusion matrix 每次epoch都清空
        loss_sigma = []  # 损失函数  记录每一次epoch每一个batch的training loss
        label_append = []
        outputs_append = []

        for i, data in enumerate(data_loader):
            inputs, labels = data
            # print(labels)
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

            outputs = model(inputs)
            optimizer.zero_grad()
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
            probs = F.softmax(outputs, dim=1)  # 模型本身是不带softmax的 如果是用nn.CrossEntropy会自行softmax处理之后再计算
            label_append.append(labels.detach())
            outputs_append.append(probs.detach())  # 这里放在outputs里面的是对应的概率哦
            # average accuracy of each batch
            acc_avg = conf_mat.trace() / conf_mat.sum()  # accuracy_average of each batch

        return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append

    @ staticmethod
    def train_apex(data_loader, model, loss_f, optimizer, epoch_id, gpu, max_epoch, num_classes):
        ''' 这个函数是训练单个epoch
        :param data_loader: 就是用DataLoader函数导出来的batches
        :param model:
        :param loss_f: nn.CrossEntropyLoss 这种就可以 FocalLoss的写法应该和ce的是一致的咯
        :param optimizer: optim.SGD
        :param epoch_id: 第几个epoch
        :param gpu: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        :param max_epoch: total epoches
        :param num_classes: class number
        :return: loss_train, acc_train, mat_train, y_true_train, y_outputs_train
                 训练损失     训练准确率   混淆矩阵   真实label     预测的概率（对于每个class都有，二维list）
        '''
        model.train()
        # Some models use modules which have different training and evaluation behavior, such as batch normalization.
        # To switch between these modes, use model.train() or model.eval() as appropriate.

        conf_mat = np.zeros((num_classes, num_classes))  # confusion matrix 每次epoch都清空
        loss_sigma = []  # 损失函数  记录每一次epoch每一个batch的training loss
        label_append = []
        outputs_append = []

        for i, data in enumerate(data_loader):
            inputs, labels = data
            # print(labels)
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

            outputs = model(inputs)
            optimizer.zero_grad()
            loss = loss_f(outputs, labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
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
            probs = F.softmax(outputs, dim=1)  # 模型本身是不带softmax的 如果是用nn.CrossEntropy会自行softmax处理之后再计算
            label_append.append(labels.detach())
            outputs_append.append(probs.detach())  # 这里放在outputs里面的是对应的概率哦
            # average accuracy of each batch
            acc_avg = conf_mat.trace() / conf_mat.sum()  # accuracy_average of each batch

        return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append

    @staticmethod
    # mixup, add mix_prob
    def train_mixup(data_loader, model, loss_f, optimizer, epoch_id, gpu, max_epoch, num_classes, mixup_alpha, mix_prob):
        model.train()

        conf_mat = np.zeros((num_classes, num_classes))  # confusion matrix 每次epoch都清空
        loss_sigma = []  # 损失函数  记录每一次epoch每一个batch的training loss
        outputs_append = []
        label_append_a = []
        label_append_b = []
        lam_append = []

        for i, data in enumerate(data_loader):

            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

            r = np.random.rand(1)
            if r < mix_prob: # 再次加入随机性
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, gpu)
                # 所以这里其实是对一个batch里面的数据进行mixup操作
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                outputs = model(inputs)
                optimizer.zero_grad()
                loss = mixup_criterion(loss_f, outputs, targets_a, targets_b, lam)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                # 这里计算confusion matrix就是为了后面计算其他的东西的吧  感觉没有太多的意义的
                for j in range(len(targets_a)):
                    cate_i_a = targets_a[j].cpu().numpy()  # the first label
                    cate_i_b = targets_b[j].cpu().numpy()  # the second label
                    pre_i = predicted[j].cpu().numpy()
                    conf_mat[cate_i_a, pre_i] += lam
                    conf_mat[cate_i_b, pre_i] += 1 - lam
            else: # 不然就是一般的训练过程
                outputs = model(inputs)
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
            # 感觉这里计算的ROC和PR的auc没太多的意义吧 而且计算的方法也得改变下额
            probs = F.softmax(outputs, dim=1)  # 模型本身是不带softmax的 如果是用nn.CrossEntropy会自行softmax处理之后再计算
            # label_append_a.append(targets_a.detach())
            # label_append_b.append(targets_b.detach())
            # lam_append.append += np.repeat(lam, len(predicted)).tolist()
            outputs_append.append(probs.detach())  # 这里放在outputs里面的是对应的概率哦
            # average accuracy of each batch
            acc_avg = conf_mat.trace() / conf_mat.sum()  # accuracy_average of each batch

        # 这里返回来了一堆东西 但是有很多其实是没什么用的 还没有去改动呢
        return np.mean(loss_sigma), acc_avg, conf_mat, label_append_a, label_append_b, outputs_append, lam_append

    @ staticmethod
    def train_mixup_apex(data_loader, model, loss_f, optimizer, epoch_id, gpu, max_epoch, num_classes, mixup_alpha, mix_prob):
        model.train()

        conf_mat = np.zeros((num_classes, num_classes))  # confusion matrix 每次epoch都清空
        loss_sigma = []  # 损失函数  记录每一次epoch每一个batch的training loss
        outputs_append = []
        label_append_a = []
        label_append_b = []
        lam_append = []

        for i, data in enumerate(data_loader):

            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

            r = np.random.rand(1)
            if r < mix_prob:  # 再次加入随机性
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, gpu)
                # 所以这里其实是对一个batch里面的数据进行mixup操作
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                outputs = model(inputs)
                optimizer.zero_grad()
                loss = mixup_criterion(loss_f, outputs, targets_a, targets_b, lam)
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                # 这里计算confusion matrix就是为了后面计算其他的东西的吧  感觉没有太多的意义的
                for j in range(len(targets_a)):
                    cate_i_a = targets_a[j].cpu().numpy()  # the first label
                    cate_i_b = targets_b[j].cpu().numpy()  # the second label
                    pre_i = predicted[j].cpu().numpy()
                    conf_mat[cate_i_a, pre_i] += lam
                    conf_mat[cate_i_b, pre_i] += 1 - lam
            else:  # 不然就是一般的训练过程
                outputs = model(inputs)
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
            # 感觉这里计算的ROC和PR的auc没太多的意义吧 而且计算的方法也得改变下额
            probs = F.softmax(outputs, dim=1)  # 模型本身是不带softmax的 如果是用nn.CrossEntropy会自行softmax处理之后再计算
            # label_append_a.append(targets_a.detach())
            # label_append_b.append(targets_b.detach())
            # lam_append.append += np.repeat(lam, len(predicted)).tolist()
            outputs_append.append(probs.detach())  # 这里放在outputs里面的是对应的概率哦
            # average accuracy of each batch
            acc_avg = conf_mat.trace() / conf_mat.sum()  # accuracy_average of each batch

        # 这里返回来了一堆东西 但是有很多其实是没什么用的 还没有去改动呢
        return np.mean(loss_sigma), acc_avg, conf_mat, label_append_a, label_append_b, outputs_append, lam_append

    @ staticmethod
    # cutmix
    def train_cutmix(data_loader, model, loss_f, optimizer, epoch_id, gpu, max_epoch, num_classes, mixup_alpha, mix_prob):
        model.train()

        conf_mat = np.zeros((num_classes, num_classes))  # confusion matrix 每次epoch都清空
        loss_sigma = []  # 损失函数  记录每一次epoch每一个batch的training loss
        outputs_append = []
        label_append_a = []
        label_append_b = []
        lam_append = []

        for i, data in enumerate(data_loader):

            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)

            r = np.random.rand(1)
            if r < mix_prob: # 再次加入随机性

                inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, mixup_alpha, gpu)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                outputs = model(inputs)
                optimizer.zero_grad()
                # 这里loss和mixup是可以通用的
                loss = mixup_criterion(loss_f, outputs, targets_a, targets_b, lam)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                # 这里计算confusion matrix就是为了后面计算其他的东西的吧  感觉没有太多的意义的
                for j in range(len(targets_a)):
                    cate_i_a = targets_a[j].cpu().numpy()  # the first label
                    cate_i_b = targets_b[j].cpu().numpy()  # the second label
                    pre_i = predicted[j].cpu().numpy()
                    conf_mat[cate_i_a, pre_i] += lam
                    conf_mat[cate_i_b, pre_i] += 1 - lam
            else: # 不然就是一般的训练过程
                outputs = model(inputs)
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
            loss_sigma.append(loss.item())
            probs = F.softmax(outputs, dim=1)  # 模型本身是不带softmax的 如果是用nn.CrossEntropy会自行softmax处理之后再计算
            outputs_append.append(probs.detach())
            acc_avg = conf_mat.trace() / conf_mat.sum()  # accuracy_average of each batch

        return np.mean(loss_sigma), acc_avg, conf_mat, label_append_a, label_append_b, outputs_append, lam_append

    @ staticmethod
    # cutout
    def train_cutout(data_loader, model, loss_f, optimizer, epoch_id, gpu, max_epoch, num_classes, mix_prob):
        ''' 这个函数是训练单个epoch
        :param data_loader: 就是用DataLoader函数导出来的batches
        :param model:
        :param loss_f: nn.CrossEntropyLoss 这种就可以 FocalLoss的写法应该和ce的是一致的咯
        :param optimizer: optim.SGD
        :param epoch_id: 第几个epoch
        :param gpu: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        :param max_epoch: total epoches
        :param num_classes: class number
        :return: loss_train, acc_train, mat_train, y_true_train, y_outputs_train
                 训练损失     训练准确率   混淆矩阵   真实label     预测的概率（对于每个class都有，二维list）
        '''
        model.train()
        # Some models use modules which have different training and evaluation behavior, such as batch normalization.
        # To switch between these modes, use model.train() or model.eval() as appropriate.

        conf_mat = np.zeros((num_classes, num_classes))  # confusion matrix 每次epoch都清空
        loss_sigma = []  # 损失函数  记录每一次epoch每一个batch的training loss
        label_append = []
        outputs_append = []

        for i, data in enumerate(data_loader):
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
            r = np.random.rand(1)
            if r < mix_prob:
                inputs = cutout(length=30, img=inputs,gpu=gpu).cutout_img() # 就加上这一句就行了
            outputs = model(inputs)
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
            probs = F.softmax(outputs, dim=1)  # 模型本身是不带softmax的 如果是用nn.CrossEntropy会自行softmax处理之后再计算
            label_append.append(labels.detach())
            outputs_append.append(probs.detach())  # 这里放在outputs里面的是对应的概率哦
            # average accuracy of each batch
            acc_avg = conf_mat.trace() / conf_mat.sum()  # accuracy_average of each batch

        return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append

    @staticmethod
    # manifold mixup, patchup
    def train_manifold_mixup(data_loader, model, loss_f, optimizer, epoch_id, gpu, max_epoch, num_classes, mixup_alpha):
        model.train()

        conf_mat = np.zeros((num_classes, num_classes))  # confusion matrix 每次epoch都清空
        loss_sigma = []  # 损失函数  记录每一次epoch每一个batch的training loss
        outputs_append = []
        label_append_a = []
        label_append_b = []
        lam_append = []

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
            outputs, loss = model(inputs, labels)
            optimizer.zero_grad()
            # optimizer.module.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计预测信息
            # _, predicted = torch.max(outputs.data, 1)
            # 统计混淆矩阵记录
            # for j in range(len(labels)):
            #     cate_i = labels[j].cpu().numpy()
            #     pre_i = predicted[j].cpu().numpy()
            #     conf_mat[cate_i, pre_i] += 1.
            for j in range(len(labels)):
                conf_mat[4,4]+=1
            # 统计loss
            loss_sigma.append(loss.item())
            # save labels and outputs to calculate ROC_auc and PR_auc
            # probs = F.softmax(outputs, dim=1)  # 模型本身是不带softmax的 如果是用nn.CrossEntropy会自行softmax处理之后再计算
            # label_append.append(labels.detach())
            # outputs_append.append(probs.detach())  # 这里放在outputs里面的是对应的概率哦
            # average accuracy of each batch
            # acc_avg = conf_mat.trace() / conf_mat.sum()  # accuracy_average of each batch
            acc_avg = 0.98
            # 因为计算出来的这些值都没有什么实际的意义 所以随便输出什么都没关系 只是为了和写的cal_auc_mixup能够配套才这么写的
            # 真正有意义的 只有模型的参数 这个在ModelTrainer.valid()里面才用
        return np.mean(loss_sigma), acc_avg, conf_mat, label_append_a, label_append_b, outputs_append, lam_append

    @staticmethod
    def train_manifold_mixup_apex(data_loader, model, loss_f, optimizer, epoch_id, gpu, max_epoch, num_classes, mixup_alpha):
        model.train()

        conf_mat = np.zeros((num_classes, num_classes))  # confusion matrix 每次epoch都清空
        loss_sigma = []  # 损失函数  记录每一次epoch每一个batch的training loss
        outputs_append = []
        label_append_a = []
        label_append_b = []
        lam_append = []

        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
            outputs, loss = model(inputs, labels)
            # add apex
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            for j in range(len(labels)):
                conf_mat[4,4]+=1
            # 统计loss
            loss_sigma.append(loss.item())
            # save labels and outputs to calculate ROC_auc and PR_auc
            # probs = F.softmax(outputs, dim=1)  # 模型本身是不带softmax的 如果是用nn.CrossEntropy会自行softmax处理之后再计算
            # label_append.append(labels.detach())
            # outputs_append.append(probs.detach())  # 这里放在outputs里面的是对应的概率哦
            # average accuracy of each batch
            # acc_avg = conf_mat.trace() / conf_mat.sum()  # accuracy_average of each batch
            acc_avg = 0.98
            # 因为计算出来的这些值都没有什么实际的意义 所以随便输出什么都没关系 只是为了和写的cal_auc_mixup能够配套才这么写的
            # 真正有意义的 只有模型的参数 这个在ModelTrainer.valid()里面才用
        return np.mean(loss_sigma), acc_avg, conf_mat, label_append_a, label_append_b, outputs_append, lam_append

    @staticmethod  # valid不需要做任何变化的
    def valid(data_loader, model, loss_f, gpu, num_classes):
        model.eval()

        conf_mat = np.zeros((num_classes, num_classes))
        loss_sigma = []
        label_append = []
        outputs_append = []

        with torch.no_grad():
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
