####

import torch
import sys
from bisect import bisect_right
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
import cv2
from PIL import Image
import random

class Logger(object):
    """
    save the output of the console into a log file
    """
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# 混淆矩阵绘制
def show_confMat(confusion_mat, classes, set_name, out_dir, verbose=False):
    """
    混淆矩阵绘制
    :param confusion_mat: 混淆矩阵
    :param classes: 类别名
    :param set_name: trian/valid
    :param out_dir: 输出的路径 in our case, the results fold
    :param verbose: 意思就是设置运行的时候不显示详细信息
    :return: nothing 就是把图片到results里面了
    """
    cls_num = len(classes)
    # 归一化
    confusion_mat_N = confusion_mat.copy()

    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(len(classes)):
            confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 显示

    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + set_name + '.png'))
    # plt.show()
    plt.close()

    # print information (only in the final epoch, see main.py)
    # 打印了recall和precision值
    # recall: TP/(TP+FN)
    # precision: TP/(TP+FP)
    if verbose:
        print("Confusion matrix of "+set_name+" data")
        for i in range(cls_num):# for each class, print the recall and precision
            print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
                classes[i], np.sum(confusion_mat[i, :]), confusion_mat[i, i],
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])),
                confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i]))))

# 绘制训练和验证集的loss/acc/roc_auc/pr_auc曲线
def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值  acc/loss values
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    assert mode in ['loss', 'acc', 'roc_auc', 'pr_auc', 'fpr_98']
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' or mode == 'roc_auc' or mode == 'pr_auc' or mode == 'fpr_98' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()

# 计算roc auc fpr_98啥的
def cal_auc(y_true_train, y_outputs_train, good_label):
    '''
    :param y_true_train: y的真实label数值 不是onehot
    :param y_outputs_train: 预测概率矩阵
    :param good_label: pass图片在原来的数据中的数值
    :return:
    '''
    assert type(good_label) is int
    # 其实sklearn.metrics里面有很多函数可以用的
    y_true = torch.cat(y_true_train).cpu()   # 拼接
    y_score = torch.cat(y_outputs_train).cpu()
    # 把多分类问题看作是二分类问题了 good(0)/bad(1)
    # FPR TPR Precision Recall 就是拿着TP FP TN FN 算来算去的四个东西 谁记得清啊
    # roc_auc计算的是Reception Operator Curve (FPR, TPR)的曲线围成面积auc 越大越好
    roc_auc = metrics.roc_auc_score(y_true != good_label, 1. - y_score[:, good_label])
    # pr_auc计算的是Precision Recall曲线围成面积auc 越大越好
    precision, recall, _ = metrics.precision_recall_curve(y_true != good_label, 1. - y_score[:, good_label])
    pr_auc = metrics.auc(recall, precision)
    # fpr_98
    fpr, tpr, thresholds = metrics.roc_curve(y_true != good_label, 1. - y_score[:, good_label])
    fpr_98 = fpr[np.where(tpr >= 0.98)[0][0]]
    # fpr_991 = fpr[np.where(tpr >= 0.991)[0][0]]
    # fpr_993 = fpr[np.where(tpr >= 0.993)[0][0]]
    # fpr_995 = fpr[np.where(tpr >= 0.995)[0][0]]
    # fpr_997 = fpr[np.where(tpr >= 0.997)[0][0]]
    # fpr_999 = fpr[np.where(tpr >= 0.999)[0][0]]
    # fpr_1 = fpr[np.where(tpr == 1.)[0][0]]
    return roc_auc, pr_auc, fpr_98

def cal_auc_mixup(y_true_train_a, y_true_train_b, lams,  y_outputs_train, good_label):
    '''
    :param y_true_train_a/b: y的真实label数值
    :param lam: lambda in mixup
    :param y_outputs_train: 预测概率矩阵
    :param good_label: pass图片在原来的数据中的数值
    :param lams: 一个长度和y相同的lam list
    :return:
    '''
    # assert type(good_label) is int
    # y_true_a = torch.cat(y_true_train_a).cpu()
    # y_true_b = torch.cat(y_true_train_b).cpu()
    # y_score = torch.cat(y_outputs_train).cpu()
    # roc_auc_a = metrics.roc_auc_score(y_true_a != good_label, 1. - y_score[:, good_label])
    # roc_auc_b = metrics.roc_auc_score(y_true_b != good_label, 1. - y_score[:, good_label])
    # precision_a, recall_a, _ = metrics.precision_recall_curve(y_true_a != good_label, 1. - y_score[:, good_label])
    # precision_b, recall_b, _ = metrics.precision_recall_curve(y_true_b != good_label, 1. - y_score[:, good_label])
    # pr_auc_a = metrics.auc(recall_a, precision_a)
    # pr_auc_b = metrics.auc(recall_b, precision_b)
    # fpr_a, tpr_a, _ = metrics.roc_curve(y_true_a != good_label, 1. - y_score[:, good_label])
    # fpr_b, tpr_b, _ = metrics.roc_curve(y_true_b != good_label, 1. - y_score[:, good_label])
    # fpr_98_a = fpr_a[np.where(tpr_a >= 0.98)[0][0]]
    # fpr_98_b = fpr_b[np.where(tpr_b >= 0.98)[0][0]]
    # return lams*roc_auc_a+(1-lams)*roc_auc_b, lams*pr_auc_a+(1-lams)*pr_auc_b, lams*fpr_98_a+(1-lams)*fpr_98_b
    return 0.98, 0.98, 0.05

# new data augmentation
class SharpenImage(object):
    """Sharpen the inputted images"""
    def __init__(self, p=0.9):
        assert (isinstance(p, float))
        self.p = p
        self.kernel_sharpen_1 = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]])

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            image = np.array(img).copy()
            output_1 = cv2.filter2D(image, -1, self.kernel_sharpen_1)
            return Image.fromarray(output_1.astype('uint8'))
        else:
            return img

class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))    # 2020 07 26 or --> and
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w , channels = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, channels), p=[signal_pct, noise_pct/2., noise_pct/2.])
            img_[mask == 1] = 255   # 盐噪声
            img_[mask == 2] = 0     # 椒噪声
            return Image.fromarray(img_.astype('uint8'))
        else:
            return img

# inputs mixup
def mixup_data(x, y, alpha, gpu):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda(gpu)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# remix
def remix_lam(label_i, label_j, kappa, lam, tau):
    '''
    :param nums: number of samples in different classes
    :param label_i: orders of samples, list
    :param label_j:
    :param kappa:
    :param lam:
    :param tau:
    :return: lam_list
    '''
    nums = [2, 1830, 748, 532, 39446, 3399, 7527, 166, 10383]
    lam_list = []
    for count in range(0, len(label_i)):
        if nums[label_i[count]] / nums[label_j[count]] >= kappa and lam < tau:
            lam_list.append(0)
        elif nums[label_i[count]] / nums[label_j[count]] <= 1 / kappa and (1 - lam) < tau:
            lam_list.append(1)
        else:
            lam_list.append(lam)
    return lam_list

# cutmix
def rand_bbox(size, lam):
    '''
    CutMix 生成剪裁区域
    :param size: 样本的size和
    :param lam: 生成的随机lamda值
    :return:
    '''
    # inputs.size(): torch.Size([128, 3, 224, 224])
    W = size[2]
    H = size[3]
    # 1.论文里的公式2，求出B的rw,rh
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    # 2.论文里的公式2，求出B的rx,ry（bbox的中心点）
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 限制坐标区域不超过样本大小
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # 3.返回剪裁B区域的坐标值
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x,y, alpha, gpu):
    '''Returns cutmixed inputs, pairs of targets, and lambda
    参考https://blog.csdn.net/weixin_38715903/article/details/103999227
    '''

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda(gpu)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # 根据剪裁区域坐标框的值调整lam的值
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

# cutout
class cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self,  length, img, gpu):
        self.n_holes = np.random.randint(1,4) # 1 2 3
        self.length = length
        self.img = img
        self.gpu=gpu

    def cutout_img(self):
        """
        Args:
            img (Tensor): Tensor image of size (Batchsize, Channel, W, H).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        w = self.img.size(2)
        h = self.img.size(3)

        mask = np.ones((w, h), np.float32)

        for n in range(self.n_holes):
            x = np.random.randint(w)
            y = np.random.randint(h)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[x1: x2, y1: y2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(self.img)
        mask = mask.cuda(self.gpu)
        img = self.img * mask
        img  = img.cuda(self.gpu)

        return img

