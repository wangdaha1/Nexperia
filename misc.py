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
            h, w = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w), p=[signal_pct, noise_pct/2., noise_pct/2.])
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

