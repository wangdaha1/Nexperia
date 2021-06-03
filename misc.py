import torch
import sys
from bisect import bisect_right
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics


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
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' or mode == 'roc_auc' or mode == 'pr_auc' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()

# 计算
def cal_auc(y_true_train, y_outputs_train):
    '''
    :param y_true_train: y的真实label 不是onehot 得看上面那个modeltrainer函数的输出了
    :param y_outputs_train: y对于真实label的预测概率
    :return:
    '''
    # 其实sklearn.metrics里面有很多函数可以用的
    y_true = torch.cat(y_true_train).cpu()
    y_score = torch.cat(y_outputs_train).cpu()
    # 把多分类问题看作是二分类问题了 good(0)/bad(1)
    # 没有很搞懂这些个函数的作用
    roc_auc = metrics.roc_auc_score(y_true != 4, 1. - y_score[:, 4])
    precision, recall, _ = metrics.precision_recall_curve(y_true != 4, 1. - y_score[:, 4])
    pr_auc = metrics.auc(recall, precision)
    fpr, tpr, thresholds = metrics.roc_curve(y_true != 4, 1. - y_score[:, 4])
    fpr_98 = fpr[np.where(tpr >= 0.98)[0][0]]
    # fpr_991 = fpr[np.where(tpr >= 0.991)[0][0]]
    # fpr_993 = fpr[np.where(tpr >= 0.993)[0][0]]
    # fpr_995 = fpr[np.where(tpr >= 0.995)[0][0]]
    # fpr_997 = fpr[np.where(tpr >= 0.997)[0][0]]
    # fpr_999 = fpr[np.where(tpr >= 0.999)[0][0]]
    # fpr_1 = fpr[np.where(tpr == 1.)[0][0]]
    return roc_auc, pr_auc, fpr_98

