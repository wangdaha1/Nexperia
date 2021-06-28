
import sys
import os
import argparse
import torch.nn as nn
import torch
from torchvision import models
from data_loader import get_dataloader
from loss_func import FocalLoss, NFLandRCE, NormalizedFocalLoss, NormalizedFocalLoss_10
import torch.optim as optim
from misc import show_confMat, plot_line, cal_auc, Logger
from trainer import WarmupMultiStepLR, ModelTrainer
from datetime import datetime
import numpy as np
from models import MLPMixer
from checkpoint import save_checkpoint, takeFirst
from SELFIE import Correcter, SELFIE_ModelTrainer

# 这个东西得写在main函数外面
os.environ['CUDA_VISIBLE_DEVICES'] = "8"

# 学一学这个传参数的登西 还不知道咋用额 要学一下写bash？
parser = argparse.ArgumentParser(description='Nexperia training')
parser.add_argument('--recording_file', type = str,default='SELFIE_second_try')
parser.add_argument('--BATCH_SIZE', type=int, default=128)
parser.add_argument('--MAX_EPOCH', type=int, default=150)
parser.add_argument('--SELFIE_start_epoch', default=120)
parser.add_argument('--SELFIE_start_record_NOISE_SET_epoch',default=140)
parser.add_argument('--corrector', default=Correcter(noise_rate=0.01, history_length=25, threshold = 0.01))
parser.add_argument('--warmup_LR', type=bool, default=True) # 试验的时候就不要用warmupLR了
parser.add_argument('--milestones', type = list, default=[50,100])
parser.add_argument('--LR', type = float, default=0.1)
parser.add_argument('--MODEL', default='resnet34')
# parser.add_argument('--MODEL_MLPMixer_patch_size', type=int, default=-1)
# parser.add_argument('--MODEL_MLPMixer_channel_dim', type=int, default=32)
# parser.add_argument('--MODEL_MLPMixer_num_blocks', type=int, default=16)
# parser.add_argument('--MODEL_MLPMixer_fig_size', default=(224,224))
parser.add_argument('--loss_func', default=nn.CrossEntropyLoss())
parser.add_argument('--loss_C_selection', default=nn.CrossEntropyLoss(reduction='none'))
args = parser.parse_args()



def main_SELFIE():

    # some basic settings, will not change
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #'/import/home/xwangfy/projects_xwangfy/nexperia_wxs'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ('chipping', 'device_flip', 'empty_pocket', 'foreign_material', 'good', 'lead_defect', 'lead_glue', 'marking_defect')
    num_classes = len(class_names)

    # record the results
    # 换一种文件名看看呢
    # start_time_str = start_time_str + "_10NFL"
    log_dir = os.path.join(BASE_DIR,  "results", args.recording_file)
    log_dir_train = os.path.join(log_dir, "train")
    log_dir_val = os.path.join(log_dir, "val")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(log_dir_train):
        os.makedirs(log_dir_train)
    if not os.path.exists(log_dir_val):
        os.makedirs(log_dir_val)
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'), sys.stdout)

    # record the start time and the related parameters information
    start_time = datetime.now()
    start_time_str = datetime.strftime(start_time, '%m-%d_%H-%M')
    print(start_time_str)
    print(args)  # 把args也记录下 怎么让那个模型的细节不输出啊


    #=============== load the training data ================#
    train_loader = get_dataloader(batch_size=args.BATCH_SIZE).trainloader(path=True)
    valid_loader = get_dataloader(batch_size=args.BATCH_SIZE).validloader(path=False)

    #====================== Model ==========================#
    assert args.MODEL in ['MLPMixer', 'resnet34', 'resnet18']
    if args.MODEL == 'MLPMixer':
        MODEL = MLPMixer(patch_size=args.MODEL_MLPMixer_patch_size, channel_dim=args.MODEL_MLPMixer_channel_dim, \
                         num_blocks=args.MODEL_MLPMixer_num_blocks, fig_size=args.MODEL_MLPMixer_fig_size)
        num_ftrs = MODEL.out_fc.in_features
        MODEL.out_fc = nn.Linear(num_ftrs, num_classes)

    elif args.MODEL == 'resnet34':
        MODEL = models.resnet34()
        num_ftrs = MODEL.fc.in_features
        MODEL.fc = nn.Linear(num_ftrs, num_classes)  # 只改变了最后一层的输出个数

    elif args.MODEL == 'resnet18':
        MODEL = models.resnet18()
        num_ftrs = MODEL.fc.in_features
        MODEL.fc = nn.Linear(num_ftrs, num_classes)

    MODEL.to(device)

    #===================== loss function ====================#
    criterion = args.loss_func
    # criterion = NFLandRCE(alpha=10, beta=0.1,gamma=2)
    # criterion = nn.CrossEntropyLoss()
    # criterion = NormalizedFocalLoss(num_classes=num_classes, alpha=None, gamma=2)

    #====================== optimizer =======================#
    optimizer = optim.SGD(MODEL.parameters(), lr=args.LR, momentum=0.9, weight_decay=1e-4)
    if args.warmup_LR ==True:
        scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                      milestones=args.milestones,
                                      gamma=0.1,
                                      warmup_factor=0.1,
                                      warmup_iters=5,
                                      warmup_method="linear",
                                      last_epoch=-1)


    #====================== Train model =====================#
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    roc_auc_rec = {"train": [], "valid": []}
    pr_auc_rec = {"train": [], "valid": []}
    best_auc_epoch_checkpoints = [[0,0,{'None':None}] for _ in range(0,3)] # choose the best three models according to auc
    last_epoch_checkpoints = [[0,{'None':None}] for _ in range(0,3)] # 保存最后三个epoch的模型

    for epoch in range(0, args.MAX_EPOCH):
        state = 'warm_up' if epoch<args.SELFIE_start_epoch-1 else 'SELFIE'
        args.corrector.record_noisy_samples = False if epoch < args.SELFIE_start_record_NOISE_SET_epoch-1 else True
        loss_train, acc_train, mat_train, y_true_train, y_outputs_train = \
            SELFIE_ModelTrainer(corrector=args.corrector, state=state).train(train_loader, MODEL, criterion, optimizer, epoch, device, args.MAX_EPOCH, args.loss_C_selection)
        loss_valid, acc_valid, mat_valid, y_true_valid, y_outputs_valid = \
            SELFIE_ModelTrainer(corrector=args.corrector, state=state).valid(valid_loader, MODEL, criterion, device)

        roc_auc_train, pr_auc_train, fpr_98_train = cal_auc(y_true_train, y_outputs_train)
        roc_auc_valid, pr_auc_valid, fpr_98_valid = cal_auc(y_true_valid, y_outputs_valid)

        # 每个epoch都打印结果
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} Train fpr_98:{:.2%}\
               Valid fpr_98:{:.2%} Train AUC:{:.2%} Valid AUC:{:.2%} LR:{}".format(
                epoch + 1, args.MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, fpr_98_train, fpr_98_valid, roc_auc_train,
                roc_auc_valid, optimizer.param_groups[0]["lr"]))
        if args.warmup_LR == True:
            scheduler.step()  # 用可变学习率 Update learning rate


        # record the results of all epochs
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        roc_auc_rec["train"].append(roc_auc_train), roc_auc_rec["valid"].append(roc_auc_valid)
        pr_auc_rec["train"].append(pr_auc_train), pr_auc_rec["valid"].append(pr_auc_valid)

        # 这是每个epoch都save 都画图吗 这样就算没跑完也能看当时的结果啦
        np.save(os.path.join(log_dir_train, 'loss_rec.npy'), loss_rec["train"])
        np.save(os.path.join(log_dir_train, 'acc_rec.npy'), acc_rec["train"])
        np.save(os.path.join(log_dir_train, 'roc_auc_rec.npy'), roc_auc_rec["train"])
        np.save(os.path.join(log_dir_train, 'pr_auc_rec.npy'), pr_auc_rec["train"])

        np.save(os.path.join(log_dir_val, 'loss_rec.npy'), loss_rec["valid"])
        np.save(os.path.join(log_dir_val, 'acc_rec.npy'), acc_rec["valid"])
        np.save(os.path.join(log_dir_val, 'roc_auc_rec.npy'), roc_auc_rec["valid"])
        np.save(os.path.join(log_dir_val, 'pr_auc_rec.npy'), pr_auc_rec["valid"])

        # train和valid的confusion matrix图片都储存下来
        # verbose=epoch ==MAX_EPOCH-1指的是只有在最后一个epoch画最终的confusion matrix的时候才打印信息
        # 在log日志里 先打印train再valid
        show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == args.MAX_EPOCH - 1)
        show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == args.MAX_EPOCH - 1)

        # train和valid的loss/acc图片储存下来
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)
        # train和valid的roc_auc/pr_auc曲线
        plot_line(plt_x, roc_auc_rec["train"], plt_x, roc_auc_rec["valid"], mode="roc_auc", out_dir=log_dir)
        plot_line(plt_x, pr_auc_rec["train"], plt_x, pr_auc_rec["valid"], mode="pr_auc", out_dir=log_dir)

        # save the models
        checkpoint = {"model_state_dict": MODEL.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch + 1,
                      "best_auc": roc_auc_valid}
        # save the best 3 models according to auc 第一个跑完一半的要求还是保留一下
        if epoch > (args.MAX_EPOCH / 2) and best_auc_epoch_checkpoints[2][0] < roc_auc_valid:
            # best_auc里面从大到小排序 大于最后一名的auc就进入队列
            new_best_element = [roc_auc_valid, epoch+1, checkpoint]
            best_auc_epoch_checkpoints.pop() # 去掉最后一名
            best_auc_epoch_checkpoints.append(new_best_element) # 新的加进来 这里是不能有返回值的吗
            # 按照auc排序
            best_auc_epoch_checkpoints.sort(key=takeFirst)
            best_auc_epoch_checkpoints.reverse() # 这两行居然不能连起来写的吗
            # 每次出现更新就把checkpoint_best_1st/2nd/3rd都更新一下吧  这样没跑完也能看结果了
            for i in range(0, 3):
                save_checkpoint(log_dir, best_auc_epoch_checkpoints[i][2], order=i + 1, is_best=True)

        # save the last 3 models 保存最近的最后的三个模型
        new_last_element = [epoch+1, checkpoint]
        last_epoch_checkpoints.pop() # 去掉最后一个
        last_epoch_checkpoints.insert(0, new_last_element)
        for i in range(0,3):
            save_checkpoint(log_dir, last_epoch_checkpoints[i][1], order=i+1, is_last=True)

    print("print the Refurbished samples' paths and original labels and refurbished labels")
    print(args.corrector.NOISY_SET)

    print("Finished!!!! {}, best aucs: {} in :{} epochs (from best1 to best3). ".format(\
        datetime.strftime(datetime.now(), '%m-%d_%H-%M'), [i[0] for i in best_auc_epoch_checkpoints],
        [i[1] for i in best_auc_epoch_checkpoints]))

    end_time = datetime.now()
    end_time_str = datetime.strftime(end_time, '%m-%d_%H-%M')
    print(end_time_str)

    f = open(os.path.join(log_dir, 'log.txt'), 'a')
    sys.stdout = f
    sys.stderr = f


if __name__ == '__main__':
    main_SELFIE()

