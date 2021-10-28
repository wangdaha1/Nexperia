import sys
import os
import argparse
import torch.nn as nn
import torch
from torchvision import models
from data_loader import get_dataloader
from loss_func import FocalLoss, NFLandRCE, NormalizedFocalLoss, NormalizedCrossEntropy
import torch.optim as optim
from misc import show_confMat, plot_line, cal_auc, Logger, cal_auc_mixup
from trainer import WarmupMultiStepLR, ModelTrainer
from datetime import datetime
import numpy as np
from models import MLPMixer
from checkpoint import save_checkpoint, takeFirst, save_model
from models import ManifoldMixupModel, PatchUpModel, Remix_ManifoldMixupModel
from apex import amp

# PARAMETERS 都采用小写了
parser = argparse.ArgumentParser(description='Nexperia training')
parser.add_argument('--cuda_visible_devices', default='3', help='assign the gpu')
parser.add_argument('--recording_file', type=str, default='caogao', help='file of recording')
parser.add_argument('--loss_func', default=nn.CrossEntropyLoss())
# parser.add_argument('--loss_func', default=FocalLoss(num_classes=9, alpha=None, gamma=2))
parser.add_argument('--model', default='resnet50', help='backbone of model, choose from [resnet18, resnet34, resnet50, \
        resnet34_mmix, resnet50_mmix, resnet18_patchup, resnet34_patchup, resnet50_patchup, resnet18_rmmix, resnet50_rmmix, MLPMixer]')
parser.add_argument('--train', default='vanilla', # 这里的train都改成了apex对应的代码了
                    help='the type of training, choose from [vanilla, mixup, manifold_mixup, patchup, cutout, cutmix]')

# fixed parameters, do not need to change in usual case
parser.add_argument('--good_label', default=4, help='the label of pass img in the dataloader')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--max_epoch', type=int, default=150)
parser.add_argument('--warmup_lr', type=bool, default=True)  # 试验的时候就不要用warmuplr了
parser.add_argument('--milestones', type=list, default=[90, 120], help='two steps in warmup lr')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--imbalanced_sampler', type=bool, default=False)

# mixup related parameters
parser.add_argument('--mix_prob', default=1, type=float, help='used in mixup, cutmix, cutout')
parser.add_argument('--mixup_alpha', default=0.3, type=float, help='used in mixup, manifold mixup, cutmix')
# parser.add_argument('--remix_kappa', default=8, help='param in remix')
# parser.add_argument('--remix_tau', default=0.5, help='param in remix')
parser.add_argument('--patchup_gamma', default=0.5, type=float, help='used in patchup')

# Class-Balanced Loss Based on Effective Number of Samples
parser.add_argument('--weight_balance', default=False, type=bool, help='to use weighted sample losses or not')
# parser.add_argument('--CB_beta', default=0.999, help='param in paper weight_balance')

# MLPMixer model related parameters
# parser.add_argument('--model_MLPMixer_patch_size', type=int, default=-1)
# parser.add_argument('--model_MLPMixer_channel_dim', type=int, default=32)
# parser.add_argument('--model_MLPMixer_num_blocks', type=int, default=16)
# parser.add_argument('--model_MLPMixer_fig_size', default=(224,224))

args = parser.parse_args()

# 这个东西得写在main函数外面
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices


def main():
    # some basic settings, will not change
    cls2int = {
        'Others': 0,
        'Marking_defect': 1,
        'Lead_glue': 2,
        'Lead_defect': 3,
        'Pass': 4,
        'Foreign_material': 5,
        'Empty_pocket': 6,
        'Device_flip': 7,
        'Chipping': 8}
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # '/import/home/xwangfy/projects_xwangfy/nexperia_wxs'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = (
    'Others', 'Marking_defect', 'Lead_glue', 'Lead_defect', 'Pass', 'Foreign_material', 'Empty_pocket', 'Device_flip',
    'Chipping')
    num_classes = len(class_names)

    # record the results
    log_dir = os.path.join(BASE_DIR, "results", args.recording_file)
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
    print(args)  # 把args也记录下

    # =============== load the training data ================#
    dataloaders = get_dataloader(batch_size=args.batch_size, imbalanced_sampler=args.imbalanced_sampler)
    train_loader = dataloaders.trainloader()
    valid_loader = dataloaders.validloader()

    # ====================== Model ==========================#
    assert args.model in ['MLPMixer', 'resnet18', 'resnet34', 'resnet50', 'resnet34_mmix', 'resnet50_mmix', \
                          'resnet18_patchup', 'resnet34_patchup', 'resnet50_patchup', \
                          'resnet18_rmmix', 'resnet50_rmmix']
    if args.model == 'MLPMixer':
        MODEL = MLPMixer(patch_size=args.model_MLPMixer_patch_size, channel_dim=args.model_MLPMixer_channel_dim, \
                         num_blocks=args.model_MLPMixer_num_blocks, fig_size=args.model_MLPMixer_fig_size)
        num_ftrs = MODEL.out_fc.in_features
        MODEL.out_fc = nn.Linear(num_ftrs, num_classes)
    elif args.model == 'resnet18':
        MODEL = models.resnet18(num_classes=num_classes)
    elif args.model == 'resnet34':
        MODEL = models.resnet34(num_classes=num_classes)
        # MODEL = models.resnet34()
        # num_ftrs = MODEL.fc.in_features
        # MODEL.fc = nn.Linear(num_ftrs, num_classes)  # 只改变了最后一层的输出个数
    elif args.model == 'resnet50':
        MODEL = models.resnet50(num_classes=num_classes)
    elif args.model == 'resnet34_mmix':
        MODEL = ManifoldMixupModel(models.resnet34(), num_classes=num_classes, alpha=args.mixup_alpha)
    elif args.model == 'resnet50_mmix':
        MODEL = ManifoldMixupModel(models.resnet50(), num_classes=num_classes, alpha=args.mixup_alpha)
    elif args.model == 'resnet18_patchup':
        MODEL = PatchUpModel(models.resnet18(), num_classes=num_classes, block_size=7, gamma=args.patchup_gamma,
                             patchup_type='hard')
    elif args.model == 'resnet34_patchup':
        MODEL = PatchUpModel(models.resnet34(), num_classes=num_classes, block_size=7, gamma=args.patchup_gamma,
                             patchup_type='hard')
    elif args.model == 'resnet50_patchup':
        MODEL = PatchUpModel(models.resnet50(), num_classes=num_classes, block_size=7, gamma=args.patchup_gamma,
                             patchup_type='hard')
    elif args.model == 'resnet18_rmmix':
        MODEL = Remix_ManifoldMixupModel(models.resnet18(), num_classes=num_classes, alpha=args.mixup_alpha,
                                         kappa=args.remix_kappa, tau=args.remix_tau)
    elif args.model == 'resnet50_rmmix':
        MODEL = Remix_ManifoldMixupModel(models.resnet50(), num_classes=num_classes, alpha=args.mixup_alpha,
                                         kappa=args.remix_kappa, tau=args.remix_tau)
    MODEL.to(device)
    # ===================== loss function ====================#
    if args.weight_balance == False:
        criterion = args.loss_func
    else:  # 计算weight 这里先试试crossentropy的weighted version
        samples_per_cls = [2, 1830, 748, 532, 39446, 3399, 7527, 166, 10383]
        effective_num = 1.0 - np.power(args.CB_beta, samples_per_cls)
        weights = (1. - args.CB_beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        print("the sample_per_class we use: ", samples_per_cls)
        print("the beta we use: ", args.CB_beta)
        print("the corresponding weights for each class: ", weights)
        weights = torch.tensor(weights).float()
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)

    # criterion = NFLandRCE(alpha=10, beta=0.1,gamma=2)
    # criterion = nn.CrossEntropyLoss()
    # criterion = NormalizedFocalLoss(num_classes=num_classes, alpha=None, gamma=2)

    # ====================== optimizer =======================#
    optimizer = optim.SGD(MODEL.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    if args.warmup_lr == True:
        scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                      milestones=args.milestones,
                                      gamma=0.1,
                                      warmup_factor=0.1,
                                      warmup_iters=5,
                                      warmup_method="linear",
                                      last_epoch=-1)
    # add apex to train
    opt_level = 'O1'
    MODEL, optimizer = amp.initialize(MODEL, optimizer, opt_level=opt_level)

    # ====================== Train model =====================#
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    roc_auc_rec = {"train": [], "valid": []}
    pr_auc_rec = {"train": [], "valid": []}
    fpr_98_rec = {"train": [], "valid": []}
    # best_auc_epoch_checkpoints = [[0,0,{'None':None}] for _ in range(0,3)] # choose the best three models according to auc
    # last_epoch_checkpoints = [[0,{'None':None}] for _ in range(0,3)] # 保存最后三个epoch的模型
    topk = [0 for _ in range(3)];
    best_auc = 0  # use jasper's code

    assert args.train in ['vanilla', 'mixup', 'manifold_mixup', 'patchup', 'cutmix', 'cutout']
    for epoch in range(0, args.max_epoch):
        if args.train == 'vanilla':
            # DataLoader这个函数还蛮妙的 不需要在每个epoch里面都去重新定义，每跑一个epoch会自动更新成新的batches
            loss_train, acc_train, mat_train, y_true_train, y_outputs_train = \
                ModelTrainer.train_apex(train_loader, MODEL, criterion, optimizer, epoch, device, args.max_epoch,
                                   num_classes)
        elif args.train == 'mixup':
            loss_train, acc_train, mat_train, y_true_a_train, y_true_b_train, y_outputs_train, lams = \
                ModelTrainer.train_mixup_apex(train_loader, MODEL, criterion, optimizer, epoch, device, args.max_epoch, \
                                         num_classes, args.mixup_alpha, args.mix_prob)
        elif args.train == 'cutmix':
            loss_train, acc_train, mat_train, y_true_a_train, y_true_b_train, y_outputs_train, lams = \
                ModelTrainer.train_cutmix(train_loader, MODEL, criterion, optimizer, epoch, device, args.max_epoch, \
                                          num_classes, args.mixup_alpha, args.mix_prob)
        elif args.train == 'cutout':
            loss_train, acc_train, mat_train, y_true_train, y_outputs_train = \
                ModelTrainer.train_cutout(train_loader, MODEL, criterion, optimizer, epoch, device, args.max_epoch, \
                                          num_classes, args.mix_prob)
        elif args.train in ['manifold_mixup', 'patchup']:
            loss_train, acc_train, mat_train, y_true_a_train, y_true_b_train, y_outputs_train, lams = \
                ModelTrainer.train_manifold_mixup_apex(train_loader, MODEL, criterion, optimizer, epoch, device,
                                                  args.max_epoch, \
                                                  num_classes, args.mixup_alpha)

        loss_valid, acc_valid, mat_valid, y_true_valid, y_outputs_valid = \
            ModelTrainer.valid(valid_loader, MODEL, criterion, device, num_classes)

        if args.train in ['vanilla', 'cutout']:
            roc_auc_train, pr_auc_train, fpr_98_train = cal_auc(y_true_train, y_outputs_train, args.good_label)
        elif args.train in ['mixup', 'manifold_mixup', 'patchup', 'cutmix']:
            roc_auc_train, pr_auc_train, fpr_98_train = 0.98, 0.98, 0.05
        roc_auc_valid, pr_auc_valid, fpr_98_valid = cal_auc(y_true_valid, y_outputs_valid, args.good_label)

        # 每个epoch都打印结果
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} Train fpr_98:{:.2%}\
               Valid fpr_98:{:.2%} Train AUC:{:.2%} Valid AUC:{:.2%} LR:{}".format(
            epoch + 1, args.max_epoch, acc_train, acc_valid, loss_train, loss_valid, fpr_98_train, fpr_98_valid,
            roc_auc_train,
            roc_auc_valid, optimizer.param_groups[0]["lr"]))

        # 用可变学习率 Update learning rate
        if args.warmup_lr == True:
            scheduler.step()

        # record the results of all epochs
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)
        roc_auc_rec["train"].append(roc_auc_train), roc_auc_rec["valid"].append(roc_auc_valid)
        pr_auc_rec["train"].append(pr_auc_train), pr_auc_rec["valid"].append(pr_auc_valid)
        fpr_98_rec["train"].append(fpr_98_train), fpr_98_rec["valid"].append(fpr_98_valid)
        # 这是每个epoch都save 都画图 这样就算没跑完也能看当时的结果啦
        np.save(os.path.join(log_dir_train, 'loss_rec.npy'), loss_rec["train"])
        np.save(os.path.join(log_dir_train, 'acc_rec.npy'), acc_rec["train"])
        np.save(os.path.join(log_dir_train, 'roc_auc_rec.npy'), roc_auc_rec["train"])
        np.save(os.path.join(log_dir_train, 'pr_auc_rec.npy'), pr_auc_rec["train"])
        np.save(os.path.join(log_dir_train, 'fpr_98_rec.npy'), fpr_98_rec["train"])

        np.save(os.path.join(log_dir_val, 'loss_rec.npy'), loss_rec["valid"])
        np.save(os.path.join(log_dir_val, 'acc_rec.npy'), acc_rec["valid"])
        np.save(os.path.join(log_dir_val, 'roc_auc_rec.npy'), roc_auc_rec["valid"])
        np.save(os.path.join(log_dir_val, 'pr_auc_rec.npy'), pr_auc_rec["valid"])
        np.save(os.path.join(log_dir_val, 'fpr_98_rec.npy'), fpr_98_rec["valid"])

        # train和valid的confusion matrix图片都储存下来
        # verbose=epoch ==max_epoch-1指的是只有在最后一个epoch画最终的confusion matrix的时候才打印信息
        # 在log日志里 先打印train再valid
        show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == args.max_epoch - 1)
        show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == args.max_epoch - 1)

        # train和valid的loss/acc图片储存下来
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)
        # train和valid的roc_auc/pr_auc/fpr_98曲线
        plot_line(plt_x, roc_auc_rec["train"], plt_x, roc_auc_rec["valid"], mode="roc_auc", out_dir=log_dir)
        plot_line(plt_x, pr_auc_rec["train"], plt_x, pr_auc_rec["valid"], mode="pr_auc", out_dir=log_dir)
        plot_line(plt_x, fpr_98_rec["train"], plt_x, fpr_98_rec["valid"], mode="fpr_98", out_dir=log_dir)

        # save the models
        # use jasper's codes 但我实在是看不出来我自己的错在哪里了欸...
        if epoch > args.max_epoch / 2 - 10:
            # if best_auc < roc_auc_valid:  # 根据roc_auc_valid来排序的
            #     best_auc = roc_auc_valid  # reschedule the best_auc
            save_model(epoch, roc_auc_valid, log_dir, MODEL.state_dict(), args.max_epoch, topk)
        # checkpoint = {"model_state_dict": MODEL.state_dict(),
        #               "optimizer_state_dict": optimizer.state_dict(),
        #               "epoch": epoch + 1,
        #               "best_auc": roc_auc_valid}
        # save the best 3 models according to auc 第一个跑完一半的要求还是保留一下
        # if epoch > (args.max_epoch / 2) and best_auc_epoch_checkpoints[2][0] < roc_auc_valid:
        #
        #     # best_auc里面从大到小排序 大于最后一名的auc就进入队列
        #     new_best_element = [roc_auc_valid, epoch+1, checkpoint]
        #     best_auc_epoch_checkpoints.pop() # 去掉最后一名
        #     best_auc_epoch_checkpoints.append(new_best_element)
        #     # 按照auc排序
        #     best_auc_epoch_checkpoints.sort(key=takeFirst)
        #     best_auc_epoch_checkpoints.reverse() # 这两行居然不能连起来写
        #     # 每次出现更新就把checkpoint_best_1st/2nd/3rd都更新一下这样没跑完也能看结果了
        #     for i in range(0, 3):
        #         save_checkpoint(log_dir, best_auc_epoch_checkpoints[i][2], order=i + 1, is_best=True)
        #
        # # save the last 3 models 保存最近的最后的三个模型
        # new_last_element = [epoch+1, checkpoint]
        # last_epoch_checkpoints.pop() # 去掉最后一个
        # last_epoch_checkpoints.insert(0, new_last_element)
        # for i in range(0,3):
        #     save_checkpoint(log_dir, last_epoch_checkpoints[i][1], order=i+1, is_last=True)

    # print(" Finished!!!! {}, best aucs: {} in :{} epochs (from best1 to best3). ".format(\
    #     datetime.strftime(datetime.now(), '%m-%d_%H-%M'), [i[0] for i in best_auc_epoch_checkpoints],
    #     [i[1] for i in best_auc_epoch_checkpoints]))
    end_time = datetime.now()
    end_time_str = datetime.strftime(end_time, '%m-%d_%H-%M')
    print("Training is Finished at " + end_time_str + " !!!!")

    f = open(os.path.join(log_dir, 'log.txt'), 'a')
    sys.stdout = f
    sys.stderr = f


if __name__ == '__main__':
    main()
