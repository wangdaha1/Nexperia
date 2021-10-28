
import sys
import os
import argparse
import torch.nn as nn
import torch
from torchvision import models
from data_loader import get_dataloader
from loss_func import FocalLoss, NFLandRCE, NormalizedFocalLoss, NormalizedCrossEntropy
import torch.optim as optim
from misc import show_confMat, plot_line, cal_auc, Logger
from trainer import WarmupMultiStepLR, ModelTrainer
from datetime import datetime
import numpy as np
from models import MLPMixer
from checkpoint import save_checkpoint, takeFirst
import torch.nn.functional as F
from models import ManifoldMixupModel, PatchUpModel, Remix_ManifoldMixupModel



parser = argparse.ArgumentParser(description='Nexperia testing')
parser.add_argument('--cuda_visible_devices', default= '8', help='assign the gpu')
parser.add_argument('--recorded_file', type = str,default='07-31_mixup_0.3_newaug')  # 这里输入之前记录的文件名
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_data', default='Jan') # Test, Jan, Feb, Mar
parser.add_argument('--model_saved', default="_best1.pth") # saved model file
parser.add_argument('--model', default='resnet50')
parser.add_argument('--good_label', default=4)
parser.add_argument('--loss_func', default=nn.CrossEntropyLoss())
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

def ModelTester(data_loader, model, loss_f, device, num_classes):
    '''
    test model on batch3
    return: loss_avg, acc_avg, conf_mat, label_append, outputs_append
    '''

    model.eval() # 进入测试状态
    conf_mat = np.zeros((num_classes, num_classes))
    loss_sigma = []
    label_append = []
    outputs_append = []

    for i, data in enumerate(data_loader):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)

        # 统计混淆矩阵
        for j in range(len(labels)):
            cate_i = labels[j].cpu().numpy()
            pre_i = predicted[j].cpu().numpy()
            conf_mat[cate_i, pre_i] += 1.

        # 统计loss
        loss = loss_f(outputs, labels)
        loss_sigma.append(loss.item())

        # save labels and outputs to calculate ROC_auc and PR_auc
        probs = F.softmax(outputs, dim=1)
        label_append.append(labels.detach())
        outputs_append.append(probs.detach())

    acc_avg = conf_mat.trace() / conf_mat.sum()

    return np.mean(loss_sigma),acc_avg, conf_mat, label_append, outputs_append



# 测试模型
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 记录test的结果的位置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    saved_dir = os.path.join(BASE_DIR, "results", args.recorded_file)
    log_dir = os.path.join(saved_dir,"test")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(os.path.join(log_dir, 'log_test.txt'), sys.stdout)
    print("Start testing the model "+args.model_saved + ' on data '+ args.test_data)
    start_time = datetime.now()
    start_time_str = datetime.strftime(start_time, '%m-%d_%H-%M')
    print(start_time_str)

    class_names = ('Others', 'Marking_defect', 'Lead_glue', 'Lead_defect', 'Pass', 'Foreign_material', 'Empty_pocket', 'Device_flip', 'Chipping')
    num_classes = len(class_names)

    #=============== load the testing data ================#

    # batch3_train_dir = os.path.join('/import/home/xwangfy/projects_xwangfy/data_nexperia/all','train' )
    # batch3_valid_dir = os.path.join('/import/home/xwangfy/projects_xwangfy/data_nexperia/all','val')
    # batch3_test_dir = os.path.join('/import/home/xwangfy/projects_xwangfy/data_nexperia/all','test')
    #
    # test_data1 = get_dataloader(batch_size=args.batch_size).testdata(batch3_train_dir)
    # test_data2 = get_dataloader(batch_size=args.batch_size).testdata(batch3_valid_dir)
    # test_data3 = get_dataloader(batch_size=args.batch_size).testdata(batch3_test_dir)
    # combined_data = torch.utils.data.ConcatDataset([test_data1, test_data2, test_data3])
    # test_loader = torch.utils.data.DataLoader(dataset=combined_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    assert args.test_data in ['Test', 'Jan', 'Feb', 'Mar']
    if args.test_data == 'Test':
        test_loader = get_dataloader(batch_size=args.batch_size).testloader()
    elif args.test_data == 'Jan':
        test_loader = get_dataloader(batch_size=args.batch_size).testloader_Jan()
    elif args.test_data == 'Feb':
        test_loader = get_dataloader(batch_size=args.batch_size).testloader_Feb()
    elif args.test_data == 'Mar':
        test_loader = get_dataloader(batch_size=args.batch_size).testloader_Mar()

    #====================== model ==========================#
    checkpoint_file_loc = os.path.join(saved_dir, args.model_saved)
    assert args.model in ['MLPMixer', 'resnet18', 'resnet34', 'resnet50', 'resnet34_mmix', 'resnet50_mmix',
                          'resnet34_patchup', 'resnet50_patchup', 'resnet50_rmmix']
    if args.model == 'MLPMixer':
        model = MLPMixer(patch_size=args.model_MLPMixer_patch_size, channel_dim=args.model_MLPMixer_channel_dim, \
                         num_blocks=args.model_MLPMixer_num_blocks, fig_size=args.model_MLPMixer_fig_size)  # 这些参数必须要吗
        num_ftrs = model.out_fc.in_features
        model.out_fc = nn.Linear(num_ftrs, num_classes)
    elif args.model =='resnet18':
        model  = models.resnet18(num_classes=num_classes)
    elif args.model == 'resnet34':
        model = models.resnet34(num_classes=num_classes)
    elif args.model=='resnet50':
        model = models.resnet50(num_classes=num_classes)
    elif args.model == 'resnet34_mmix':
        model = ManifoldMixupModel(models.resnet34(), num_classes=num_classes, alpha=0)
    elif args.model == 'resnet50_mmix':
        model = ManifoldMixupModel(models.resnet50(), num_classes=num_classes, alpha=0)  # 测试的时候是不需要mix的
    elif args.model =='resnet34_patchup':
        model = PatchUpModel(models.resnet34(), num_classes=num_classes, block_size=7, gamma=0, patchup_type='hard')
    elif args.model =='resnet50_patchup':
        model = PatchUpModel(models.resnet50(), num_classes=num_classes, block_size=7, gamma=0, patchup_type='hard')
    elif args.model == 'resnet50_rmmix':
        model = Remix_ManifoldMixupModel(models.resnet50(), num_classes=num_classes,alpha=0,kappa=1, tau = -1)
    checkpoint = torch.load(checkpoint_file_loc)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)

    #===================== loss function ====================#
    criterion = args.loss_func

    # ====================== Test model =====================#
    # loss_rec = []
    # acc_rec = []
    # roc_auc_rec = []
    # pr_auc_rec = []

    loss_test , acc_test, mat_test, y_true_test, y_outputs_test = ModelTester(test_loader, model, criterion, device, num_classes)

    roc_auc_test, pr_auc_test, fpr_test = cal_auc(y_true_test, y_outputs_test, args.good_label)

    print("test Acc:{:.2%} test loss:{:.2%} test fpr_98:{:.2%} test AUC:{:.2%} ".format(acc_test,loss_test, fpr_test, roc_auc_test))

    # loss_rec.append(loss_test)
    # acc_rec.append(acc_test)
    # roc_auc_rec.append(roc_auc_test)
    # pr_auc_rec.append(pr_auc_test)

    np.save(os.path.join(log_dir, 'loss_rec.npy'), loss_test)
    np.save(os.path.join(log_dir, 'acc_rec.npy'), acc_test)
    np.save(os.path.join(log_dir, 'roc_auc_rec.npy'), roc_auc_test)
    np.save(os.path.join(log_dir, 'pr_auc_rec.npy'), pr_auc_test)

    # save confusion matrix
    show_confMat(mat_test, class_names, "valid", log_dir, verbose=True)

    print(" Finished Test!!!! {} ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M')))
    print("\n")
    f = open(os.path.join(log_dir, 'log_test.txt'), 'a')
    sys.stdout = f
    sys.stderr = f

