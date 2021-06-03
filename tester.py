
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
import torch.nn.functional as F



os.environ['CUDA_VISIBLE_DEVICES'] = "5"

parser = argparse.ArgumentParser(description='Nexperia testing')
parser.add_argument('--recorded_file', type = str,default='MLP_Mixer_5_17-22_45_CE')  # 这里输入之前记录的文件名
parser.add_argument('--BATCH_SIZE', type=int, default=128)
parser.add_argument('--MODEL_saved', default="checkpoint_best_1st.pkl") # saved model file
parser.add_argument('--MODEL_type', default='MLPMixer')
parser.add_argument('--MODEL_MLPMixer_patch_size', type=int, default=-1)
parser.add_argument('--MODEL_MLPMixer_channel_dim', type=int, default=32)
parser.add_argument('--MODEL_MLPMixer_num_blocks', type=int, default=16)
parser.add_argument('--MODEL_MLPMixer_fig_size', default=(224,224))
parser.add_argument('--loss_func', default=nn.CrossEntropyLoss())
args = parser.parse_args()


def ModelTester(data_loader, model,loss_f,device):
    '''
    test model on batch3
    return: loss_avg, acc_avg, conf_mat, label_append, outputs_append
    '''

    model.eval() # 进入测试状态
    conf_mat = np.zeros((8, 8))
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

    # 路径定义好 每次训练的新的结果模型都需要改
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    saved_dir = os.path.join(BASE_DIR, "results", args.recorded_file)

    # 记录test的结果的位置

    log_dir = os.path.join(saved_dir,"test")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(os.path.join(log_dir, 'log_test.txt'), sys.stdout)
    print("Start testing the model"+args.MODEL_saved)
    start_time = datetime.now()
    start_time_str = datetime.strftime(start_time, '%m-%d_%H-%M')
    print(start_time_str)

    class_names = ('chipping', 'device_flip', 'empty_pocket', 'foreign_material', 'good', 'lead_defect', 'lead_glue', 'marking_defect')
    num_classes = len(class_names)

    #=============== load the testing data ================#

    batch3_train_dir = os.path.join('/import/home/xwangfy/projects_xwangfy/data_nexperia/all','train' )
    batch3_valid_dir = os.path.join('/import/home/xwangfy/projects_xwangfy/data_nexperia/all','val')
    batch3_test_dir = os.path.join('/import/home/xwangfy/projects_xwangfy/data_nexperia/all','test')

    test_data1 = get_dataloader(batch_size=args.BATCH_SIZE).testdata(batch3_train_dir)
    test_data2 = get_dataloader(batch_size=args.BATCH_SIZE).testdata(batch3_valid_dir)
    test_data3 = get_dataloader(batch_size=args.BATCH_SIZE).testdata(batch3_test_dir)
    combined_data = torch.utils.data.ConcatDataset([test_data1, test_data2, test_data3])
    test_loader = torch.utils.data.DataLoader(dataset=combined_data, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ====================== Model ==========================#
    checkpoint_file_loc = os.path.join(saved_dir, args.MODEL_saved)
    if args.MODEL_type == 'MLPMixer':
        MODEL = MLPMixer(patch_size=args.MODEL_MLPMixer_patch_size, channel_dim=args.MODEL_MLPMixer_channel_dim, \
                         num_blocks=args.MODEL_MLPMixer_num_blocks, fig_size=args.MODEL_MLPMixer_fig_size)  # 这些参数必须要吗
        num_ftrs = MODEL.out_fc.in_features
        MODEL.out_fc = nn.Linear(num_ftrs, num_classes)
    elif args.MODEL_type == 'resnet34':
        MODEL = models.resnet34(num_classes=num_classes)
    else:
        print("The input of MODEL is wrong!!!")
    checkpoint = torch.load(checkpoint_file_loc)
    state_dict = checkpoint['model_state_dict']
    MODEL.load_state_dict(state_dict)
    MODEL.to(device)

    #===================== loss function ====================#
    criterion = args.loss_func

    # ====================== Test model =====================#
    # loss_rec = []
    # acc_rec = []
    # roc_auc_rec = []
    # pr_auc_rec = []

    loss_test , acc_test, mat_test, y_true_test, y_outputs_test = ModelTester(test_loader, MODEL, criterion, device)

    roc_auc_test, pr_auc_test, fpr_test = cal_auc(y_true_test, y_outputs_test)

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

    f = open(os.path.join(log_dir, 'log_test.txt'), 'a')
    sys.stdout = f
    sys.stderr = f

