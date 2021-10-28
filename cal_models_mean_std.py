# calculate 3 models' performance of mean and std

import numpy as np
import re

def cal_models_mean_std():
    with open(r"./results/09-27_patchup+apex/test/log_test.txt") as f:
        lines = f.readlines()
    results = [s for s in lines if ("test Acc:" in s)] # 把记录results的值筛选出来
    fpr_98 = [float(re.findall(r"test fpr_98:(.+?)%", str)[0]) for str in results]
    auc = [float(re.findall(r"test AUC:(.+?)%", str)[0]) for str in results]
    # fpr_98 = [float(s[45:50]) for s in results] # 不是很robust 还得改改
    # auc = [float(s[61:66]) for s in results]

    print("Jan fpr")
    fpr_98_best_Jan = fpr_98[0:3]; print("%.2f(%.2f)" %(np.mean(fpr_98_best_Jan), np.std(fpr_98_best_Jan)))
    fpr_98_last_Jan = fpr_98[3:6]; print("%.2f(%.2f)" %(np.mean(fpr_98_last_Jan), np.std(fpr_98_last_Jan)))
    print("%.2f(%.2f)" %(np.mean(fpr_98_best_Jan+fpr_98_last_Jan), np.std(fpr_98_best_Jan+fpr_98_last_Jan)))

    print("Feb fpr")
    fpr_98_best_Feb = fpr_98[6:9]; print("%.2f(%.2f)" %(np.mean(fpr_98_best_Feb), np.std(fpr_98_best_Feb)))
    fpr_98_last_Feb = fpr_98[9:12]; print("%.2f(%.2f)" %(np.mean(fpr_98_last_Feb), np.std(fpr_98_last_Feb)))
    print("%.2f(%.2f)" %(np.mean(fpr_98_best_Feb + fpr_98_last_Feb), np.std(fpr_98_best_Feb + fpr_98_last_Feb)))

    print("Mar fpr")
    fpr_98_best_Mar = fpr_98[12:15]; print("%.2f(%.2f)" %(np.mean(fpr_98_best_Mar), np.std(fpr_98_best_Mar)))
    fpr_98_last_Mar = fpr_98[15:18]; print("%.2f(%.2f)" %(np.mean(fpr_98_last_Mar), np.std(fpr_98_last_Mar)))
    print("%.2f(%.2f)" %(np.mean(fpr_98_best_Mar + fpr_98_last_Mar), np.std(fpr_98_best_Mar + fpr_98_last_Mar)))

    print("Jan auc")
    auc_best_Jan = auc[0:3]; print("%.2f(%.2f)" %(np.mean(auc_best_Jan), np.std(auc_best_Jan)))
    auc_last_Jan = auc[3:6]; print("%.2f(%.2f)" %(np.mean(auc_last_Jan), np.std(auc_last_Jan)))
    print("%.2f(%.2f)" %(np.mean(auc_best_Jan + auc_last_Jan), np.std(auc_best_Jan + auc_last_Jan)))

    print("Feb auc")
    auc_best_Feb = auc[6:9]; print("%.2f(%.2f)" %(np.mean(auc_best_Feb), np.std(auc_best_Feb)))
    auc_last_Feb = auc[9:12]; print("%.2f(%.2f)" %(np.mean(auc_last_Feb), np.std(auc_last_Feb)))
    print("%.2f(%.2f)" %(np.mean(auc_best_Feb + auc_last_Feb), np.std(auc_best_Feb + auc_last_Feb)))

    print("Mar auc")
    auc_best_Mar = auc[12:15]; print("%.2f(%.2f)" %(np.mean(auc_best_Mar), np.std(auc_best_Mar)))
    auc_last_Mar = auc[15:18]; print("%.2f(%.2f)" %(np.mean(auc_last_Mar), np.std(auc_last_Mar)))
    print("%.2f(%.2f)" %(np.mean(auc_best_Mar + auc_last_Mar), np.std(auc_best_Mar + auc_last_Mar)))

    return


if __name__ == '__main__':
    cal_models_mean_std()
