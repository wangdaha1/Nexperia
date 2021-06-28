# SELFIE only focus on the optimizing process
# we only need to replace the ModelTrainer function with SELFIE function

import numpy as np
import operator
import torch
from data_loader import get_dataloader
import torch.nn.functional as F

class MiniBatch(object):
    # 这里处理后的格式要和原来的inputs的格式是一样的才行啊啊啊啊  看看怎么改
    # 好像不需要改 直接在用的时候将对应的images和labels改成tensor格式的就可以
    def __init__(self):
        self.paths = []
        # images和labels里面append tensor元素进来也是可以的 到时候用的时候把这两个元素取出来合成tensor再放入model就可以
        self.images = []
        self.labels = []

    def append(self, id, image, label):
        self.paths.append(id)
        self.images.append(image)
        self.labels.append(label)

    def get_size(self):
        return len(self.paths)

class Correcter(object):
    def __init__(self, num_classes=8, history_length=15, threshold=0.05, noise_rate = 0.01,\
                 paths = get_dataloader(batch_size=128).get_all_paths(), record_noisy_samples = False):
        self.paths = paths
        self.size_of_data = len(paths)
        self.num_classes = num_classes
        self.history_length = history_length   # 参数q
        self.threshold = threshold  # 参数epsilon
        self.noise_rate = noise_rate
        self.record_noisy_samples = record_noisy_samples  # 是否要记录noisy samples到NOISY_SET里面去，一般到了很后面才记录吧
        self.NOISY_SET = {'Corresponding value':['original label', 'refurbished label']}
        # 保存训练过程中被更新过的samples的paths original labels和refurbished labels  refurbished label要和原来的label比较一下的
        print("history_lengh: {a}, threshold: {b}, noise_rate: {c}".format(a = self.history_length, b = self.threshold, c = self.noise_rate))

        # prediction histories of samples
        self.all_predictions = {}  # dict用的好啊
        for i in range(self.size_of_data):
            self.all_predictions[self.paths[i]] = np.zeros(history_length, dtype=int) # 原始数据的label是01234567 但是也没事吧

        # Max predictive uncertainty  计算delta
        self.max_certainty = -np.log(1.0/float(self.num_classes))

        # Corrected label map 记录改变后数据的label
        self.corrected_labels = {}
        for i in range(self.size_of_data):
            self.corrected_labels[self.paths[i]] = -1

        self.update_counters = {}
        for i in range(self.size_of_data):
            self.update_counters[self.paths[i]] = 0


        # For Logging  这个到底是干啥的？是具体的数据的输入吗
        # self.loaded_data = None
        # if loaded_data is not None:
        #     self.loaded_data = loaded_data

    # update all_predictions and update_counters 这个函数也是我们要用的
    def async_update_prediction_matrix(self, ids, softmax_matrix):
        # 这里的ids是Minibatch的数据的paths
        for i in range(len(ids)):
            id = ids[i]
            predicted_label = np.argmax(softmax_matrix[i])
            # append the predicted label to the prediction matrix
            cur_index = self.update_counters[id] % self.history_length
            self.all_predictions[id][cur_index] = predicted_label  # 更新位置是id的那个sample的cur_index次的predicted_label
            self.update_counters[id] += 1  # 更新了多少次

    # 集合C的选取 根据lost_array和noise_rate来选取进入C的samples
    def separate_clean_and_unclean_samples(self, paths, images, labels, loss_list):
        clean_batch = MiniBatch()  # 这里只是创建ids, images, labels三个空的list， work for each batch
        unclean_batch = MiniBatch()
        num_clean_instances = int(np.ceil(float(len(paths)) * (1.0 - self.noise_rate)))

        loss_map = {}
        image_map = {}
        label_map = {}

        for i in range(len(paths)):
            loss_map[paths[i]] = loss_list[i]
            image_map[paths[i]] = images[i]
            label_map[paths[i]] = labels[i]

        # sort loss by descending order
        loss_map = dict(sorted(loss_map.items(), key=operator.itemgetter(1), reverse=False))

        index = 0
        for key in loss_map.keys():
            if index < num_clean_instances:
                clean_batch.append(key, image_map[key], label_map[key])
            else:
                unclean_batch.append(key, image_map[key], label_map[key])
            index += 1

        return clean_batch, unclean_batch

    # 集合R的选取 根据predictive uncertainty来选取进入R的samples
    def get_refurbishable_samples(self, paths, images, labels):
        corrected_batch = MiniBatch()

        # check predictive uncertainty
        accumulator = {}
        for i in range(len(paths)):
            id = paths[i]
            image = images[i]
            label = labels[i]

            predictions = self.all_predictions[id]     # 最近的q个历史预测数据
            accumulator.clear()

            for prediction in predictions:
                if prediction not in accumulator:
                    accumulator[prediction] = 1
                else:
                    accumulator[prediction] = accumulator[prediction] + 1

            p_dict = np.zeros(self.num_classes, dtype=float)
            for key, value in accumulator.items():
                p_dict[key] = float(value) / float(self.history_length)  # 计算P(key|x, q)

            # compute predictive uncertainty
            negative_entropy = 0.0
            for i in range(len(p_dict)):
                if p_dict[i] == 0:
                    negative_entropy += 0.0
                else:
                    negative_entropy += p_dict[i] * np.log(p_dict[i])
            certainty = - negative_entropy / self.max_certainty          # 计算F(x; q)

            ############### correspond to the lines 12--19 of the paper ################
            # check refurbishable condition
            if certainty <= self.threshold:
                confident_label = np.argmax(p_dict)
                self.corrected_labels[id] = confident_label
                corrected_batch.append(id, image, confident_label)
                if confident_label != label and self.record_noisy_samples==True: # 如果和原来的label不一样 并且记录机关开启 加入NOISY_SET集合
                    self.NOISY_SET[id] = [label.item(), confident_label]

                # For logging ###########################################################
                # if self.loaded_data is not None:
                #     self.loaded_data[id].corrected = True
                #     self.loaded_data[id].last_corrected_label = self.corrected_labels[id]
                #########################################################################

            # reuse previously classified refurbishable samples
            # As we tested, this part degraded the performance marginally around 0.3%p
            # because uncertainty of the sample may affect the performance
            # elif self.corrected_labels[id] != -1:
            #     corrected_batch.append(id, image, self.corrected_labels[id])
        # 这里返回的corrected_batch是所有的满足entropy<threshold的sample
        return corrected_batch

    # 将C和R合起来
    def merge_clean_and_corrected_samples(self, clean_batch, corrected_batch):

        final_batch = MiniBatch()
        corrected_batch_paths = set()

        # add corrected batch
        for i in range(len(corrected_batch.paths)):
            corrected_batch_paths.add(corrected_batch.paths[i])
            final_batch.append(corrected_batch.paths[i], corrected_batch.images[i], corrected_batch.labels[i])

        # merge clean with refurbishable samples
        # If a sample is included in clean_batch and refurbishable_batch at the same time, then the samples is treated as refurbishable
        for i in range(len(clean_batch.paths)):
            if clean_batch.paths[i] in corrected_batch_paths:
                continue

            if self.corrected_labels[clean_batch.paths[i]] != -1:
                # if the sample was corrected at previous epoch, we reuse the corrected label for current mini-batch
                # WHY??????
                final_batch.append(clean_batch.paths[i], clean_batch.images[i], self.corrected_labels[clean_batch.paths[i]])
            else:
                final_batch.append(clean_batch.paths[i], clean_batch.images[i], clean_batch.labels[i])

        # return final_batch.ids, final_batch.images, final_batch.labels
        return final_batch

    # 将前面的函数综合起来  这个函数就是我们要用的啦
    def patch_clean_with_refurbishable_sample_batch(self, paths, images, labels, loss_list):
        # 1. separate clean and unclean samples
        clean_batch, unclean_batch = self.separate_clean_and_unclean_samples(paths, images, labels, loss_list)
        # 2. get refurbishable samples
        corrected_batch = self.get_refurbishable_samples(paths, images, labels)
        # 3. merging
        return self.merge_clean_and_corrected_samples(clean_batch, corrected_batch)

    # 下面两个函数应该是reuse算法的时候采用的 我们先暂时不要用这个
    def predictions_clear(self):
        self.all_predictions.clear()
        for i in range(self.size_of_data):
            self.all_predictions[i] = np.zeros(self.history_length, dtype=int)

    def compute_new_noise_ratio(self):
        num_corrected_sample = 0
        for key, value in self.corrected_labels.items():
            if value != -1:
                num_corrected_sample += 1

        return 1.0 - float(num_corrected_sample) / float(self.size_of_data)

class SELFIE_ModelTrainer:
    # 要在这个函数外面定义好Correcter
    def __init__(self, corrector, state):
        # parameters
        self.uncertainty_threshold = corrector.threshold
        self.history_length = corrector.history_length
        self.noise_rate = corrector.noise_rate
        # recorder
        self.corrector = corrector
        # warm up or selfie
        self.state = state
        assert state in ['warm_up','SELFIE']

    def train(self, data_loader, model, loss_f, optimizer, epoch_id, gpu, max_epoch, loss_C_selection):
        '''
        这个是训练一个epoch的函数
        用SELFIE的方法来训练模型 主要是在每个batch里面进行操作
        :param data_loader:
        :param model:
        :param loss_f:
        :param optimizer:
        :param epoch_id:
        :param gpu:
        :param max_epoch:
        :return:
        '''
        model.train()
        conf_mat = np.zeros((8, 8))  # confusion matrix 每次epoch都清空
        loss_sigma = []  # 损失函数  记录每一次epoch每一个batch的training loss
        label_append = []
        outputs_append = []

        for i, data in enumerate(data_loader):
            # 每个minibatch
            inputs, labels, paths = data
            inputs, labels = inputs.cuda(gpu), labels.cuda(gpu)
            paths_list = list(paths) # tuple 转为 list

            if self.state == 'warm_up':
                # 就是正常的训练 但是要把prediction记录下来
                outputs = model(inputs) # 出来的结果是tensor 是一个估计矩阵
                optimizer.zero_grad()
                loss = loss_f(outputs, labels)
                loss.backward()
                optimizer.step()

                outputs_list = outputs.tolist()  # 转成list
                self.corrector.async_update_prediction_matrix(paths_list, outputs_list)
                # outputs是一个估计矩阵 还没有通过softmax的 但是我们只要通过一个argmax就可以了 所以也没事

            elif self.state == 'SELFIE':
                # prepare for Clean sample selection
                # 先把所有的数据通过model看一下 但是不更新params
                outputs_C = model(inputs)
                loss_C = loss_C_selection(outputs_C, labels)  # 这里是要算出来每个sample的loss
                loss_C_list  = loss_C.tolist()

                # prepare for Refurbishable sample selection
                MiniBatch_SELFIE = self.corrector.patch_clean_with_refurbishable_sample_batch(paths=paths_list, images = inputs, labels = labels, loss_list=loss_C_list)
                # paths, images, labels, loss_array 得到的是一个定义的MiniBatch的形式 要改成tensor的形式再放入model训练
                inputs = torch.tensor([item.cpu().detach().numpy() for item in MiniBatch_SELFIE.images]).cuda(gpu)
                # print(inputs)
                labels = torch.tensor(MiniBatch_SELFIE.labels).cuda(gpu)
                # print(labels)
                # print(len(labels))
                paths_list = MiniBatch_SELFIE.paths

                # 将最终得到的数据进入model
                outputs = model(inputs)
                loss = loss_f(outputs, labels)
                # print(loss)
                loss.backward()
                optimizer.step()

                outputs_list = outputs.tolist()  # 转成list
                self.corrector.async_update_prediction_matrix(paths_list, outputs_list)

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
                print("SELFIE Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(\
                    epoch_id + 1, max_epoch, i + 1, len(data_loader), np.mean(loss_sigma), acc_avg))

        return np.mean(loss_sigma), acc_avg, conf_mat, label_append, outputs_append


    def valid(self, data_loader, model, loss_f, gpu):
        # 这里不需要做任何变化的
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

if __name__ == '__main__':
    class A(object):
        def __init__(self, a):
            self.a = a

        def forward(self):
            self.a += 1
            print(self.a)

    example = A(a=1)
    for i in range(5):
        example.forward()
        if i == 2:
            example.a = 10

