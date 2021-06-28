from dataset import textReadDataset, dataset_info
from torch.utils.data import DataLoader, ConcatDataset
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from misc import SharpenImage, AddPepperNoise


# 牛了
# class ImageFolderWithPaths(datasets.ImageFolder):
#     """Custom dataset that includes image file paths. Extends
#     torchvision.datasets.ImageFolder
#
#     # EXAMPLE USAGE:
#     # instantiate the dataset and dataloader
#     data_dir = "your/data_dir/here"
#     dataset = ImageFolderWithPaths(data_dir) # our custom dataset
#     dataloader = torch.utils.DataLoader(dataset)
#
#     for inputs, labels, paths in dataloader:
#     # use the above variables freely
#         print(inputs, labels, paths)
#     """
#
#     # override the __getitem__ method. this is the method that dataloader calls
#     def __getitem__(self, index):
#         # this is what ImageFolder normally returns
#         original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
#         # the image file path
#         path = self.imgs[index][0]   # 这里得到的path会不会占用的字符太多了。。。
#         # make a new tuple that includes original and the path
#         tuple_with_path = (original_tuple + (path,))
#         return tuple_with_path


############ Jasper added to process the empty images ###################
def collate_fn(batch):
    """
    Jasper added to process the empty images
    referece: https://github.com/pytorch/pytorch/issues/1137
    :param batch:
    :return:
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class get_dataloader:
    def __init__(self, batch_size):
        # self.norm_mean = [0.2203, 0.2203, 0.2203]
        # self.norm_std = [0.1407, 0.1407, 0.1407]
        # self.train_transform = transforms.Compose([
        # transforms.Resize(255), # 将输入PIL图像的大小调整为给定大小 缩放或者拉伸
        # transforms.CenterCrop(224), # 224(resnet34)->32(resnet18)  # 依据给定的size从中心裁剪
        # transforms.ColorJitter(0.2, 0.2, 0.2), # 修改亮度、对比度和饱和度
        # transforms.RandomAffine(degrees=10, translate=(0.15, 0.1), scale=(0.75, 1.05)), # 图像保持中心不变的随机仿射变换
        # transforms.RandomHorizontalFlip(p=0.5), # 依概率p水平翻转
        # transforms.ToTensor(), # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1] 注意事项：归一化至[0-1]是直接除以255
        # transforms.Normalize(self.norm_mean, self.norm_std),])# 用平均值和标准偏差归一化张量图像
        # self.valid_transform = transforms.Compose([  # val的时候对图片不要做data augmentation
        # transforms.Resize(255),
        # transforms.CenterCrop(224),  # 224->32
        # transforms.ToTensor(),
        # transforms.Normalize(self.norm_mean, self.norm_std),])
        self.norm_mean = [0.5]
        self.norm_std = [0.5]
        self.BATCH_SIZE = batch_size
        # self.train_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.RandomCrop(224, padding=4, padding_mode='edge'),
        #     SharpenImage(p=0.5),
        #     AddPepperNoise(0.9, p=0.3),
        #     transforms.RandomChoice([
        #         transforms.RandomAffine(degrees=4, shear=4, translate=(0.1, 0.1), scale=(0.95, 1.05)),
        #         transforms.RandomAffine(degrees=0),
        #     ]),
        #     transforms.RandomHorizontalFlip(p=0.3),
        #     transforms.ColorJitter(brightness=0.7),
        #     transforms.ToTensor(),
        #     transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 2), value=(0)),
        #     transforms.Normalize(self.norm_mean, self.norm_std),
        # ])
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.RandomAffine(degrees=5, translate=(0.15, 0.1), scale=(0.75, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(self.norm_mean, self.norm_std),
        ])
        self.valid_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.norm_mean, self.norm_std),
        ])
        self.data_dir = '/import/home/share/from_Nexperia_April2021/Nex_trainingset'


    def trainloader(self):
        name_train, labels_train = dataset_info(os.path.join(self.data_dir, 'Nex_trainingset_train.txt'))
        train_data = textReadDataset(self.data_dir, name_train, labels_train, self.train_transform)
        # if path == True:
        #     train_data = ImageFolderWithPaths(self.data_dir, self.train_transform)
        # else:
        #     train_data = datasets.ImageFolder(self.data_dir, self.train_transform)
        train_loader = DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        return train_loader

    def validloader(self):
        # 因为后面用selfie训练的时候即使用selfie也是不需要做任何变化的，所以path不要设为true
        name_val, labels_val = dataset_info(os.path.join(self.data_dir, 'Nex_trainingset_val.txt'))
        valid_data = textReadDataset(self.data_dir, name_val, labels_val, self.valid_transform)
        # if path==True:
        #     valid_data = ImageFolderWithPaths(self.data_dir, self.valid_transform)
        # else:
        #     valid_data = datasets.ImageFolder(self.data_dir, self.valid_transform)
        valid_loader = DataLoader(dataset=valid_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
        return valid_loader

    def testloader(self):
        name_test, labels_test = dataset_info(os.path.join(self.data_dir, 'Nex_trainingset_test.txt'))
        test_data = textReadDataset(self.data_dir, name_test, labels_test, self.valid_transform)
        # test_data = datasets.ImageFolder(self.data_dir, self.valid_transform) # test的时候也是做了和val一样的transform欸
        test_loader = DataLoader(dataset=test_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
        return test_loader

    def testloader_Jan(self):
        name_jan_train, labels_jan_train = dataset_info('/import/home/share/from_Nexperia_April2021/Jan2021/Jan2021_train_down.txt')
        name_jan_valid, labels_jan_valid = dataset_info('/import/home/share/from_Nexperia_April2021/Jan2021/Jan2021_val_down.txt')
        name_jan_test, labels_jan_test = dataset_info('/import/home/share/from_Nexperia_April2021/Jan2021/Jan2021_test_down.txt')
        train_jan = textReadDataset('/import/home/share/from_Nexperia_April2021/Jan2021', name_jan_train, labels_jan_train, self.valid_transform)
        valid_jan = textReadDataset('/import/home/share/from_Nexperia_April2021/Jan2021', name_jan_valid, labels_jan_valid, self.valid_transform)
        test_jan = textReadDataset('/import/home/share/from_Nexperia_April2021/Jan2021', name_jan_test, labels_jan_test, self.valid_transform)
        combined_jan = ConcatDataset([train_jan, valid_jan, test_jan])
        jan_loader = DataLoader(dataset = combined_jan, collate_fn= collate_fn, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=8)
        return jan_loader

    def testloader_Feb(self):
        name_feb_train, labels_feb_train = dataset_info('/import/home/share/from_Nexperia_April2021/Feb2021/Feb2021_train_down.txt')
        name_feb_valid, labels_feb_valid = dataset_info('/import/home/share/from_Nexperia_April2021/Feb2021/Feb2021_val_down.txt')
        name_feb_test, labels_feb_test = dataset_info('/import/home/share/from_Nexperia_April2021/Feb2021/Feb2021_test_down.txt')
        train_feb = textReadDataset('/import/home/share/from_Nexperia_April2021/Feb2021', name_feb_train, labels_feb_train, self.valid_transform)
        valid_feb = textReadDataset('/import/home/share/from_Nexperia_April2021/Feb2021', name_feb_valid, labels_feb_valid, self.valid_transform)
        test_feb = textReadDataset('/import/home/share/from_Nexperia_April2021/Feb2021', name_feb_test, labels_feb_test, self.valid_transform)
        combined_feb = ConcatDataset([train_feb, valid_feb, test_feb])
        feb_loader = DataLoader(dataset=combined_feb, collate_fn=collate_fn, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=8)
        return feb_loader

    def testloader_Mar(self):
        name_mar_train, labels_mar_train = dataset_info('/import/home/share/from_Nexperia_April2021/Mar2021/Mar2021_train_down.txt')
        name_mar_valid, labels_mar_valid = dataset_info('/import/home/share/from_Nexperia_April2021/Mar2021/Mar2021_val_down.txt')
        name_mar_test, labels_mar_test = dataset_info('/import/home/share/from_Nexperia_April2021/Mar2021/Mar2021_test_down.txt')
        train_mar = textReadDataset('/import/home/share/from_Nexperia_April2021/Mar2021', name_mar_train,labels_mar_train, self.valid_transform)
        valid_mar = textReadDataset('/import/home/share/from_Nexperia_April2021/Mar2021', name_mar_valid,labels_mar_valid, self.valid_transform)
        test_mar = textReadDataset('/import/home/share/from_Nexperia_April2021/Mar2021', name_mar_test, labels_mar_test,self.valid_transform)
        combined_mar = ConcatDataset([train_mar, valid_mar, test_mar])
        mar_loader = DataLoader(dataset=combined_mar, collate_fn=collate_fn, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=8)
        return mar_loader

    # def get_all_paths(self):

        # 将所有训练数据的paths集合起来 后面要用的
        # train_dir = get_dataset_dir().trainset()
        # train_data = ImageFolderWithPaths(train_dir, self.train_transform)  # 居然是一定要一个transform的
        # paths_set = []
        # train_loader = DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
        # for i, data in enumerate(train_loader):
        #     inputs, labels, paths = data
        #     paths_list = list(paths)
        #     paths_set += paths_list
        # return paths_set



if __name__ == '__main__':
    loader = get_dataloader(batch_size=128).testloader_Feb()
    loader = get_dataloader(batch_size=128).testloader_Mar()
    print("Yes OK")
