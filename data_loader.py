from dataset import get_dataset
from torch.utils.data import DataLoader


import torchvision.datasets as datasets
import torchvision.transforms as transforms

class get_dataloader:
    def __init__(self, batch_size):
        self.norm_mean = [0.2203, 0.2203, 0.2203]
        self.norm_std = [0.1407, 0.1407, 0.1407]
        self.train_transform = transforms.Compose([
        transforms.Resize(255), # 将输入PIL图像的大小调整为给定大小 缩放或者拉伸
        transforms.CenterCrop(224), # 224(resnet34)->32(resnet18)  # 依据给定的size从中心裁剪
        transforms.ColorJitter(0.2, 0.2, 0.2), # 修改亮度、对比度和饱和度
        transforms.RandomAffine(degrees=10, translate=(0.15, 0.1), scale=(0.75, 1.05)), # 图像保持中心不变的随机仿射变换
        transforms.RandomHorizontalFlip(p=0.5), # 依概率p水平翻转
        transforms.ToTensor(), # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1] 注意事项：归一化至[0-1]是直接除以255
        transforms.Normalize(self.norm_mean, self.norm_std),])# 用平均值和标准偏差归一化张量图像
        self.valid_transform = transforms.Compose([  # val的时候对图片不要做data augmentation  但是这里还是稍微处理了一下图片的size
        transforms.Resize(255),
        transforms.CenterCrop(224),  # 224->32
        transforms.ToTensor(),
        transforms.Normalize(self.norm_mean, self.norm_std),])
        self.BATCH_SIZE = batch_size

    def trainloader(self):
        train_dir = get_dataset().trainset()
        train_data = datasets.ImageFolder(train_dir, self.train_transform)
        train_loader = DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
        return train_loader

    def validloader(self):
        valid_dir = get_dataset().trainset()
        valid_data = datasets.ImageFolder(valid_dir, self.valid_transform)  # 这里别写错了
        valid_loader = DataLoader(dataset=valid_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
        return valid_loader

    def testdata(self,test_dir):
        # 注意这里和train/val不一样的！！！
        test_data = datasets.ImageFolder(test_dir, self.valid_transform) # test的时候也是做了和val一样的transform欸
        return test_data


# 原始图片的size不是都是统一的
# class get_dataloader_keepsize:
#     '''
#     write for MLPMixer model since it do not require the input images to keep 224*224
#     Hence, we can keep the original size of the images
#     '''
#     def __init__(self, batch_size):
#         # The original size of the image is (275, 267)
#         self.norm_mean = [0.2203, 0.2203, 0.2203]
#         self.norm_std = [0.1407, 0.1407, 0.1407]
#         self.train_transform = transforms.Compose([
#         transforms.ColorJitter(0.2, 0.2, 0.2), # 修改亮度、对比度和饱和度
#         transforms.RandomAffine(degrees=10, translate=(0.15, 0.1), scale=(0.75, 1.05)), # 图像保持中心不变的随机仿射变换
#         transforms.RandomHorizontalFlip(p=0.5), # 依概率p水平翻转
#         transforms.ToTensor(), # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1] 注意事项：归一化至[0-1]是直接除以255
#         transforms.Normalize(self.norm_mean, self.norm_std),])# 用平均值和标准偏差归一化张量图像
#         self.valid_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(self.norm_mean, self.norm_std),])
#         self.BATCH_SIZE = batch_size
#
#     def trainloader(self):
#         train_dir = get_dataset().trainset()
#         train_data = datasets.ImageFolder(train_dir, self.train_transform)
#         train_loader = DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
#         return train_loader
#
#     def validloader(self):
#         valid_dir = get_dataset().trainset()
#         valid_data = datasets.ImageFolder(valid_dir, self.valid_transform)  # 这里别写错了
#         valid_loader = DataLoader(dataset=valid_data, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=4)
#         return valid_loader
#
#     def testdata(self,test_dir):
#         # 注意这里和train/val不一样的！！！
#         test_data = datasets.ImageFolder(test_dir, self.valid_transform) # test的时候也是做了和val一样的transform欸
#         return test_data




if __name__ == '__main__':
    print(get_dataloader(batch_size=128).trainloader())


