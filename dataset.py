import os
# import zipfile
import torch
from torch.utils.data import Dataset
from PIL import Image

# combine the files according to the rules
# pocket_damage -> lead glue  scratch -> marking defect  thin_partical -> foreign material


# Start from batch2
# class get_dataset_dir:
#     def __init__(self):
#         self.data_root = '/import/home/xwangfy/projects_xwangfy/data_nexperia/batch_2'
#
#     def trainset(self):
#         train_dir = os.path.join(self.data_root, 'train')
#         return train_dir
#
#     def validset(self):
#         valid_dir = os.path.join(self.data_root, 'val')
#         return valid_dir
#
#     def testset(self):
#         test_dir = os.path.join(self.data_root, 'test')
#         return test_dir

def dataset_info(txt_labels):
    '''
    file_names:List, labels:List
    '''
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        # file_names.append(row[0])
        file_names.append(' '.join(row[:-1]))
        try:
            # labels.append(int(row[1].replace("\n", "")))
            labels.append(int(row[-1].replace("\n", "")))
        except ValueError as err:
            # print(row[0],row[1])
            print(' '.join(row[:-1]), row[-1])
    return file_names, labels


class textReadDataset(Dataset):
    """Face Landmarks dataset"""

    def __init__(self, rootdir, names, labels, img_transformer=None):
        self.rootdir = rootdir
        self.names = names
        self.labels = labels
        # self.N = len(self.names)
        self._image_transformer = img_transformer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = index.tolist()

        img_name = self.rootdir + '/' + self.names[index]

        try:
            image = Image.open(img_name).convert('RGB')
        except:
            print(img_name)
            return None
        return self._image_transformer(image), int(self.labels[index] - 1)

    def get_labels(self):
        return self.labels

# extract from zip files
# if __name__ == '__main__':
    # process the zip file
    # import zipfile
    # def unzip_file(zip_src, dst_dir):
    #     r = zipfile.is_zipfile(zip_src)
    #     if r:
    #         fz = zipfile.ZipFile(zip_src, 'r')
    #         for file in fz.namelist():
    #             fz.extract(file, dst_dir)
    #     else:
    #         print('This is not a zip file!')
    #
    # unzip_file('/import/home/xwangfy/projects_xwangfy/data_nexperia/batch_3.zip',
    #            '/import/home/xwangfy/projects_xwangfy/data_nexperia')
