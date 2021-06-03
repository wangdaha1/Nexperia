import os
# import zipfile


# combine the files according to the rules
# pocket_damage -> lead glue  scratch -> marking defect  thin_partical -> foreign material


# Start from batch2
class get_dataset:
    def __init__(self):
        self.data_root = '/import/home/xwangfy/projects_xwangfy/data_nexperia/batch_2'

    def trainset(self):
        train_dir = os.path.join(self.data_root, 'train')
        return train_dir

    def validset(self):
        valid_dir = os.path.join(self.data_root, 'val')
        return valid_dir

    def testset(self):
        test_dir = '/import/home/xwangfy/projects_xwangfy/data_nexperia/all'
        return test_dir



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

