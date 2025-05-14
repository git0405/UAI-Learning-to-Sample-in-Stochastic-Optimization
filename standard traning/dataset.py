import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from torchvision import datasets, models, transforms
import os


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)

    
    
class SiameseNetworkDataset(Dataset):

    def __init__(self,data, label  ):
        # self.tdset = TensorDataset(pair1,pair2, label)
        self.pair1 = data[0]
        self.pair2 = data[1]
        self.label = label

        self.len_data = len(self.label)

        h, w = 224, 224
        
        transform_train_list = [transforms.Resize((h, w), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((h, w))]
        
        self.transform = transforms.Compose(transform_train_list)

    def __getitem__(self, index):
        
        img0 = self.pair1[index]
        img1 = self.pair2[index]
        label = self.label[index]
        # print("img0 shape:",img0.shape)
        # img0 = torch.tensor(img0)
        # img0 = self.transform(img0)
        # print("transformed img0 shape:",img0.shape)
        # img1 = self.transform(img1)
        # img1 = torch.tensor(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1,label

    def __len__(self):
        return self.len_data

    def get_len(self):
        return self.__len__()


def reid_data_prepare(data_list_path, train_dir_path):

    class_img_labels = dict()
    data_list= []
    target = []

    class_cnt = -1
    last_label = -2

    h, w = 224, 224
    
    transform_train_list = [transforms.Resize((h, w), interpolation=3),transforms.ToTensor()]
        
    transform = transforms.Compose(transform_train_list)
    with open(data_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            img = line
            if "CUHK01" in data_list_path:
                lbl = int(line[:4])
            else:
                lbl = int(line.split('_')[0])
            if lbl != last_label:
                class_cnt = class_cnt + 1
                # cur_list = list()
                # class_img_labels[str(class_cnt)] = cur_list
            last_label = lbl
            img = Image.open(os.path.join(train_dir_path, img))

            img = transform(img)
            data_list.append(img)
            target.append(class_cnt)
            # class_img_labels[str(class_cnt)].append(img)

    return data_list,target


class SiameseCUHK03(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, train = True):

        train_list ='./CUHK03/train.txt'
        train_dir = './CUHK03/bounding_box_train/bounding_box_train'
        test_list = './CUHK03/test.txt'
        test_dir = './CUHK03/bounding_box_test/bounding_box_test'

        h, w = 224, 224
        # self.mnist_dataset = mnist_dataset

        transform_train_list = [
            transforms.Resize((h, w), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((h, w)),

            transforms.ToTensor()]

        transform = transforms.Compose(transform_train_list)
        self.transform = transform

        self.train =  train


        if self.train:
            data_list,target = reid_data_prepare(train_list,train_dir)

            self.train_labels = target # np.array(target).reshape(-1)
            self.train_data = data_list
            self.labels_set = set(self.train_labels )
            self.label_to_indices = {label: np.where(np.array(self.train_labels) == label)[0]
                                     for label in self.labels_set}
        else:
            data_list,target = reid_data_prepare(test_list, test_dir)
            # print(len(target))
            # generate fixed pairs for testing
            self.test_labels = target # np.array(target).reshape(-1)
            self.test_data = data_list
            self.labels_set = set(self.test_labels )
            self.label_to_indices = {label: np.where( np.array(self.test_labels)  == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i, random_state.choice(list(set(self.label_to_indices[self.test_labels[i]])-set([i]))), 1]
                              for i in range(len(self.test_data))]

            negative_pairs = [
                [i, random_state.choice(self.label_to_indices[np.random.choice(list(self.labels_set- set([self.test_labels[i] ])))]), 0] for i in
                range(len(self.test_data))]


            np.savez("pair_list", positive_pairs = positive_pairs, negative_pairs =negative_pairs  )
            
            '''positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i]]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i] ]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]'''
            
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        # img1 = Image.fromarray(np.array(img1) )
        # img2 = Image.fromarray(np.array(img2))
        # if self.transform is not None:
        #     img1 = self.transform(img1)
        #     img2 = self.transform(img2)
        return (img1, img2), target
        
    def __len__(self):
        if self.train:
        
            return len(self.train_labels)
        else: 
            return len(self.test_pairs)

class SiameseNetworkDataset_ben(Dataset):

    def __init__(self,data, label  ):
        # self.tdset = TensorDataset(pair1,pair2, label)
        self.pair1 = data[0]
        self.pair2 = data[1]
        self.label = label

        self.len_data = len(self.label)

    def __getitem__(self, index):
        img0 = self.pair1[index]
        img1 = self.pair2[index]
        label = self.label[index]

        
        # print("img0 shape:",img0.shape)
        # img0 = torch.tensor(img0)
        # img0 = self.transform(img0)
        # print("transformed img0 shape:",img0.shape)
        # img1 = self.transform(img1)
        # img1 = torch.tensor(img1)

        return img0, img1,label,index

    def __len__(self):
        return self.len_data

    def get_len(self):
        return self.__len__()