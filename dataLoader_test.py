import os, torch, numpy, cv2, random, glob
import numpy as np



def load_img(path, img_name):
    img_path = os.path.join(path, img_name)
    img_ls = sorted(glob.glob(os.path.join(img_path, '*.bmp')))
    img_3d = []

    for i, img in enumerate(img_ls):
        img_3d.append(cv2.imread(img, -1))

    img_3d = np.array(img_3d)
    return img_3d


def dynamic_batch(visualFeatures, labels):
    dynamic_bs = random.choice([1, 2, 3, 4, 5])
    cut = 400 // dynamic_bs
    visual_batch, labels_batch = [], []

    for i in range(dynamic_bs):
        visual_batch.append(visualFeatures[int(i * cut): int((i + 1) * cut), ...])
        labels_batch.append(labels[int(i * cut): int((i + 1) * cut), ...])
        print(visual_batch[i].shape, labels_batch[i].shape)

    return np.array(visual_batch), np.array(labels_batch)

class train_loader_3d(object):
    def __init__(self, trialFileName, data_path, mode):
        self.path = data_path
        self.mode = mode
        self.mixLst = open(trialFileName).read().splitlines()[1:]

    def __getitem__(self, index):
        batchList = self.mixLst[index]
        # visualFeatures, labels = [], []
        img_name = batchList.split(',')[0]
        label = batchList.split(',')[-1]

        img = load_img(self.path, img_name)
        labels = np.array([label for i in range(img.shape[-1])])
        if self.mode == 'train':
            img, labels = dynamic_batch(img, labels)

        visualFeatures = torch.FloatTensor(numpy.array(img)).unsqueeze(0)
        labels = torch.LongTensor(numpy.array(labels)).unsqueeze(0)

        return  visualFeatures, labels,

    def __len__(self):
        return len(self.mixLst)

class train_loader_2d():
    def __init__(self, trialFileName, data_path, mode):
        self.path = data_path
        self.mode = mode
        self.mixLst = open(trialFileName).read().splitlines()[1:]

    def __getitem__(self, index):
        batchList = self.mixLst[index]
        # visualFeatures, labels = [], []
        img_name = batchList.split(',')[0]

        img = cv2.imread(os.path.join(self.path, img_name), -1)
        labels = batchList.split(',')[-1]

        visualFeatures = torch.FloatTensor(numpy.array(img))
        labels = torch.LongTensor(numpy.array(labels))

        return  visualFeatures, labels,

    def __len__(self):
        return len(self.mixLst)

