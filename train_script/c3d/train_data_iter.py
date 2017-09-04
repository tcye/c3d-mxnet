import os
import glob
import random
import cv2
import numpy as np
import mxnet as mx


batch_size = 16


class UCFIter(mx.io.DataIter):
    def __init__(self, ucf_dir, lst_file, batch_size, data_shape):
        super(UCFIter, self).__init__(batch_size)
        self.ucf_dir = ucf_dir
        self.lst_file = lst_file
        self.data_shape = data_shape

        self.name2label = self.read_label_map()
        self.files, self.labels = self.read_lst_data()
        self.count = len(self.files) // batch_size

        self.provide_data = [('data', (batch_size, ) + data_shape)]
        self.provide_label = [('label', (batch_size, ))]


    def read_label_map(self):
        name2label = dict()
        dirname = os.path.dirname(self.lst_file)
        label_map_file = os.path.join(dirname, 'classInd.txt')
        with open(label_map_file, 'r') as f:
            total = f.readlines()
        for line in total:
            tmp, _ = line.split('\n')
            label, name = tmp.split(' ', 1)
            name2label[name] = int(label)
        return name2label

    def decide_label_by_name(self, filename):
        name = filename.split('/')[0]
        return self.name2label.get(name, 0)

    def read_lst_data(self):
        data_1 = []
        data_2 = []
        with open(self.lst_file, 'r') as f:
            total = f.readlines()
        random.shuffle(total)
        for eachLine in total:
            tmp = eachLine.split('\n')
            tmp = tmp[0].split(' ', 1)

            if len(tmp) == 2:
                tmp_1, tmp_2 = tmp
            else:  # len(tmp) == 1
                tmp_1 = tmp[0]
                tmp_2 = self.decide_label_by_name(tmp_1)

            data_1.append(tmp_1)
            data_2.append(int(tmp_2))
        return data_1, data_2

    def read_video_frames(self, video_file):
        filename, _ = os.path.splitext(video_file)
        imgdir = os.path.join(self.ucf_dir, filename)
        imgnames = os.path.join(imgdir, '*.jpg')
        pics = sorted(glob.glob(imgnames))

        r_1 = []
        g_1 = []
        b_1 = []
        mat = []

        for pic in pics:
            img = cv2.imread(pic, cv2.IMREAD_COLOR)
            b, g, r = cv2.split(img)
            r = cv2.resize(r, (self.data_shape[3], self.data_shape[2]))
            g = cv2.resize(g, (self.data_shape[3], self.data_shape[2]))
            b = cv2.resize(b, (self.data_shape[3], self.data_shape[2]))
            r = np.multiply(r, 1/255.0) - 0.5
            g = np.multiply(g, 1/255.0) - 0.5
            b = np.multiply(b, 1/255.0) - 0.5
            r_1.append(r)
            g_1.append(g)
            b_1.append(b)

        mat.append(r_1)
        mat.append(g_1)
        mat.append(b_1)
        return mat

    def __iter__(self):
        for k in range(self.count):
            data = []
            label = []
            for i in range(self.batch_size):
                idx = k * self.batch_size + i
                pic = self.read_video_frames(self.files[idx])
                data.append(pic)
                label.append(int(self.labels[idx]))

            data_all = [mx.nd.array(data)]
            label_all = [mx.nd.array(label)]
            data_batch = mx.io.DataBatch(data_all, label_all,
                                         provide_data=self.provide_data, provide_label=self.provide_label)
            yield data_batch


def get_test_data_iter():
    return UCFIter(
        '/home/yetiancai/data/UCF-101-IMG2/',
        '/home/yetiancai/data/UCF-101-LST/trainlist01.txt',
        batch_size,
        (3, 16, 122, 122),
    )


def get_train_data_iter():
    return UCFIter(
        '/home/yetiancai/data/UCF-101-IMG2/',
        '/home/yetiancai/data/UCF-101-LST/testlist01.txt',
        batch_size,
        (3, 16, 122, 122),
    )


def get_data_iter():
    return get_train_data_iter(), get_test_data_iter()
