import argparse
import sys
import importlib
import numpy as np
import cv2
import copy

parser = argparse.ArgumentParser(description='show data iter')
parser.add_argument('network_dir', type=str, help='the network dir')
parser.add_argument('--test', action='store_true', help='get test data iter')

args = parser.parse_args()
# load modules
sys.path.insert(0, args.network_dir)
train_symbol = importlib.import_module('train_symbol')
train_data_iter = importlib.import_module('train_data_iter')

data_iter = train_data_iter.get_test_data_iter(train_symbol.data_names, train_symbol.label_names) \
    if args.test else train_data_iter.get_train_data_iter(train_symbol.data_names, train_symbol.label_names)
data_iter.reset()
input_params = train_data_iter.get_common_input_params()

is_visual = True
while data_iter.iter_next():
    ims_chw = data_iter.getdata().asnumpy()
    labels_data = data_iter.getlabel().asnumpy().reshape((-1, 1))
    # swap (c, h, w) <--> (h, w, c)
    ret_shape = copy.deepcopy(list(ims_chw.shape))
    c = ret_shape[-3]
    ret_shape[-3] = ret_shape[-1]
    ret_shape[-1] = c

    ims_hwc = ims_chw.copy().reshape(ret_shape)
    ims_hwc[:, :, :, 2] = ims_chw[:, 0, :, :] / input_params['scale'] + input_params['mean_r']
    ims_hwc[:, :, :, 1] = ims_chw[:, 1, :, :] / input_params['scale'] + input_params['mean_g']
    ims_hwc[:, :, :, 0] = ims_chw[:, 2, :, :] / input_params['scale'] + input_params['mean_b']
    ims_hwc = ims_hwc.astype(np.uint8)
    for i in range(ims_hwc.shape[0]):
        cv2.imshow('im', ims_hwc[i])
        print(labels_data[0])
        is_visual = chr(cv2.waitKey() % 256) not in ['q', 'Q']
        if not is_visual:
            break
    if not is_visual:
        break
