import mxnet as mx

input_size = 224
batch_size = 64


def get_common_input_params():
    param = dict()
    param['mean_r'] = 123
    param['mean_g'] = 117
    param['mean_b'] = 104
    param['scale'] = 1
    param['data_shape'] = (3, input_size, input_size)
    return param


def get_test_data_iter(data_names, label_names, batch_size=batch_size, num_parts=1, part_index=0):
    input_params = get_common_input_params()
    return mx.io.ImageRecordIter(
        path_imgrec='/data/violation/im_res/val.rec',
        resize_mode='square',
        resize=input_size,
        rand_crop=False,
        rand_mirror=False,
        preprocess_threads=8,
        batch_size=batch_size,
        num_parts=num_parts,
        part_index=part_index,
        data_names=data_names,
        label_names=label_names,
        **input_params)


def get_train_data_iter(data_names, label_names, batch_size=batch_size, num_parts=1, part_index=0):
    input_params = get_common_input_params()
    return mx.io.ImageRecordIter(
        path_imgrec='/data/violation/im_res/train.rec',
        resize_mode='random',
        resize_mode_ratio=(0, 1, 1),  # short_base, long_base, square
        resize=256,
        min_random_scale=0.875,  # 224 / 256
        max_random_scale=1,
        max_rotate_angle=20,
        batch_size=batch_size,
        rand_crop=True,
        rand_mirror=True,
        aug_seq='aug_default,blur',
        blur_ratio=0.5,
        blur_mode='random',
        blur_mode_ratio=(1, 1),  # gaussian, motion
        # gaussian blur ralated
        min_gaussian_blur_kernel_size=3,
        max_gaussian_blur_kernel_size=3,
        min_gaussian_blur_sigma=1.5,
        max_gaussian_blur_sigma=1.5,
        # motion blur related
        min_motion_blur_kernel_size=5,
        max_motion_blur_kernel_size=10,
        max_motion_blur_rotate_angle=5,
        # motion_blur_angle_list = (0, 45, 90, 135),   # without this, will random from [0, 180]
        preprocess_threads=24,
        num_parts=num_parts,
        part_index=part_index,
        shuffle_chunk_size=32,
        data_names=data_names,
        label_names=label_names,
        **input_params)


def get_data_iter(data_names, label_names, kv):
    return get_train_data_iter(data_names, label_names, batch_size, kv.num_workers, kv.rank), \
           get_test_data_iter(data_names, label_names, batch_size, kv.num_workers, kv.rank)
