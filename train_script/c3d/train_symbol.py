import mxnet as mx

data_names = ['data']
label_names = ['prob_label']
datas = [mx.symbol.Variable(name=name) for name in data_names]
labels = [mx.symbol.Variable(name=name) for name in label_names]


def get_symbol(is_train=True):
    data = datas[0]
    # if not is_train:
    #     data = mx.symbol.Transformer(name='transormer', data=data, swap_indices=(2, 1, 0),
    #                                  means=(123, 117, 104), scale=1)
    conv_1 = mx.symbol.Convolution(name='conv_1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7),
                                   stride=(2, 2),
                                   no_bias=False)
    bn_1 = mx.symbol.BatchNorm(name='bn_1', data=conv_1, use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_1 = bn_1
    relu_1 = mx.symbol.Activation(name='relu_1', data=scale_1, act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu_1, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                              stride=(2, 2), pool_type='max')
    conv_stage0_block0_proj_shortcut = mx.symbol.Convolution(name='conv_stage0_block0_proj_shortcut', data=pool1,
                                                             num_filter=128, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                             no_bias=False)
    bn_stage0_block0_proj_shortcut = mx.symbol.BatchNorm(name='bn_stage0_block0_proj_shortcut',
                                                         data=conv_stage0_block0_proj_shortcut, use_global_stats=False,
                                                         fix_gamma=False, eps=0.000100)
    scale_stage0_block0_proj_shortcut = bn_stage0_block0_proj_shortcut
    conv_stage0_block0_branch2a = mx.symbol.Convolution(name='conv_stage0_block0_branch2a', data=pool1, num_filter=32,
                                                        pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage0_block0_branch2a = mx.symbol.BatchNorm(name='bn_stage0_block0_branch2a', data=conv_stage0_block0_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage0_block0_branch2a = bn_stage0_block0_branch2a
    relu_stage0_block0_branch2a = mx.symbol.Activation(name='relu_stage0_block0_branch2a',
                                                       data=scale_stage0_block0_branch2a, act_type='relu')
    conv_stage0_block0_branch2b = mx.symbol.Convolution(name='conv_stage0_block0_branch2b',
                                                        data=relu_stage0_block0_branch2a, num_filter=32, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage0_block0_branch2b = mx.symbol.BatchNorm(name='bn_stage0_block0_branch2b', data=conv_stage0_block0_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage0_block0_branch2b = bn_stage0_block0_branch2b
    relu_stage0_block0_branch2b = mx.symbol.Activation(name='relu_stage0_block0_branch2b',
                                                       data=scale_stage0_block0_branch2b, act_type='relu')
    conv_stage0_block0_branch2c = mx.symbol.Convolution(name='conv_stage0_block0_branch2c',
                                                        data=relu_stage0_block0_branch2b, num_filter=128, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage0_block0_branch2c = mx.symbol.BatchNorm(name='bn_stage0_block0_branch2c', data=conv_stage0_block0_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage0_block0_branch2c = bn_stage0_block0_branch2c
    eltwise_stage0_block0 = mx.symbol.broadcast_add(name='eltwise_stage0_block0',
                                                    *[scale_stage0_block0_proj_shortcut, scale_stage0_block0_branch2c])
    relu_stage0_block0 = mx.symbol.Activation(name='relu_stage0_block0', data=eltwise_stage0_block0, act_type='relu')
    conv_stage0_block1_branch2a = mx.symbol.Convolution(name='conv_stage0_block1_branch2a', data=relu_stage0_block0,
                                                        num_filter=32, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                        no_bias=False)
    bn_stage0_block1_branch2a = mx.symbol.BatchNorm(name='bn_stage0_block1_branch2a', data=conv_stage0_block1_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage0_block1_branch2a = bn_stage0_block1_branch2a
    relu_stage0_block1_branch2a = mx.symbol.Activation(name='relu_stage0_block1_branch2a',
                                                       data=scale_stage0_block1_branch2a, act_type='relu')
    conv_stage0_block1_branch2b = mx.symbol.Convolution(name='conv_stage0_block1_branch2b',
                                                        data=relu_stage0_block1_branch2a, num_filter=32, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage0_block1_branch2b = mx.symbol.BatchNorm(name='bn_stage0_block1_branch2b', data=conv_stage0_block1_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage0_block1_branch2b = bn_stage0_block1_branch2b
    relu_stage0_block1_branch2b = mx.symbol.Activation(name='relu_stage0_block1_branch2b',
                                                       data=scale_stage0_block1_branch2b, act_type='relu')
    conv_stage0_block1_branch2c = mx.symbol.Convolution(name='conv_stage0_block1_branch2c',
                                                        data=relu_stage0_block1_branch2b, num_filter=128, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage0_block1_branch2c = mx.symbol.BatchNorm(name='bn_stage0_block1_branch2c', data=conv_stage0_block1_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage0_block1_branch2c = bn_stage0_block1_branch2c
    eltwise_stage0_block1 = mx.symbol.broadcast_add(name='eltwise_stage0_block1',
                                                    *[relu_stage0_block0, scale_stage0_block1_branch2c])
    relu_stage0_block1 = mx.symbol.Activation(name='relu_stage0_block1', data=eltwise_stage0_block1, act_type='relu')
    conv_stage0_block2_branch2a = mx.symbol.Convolution(name='conv_stage0_block2_branch2a', data=relu_stage0_block1,
                                                        num_filter=32, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                        no_bias=False)
    bn_stage0_block2_branch2a = mx.symbol.BatchNorm(name='bn_stage0_block2_branch2a', data=conv_stage0_block2_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage0_block2_branch2a = bn_stage0_block2_branch2a
    relu_stage0_block2_branch2a = mx.symbol.Activation(name='relu_stage0_block2_branch2a',
                                                       data=scale_stage0_block2_branch2a, act_type='relu')
    conv_stage0_block2_branch2b = mx.symbol.Convolution(name='conv_stage0_block2_branch2b',
                                                        data=relu_stage0_block2_branch2a, num_filter=32, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage0_block2_branch2b = mx.symbol.BatchNorm(name='bn_stage0_block2_branch2b', data=conv_stage0_block2_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage0_block2_branch2b = bn_stage0_block2_branch2b
    relu_stage0_block2_branch2b = mx.symbol.Activation(name='relu_stage0_block2_branch2b',
                                                       data=scale_stage0_block2_branch2b, act_type='relu')
    conv_stage0_block2_branch2c = mx.symbol.Convolution(name='conv_stage0_block2_branch2c',
                                                        data=relu_stage0_block2_branch2b, num_filter=128, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage0_block2_branch2c = mx.symbol.BatchNorm(name='bn_stage0_block2_branch2c', data=conv_stage0_block2_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage0_block2_branch2c = bn_stage0_block2_branch2c
    eltwise_stage0_block2 = mx.symbol.broadcast_add(name='eltwise_stage0_block2',
                                                    *[relu_stage0_block1, scale_stage0_block2_branch2c])
    relu_stage0_block2 = mx.symbol.Activation(name='relu_stage0_block2', data=eltwise_stage0_block2, act_type='relu')
    conv_stage1_block0_proj_shortcut = mx.symbol.Convolution(name='conv_stage1_block0_proj_shortcut',
                                                             data=relu_stage0_block2, num_filter=256, pad=(0, 0),
                                                             kernel=(1, 1), stride=(2, 2), no_bias=False)
    bn_stage1_block0_proj_shortcut = mx.symbol.BatchNorm(name='bn_stage1_block0_proj_shortcut',
                                                         data=conv_stage1_block0_proj_shortcut, use_global_stats=False,
                                                         fix_gamma=False, eps=0.000100)
    scale_stage1_block0_proj_shortcut = bn_stage1_block0_proj_shortcut
    conv_stage1_block0_branch2a = mx.symbol.Convolution(name='conv_stage1_block0_branch2a', data=relu_stage0_block2,
                                                        num_filter=64, pad=(0, 0), kernel=(1, 1), stride=(2, 2),
                                                        no_bias=False)
    bn_stage1_block0_branch2a = mx.symbol.BatchNorm(name='bn_stage1_block0_branch2a', data=conv_stage1_block0_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage1_block0_branch2a = bn_stage1_block0_branch2a
    relu_stage1_block0_branch2a = mx.symbol.Activation(name='relu_stage1_block0_branch2a',
                                                       data=scale_stage1_block0_branch2a, act_type='relu')
    conv_stage1_block0_branch2b = mx.symbol.Convolution(name='conv_stage1_block0_branch2b',
                                                        data=relu_stage1_block0_branch2a, num_filter=64, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage1_block0_branch2b = mx.symbol.BatchNorm(name='bn_stage1_block0_branch2b', data=conv_stage1_block0_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage1_block0_branch2b = bn_stage1_block0_branch2b
    relu_stage1_block0_branch2b = mx.symbol.Activation(name='relu_stage1_block0_branch2b',
                                                       data=scale_stage1_block0_branch2b, act_type='relu')
    conv_stage1_block0_branch2c = mx.symbol.Convolution(name='conv_stage1_block0_branch2c',
                                                        data=relu_stage1_block0_branch2b, num_filter=256, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage1_block0_branch2c = mx.symbol.BatchNorm(name='bn_stage1_block0_branch2c', data=conv_stage1_block0_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage1_block0_branch2c = bn_stage1_block0_branch2c
    eltwise_stage1_block0 = mx.symbol.broadcast_add(name='eltwise_stage1_block0',
                                                    *[scale_stage1_block0_proj_shortcut, scale_stage1_block0_branch2c])
    relu_stage1_block0 = mx.symbol.Activation(name='relu_stage1_block0', data=eltwise_stage1_block0, act_type='relu')
    conv_stage1_block1_branch2a = mx.symbol.Convolution(name='conv_stage1_block1_branch2a', data=relu_stage1_block0,
                                                        num_filter=64, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                        no_bias=False)
    bn_stage1_block1_branch2a = mx.symbol.BatchNorm(name='bn_stage1_block1_branch2a', data=conv_stage1_block1_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage1_block1_branch2a = bn_stage1_block1_branch2a
    relu_stage1_block1_branch2a = mx.symbol.Activation(name='relu_stage1_block1_branch2a',
                                                       data=scale_stage1_block1_branch2a, act_type='relu')
    conv_stage1_block1_branch2b = mx.symbol.Convolution(name='conv_stage1_block1_branch2b',
                                                        data=relu_stage1_block1_branch2a, num_filter=64, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage1_block1_branch2b = mx.symbol.BatchNorm(name='bn_stage1_block1_branch2b', data=conv_stage1_block1_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage1_block1_branch2b = bn_stage1_block1_branch2b
    relu_stage1_block1_branch2b = mx.symbol.Activation(name='relu_stage1_block1_branch2b',
                                                       data=scale_stage1_block1_branch2b, act_type='relu')
    conv_stage1_block1_branch2c = mx.symbol.Convolution(name='conv_stage1_block1_branch2c',
                                                        data=relu_stage1_block1_branch2b, num_filter=256, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage1_block1_branch2c = mx.symbol.BatchNorm(name='bn_stage1_block1_branch2c', data=conv_stage1_block1_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage1_block1_branch2c = bn_stage1_block1_branch2c
    eltwise_stage1_block1 = mx.symbol.broadcast_add(name='eltwise_stage1_block1',
                                                    *[relu_stage1_block0, scale_stage1_block1_branch2c])
    relu_stage1_block1 = mx.symbol.Activation(name='relu_stage1_block1', data=eltwise_stage1_block1, act_type='relu')
    conv_stage1_block2_branch2a = mx.symbol.Convolution(name='conv_stage1_block2_branch2a', data=relu_stage1_block1,
                                                        num_filter=64, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                        no_bias=False)
    bn_stage1_block2_branch2a = mx.symbol.BatchNorm(name='bn_stage1_block2_branch2a', data=conv_stage1_block2_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage1_block2_branch2a = bn_stage1_block2_branch2a
    relu_stage1_block2_branch2a = mx.symbol.Activation(name='relu_stage1_block2_branch2a',
                                                       data=scale_stage1_block2_branch2a, act_type='relu')
    conv_stage1_block2_branch2b = mx.symbol.Convolution(name='conv_stage1_block2_branch2b',
                                                        data=relu_stage1_block2_branch2a, num_filter=64, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage1_block2_branch2b = mx.symbol.BatchNorm(name='bn_stage1_block2_branch2b', data=conv_stage1_block2_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage1_block2_branch2b = bn_stage1_block2_branch2b
    relu_stage1_block2_branch2b = mx.symbol.Activation(name='relu_stage1_block2_branch2b',
                                                       data=scale_stage1_block2_branch2b, act_type='relu')
    conv_stage1_block2_branch2c = mx.symbol.Convolution(name='conv_stage1_block2_branch2c',
                                                        data=relu_stage1_block2_branch2b, num_filter=256, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage1_block2_branch2c = mx.symbol.BatchNorm(name='bn_stage1_block2_branch2c', data=conv_stage1_block2_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage1_block2_branch2c = bn_stage1_block2_branch2c
    eltwise_stage1_block2 = mx.symbol.broadcast_add(name='eltwise_stage1_block2',
                                                    *[relu_stage1_block1, scale_stage1_block2_branch2c])
    relu_stage1_block2 = mx.symbol.Activation(name='relu_stage1_block2', data=eltwise_stage1_block2, act_type='relu')
    conv_stage1_block3_branch2a = mx.symbol.Convolution(name='conv_stage1_block3_branch2a', data=relu_stage1_block2,
                                                        num_filter=64, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                        no_bias=False)
    bn_stage1_block3_branch2a = mx.symbol.BatchNorm(name='bn_stage1_block3_branch2a', data=conv_stage1_block3_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage1_block3_branch2a = bn_stage1_block3_branch2a
    relu_stage1_block3_branch2a = mx.symbol.Activation(name='relu_stage1_block3_branch2a',
                                                       data=scale_stage1_block3_branch2a, act_type='relu')
    conv_stage1_block3_branch2b = mx.symbol.Convolution(name='conv_stage1_block3_branch2b',
                                                        data=relu_stage1_block3_branch2a, num_filter=64, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage1_block3_branch2b = mx.symbol.BatchNorm(name='bn_stage1_block3_branch2b', data=conv_stage1_block3_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage1_block3_branch2b = bn_stage1_block3_branch2b
    relu_stage1_block3_branch2b = mx.symbol.Activation(name='relu_stage1_block3_branch2b',
                                                       data=scale_stage1_block3_branch2b, act_type='relu')
    conv_stage1_block3_branch2c = mx.symbol.Convolution(name='conv_stage1_block3_branch2c',
                                                        data=relu_stage1_block3_branch2b, num_filter=256, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage1_block3_branch2c = mx.symbol.BatchNorm(name='bn_stage1_block3_branch2c', data=conv_stage1_block3_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage1_block3_branch2c = bn_stage1_block3_branch2c
    eltwise_stage1_block3 = mx.symbol.broadcast_add(name='eltwise_stage1_block3',
                                                    *[relu_stage1_block2, scale_stage1_block3_branch2c])
    relu_stage1_block3 = mx.symbol.Activation(name='relu_stage1_block3', data=eltwise_stage1_block3, act_type='relu')
    drop_stage1 = mx.symbol.Dropout(name='drop_stage1', data=relu_stage1_block3, p=0.100000)
    conv_stage2_block0_proj_shortcut = mx.symbol.Convolution(name='conv_stage2_block0_proj_shortcut', data=drop_stage1,
                                                             num_filter=512, pad=(0, 0), kernel=(1, 1), stride=(2, 2),
                                                             no_bias=False)
    bn_stage2_block0_proj_shortcut = mx.symbol.BatchNorm(name='bn_stage2_block0_proj_shortcut',
                                                         data=conv_stage2_block0_proj_shortcut, use_global_stats=False,
                                                         fix_gamma=False, eps=0.000100)
    scale_stage2_block0_proj_shortcut = bn_stage2_block0_proj_shortcut
    conv_stage2_block0_branch2a = mx.symbol.Convolution(name='conv_stage2_block0_branch2a', data=drop_stage1,
                                                        num_filter=128, pad=(0, 0), kernel=(1, 1), stride=(2, 2),
                                                        no_bias=False)
    bn_stage2_block0_branch2a = mx.symbol.BatchNorm(name='bn_stage2_block0_branch2a', data=conv_stage2_block0_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block0_branch2a = bn_stage2_block0_branch2a
    relu_stage2_block0_branch2a = mx.symbol.Activation(name='relu_stage2_block0_branch2a',
                                                       data=scale_stage2_block0_branch2a, act_type='relu')
    conv_stage2_block0_branch2b = mx.symbol.Convolution(name='conv_stage2_block0_branch2b',
                                                        data=relu_stage2_block0_branch2a, num_filter=128, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage2_block0_branch2b = mx.symbol.BatchNorm(name='bn_stage2_block0_branch2b', data=conv_stage2_block0_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block0_branch2b = bn_stage2_block0_branch2b
    relu_stage2_block0_branch2b = mx.symbol.Activation(name='relu_stage2_block0_branch2b',
                                                       data=scale_stage2_block0_branch2b, act_type='relu')
    conv_stage2_block0_branch2c = mx.symbol.Convolution(name='conv_stage2_block0_branch2c',
                                                        data=relu_stage2_block0_branch2b, num_filter=512, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage2_block0_branch2c = mx.symbol.BatchNorm(name='bn_stage2_block0_branch2c', data=conv_stage2_block0_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block0_branch2c = bn_stage2_block0_branch2c
    eltwise_stage2_block0 = mx.symbol.broadcast_add(name='eltwise_stage2_block0',
                                                    *[scale_stage2_block0_proj_shortcut, scale_stage2_block0_branch2c])
    relu_stage2_block0 = mx.symbol.Activation(name='relu_stage2_block0', data=eltwise_stage2_block0, act_type='relu')
    conv_stage2_block1_branch2a = mx.symbol.Convolution(name='conv_stage2_block1_branch2a', data=relu_stage2_block0,
                                                        num_filter=128, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                        no_bias=False)
    bn_stage2_block1_branch2a = mx.symbol.BatchNorm(name='bn_stage2_block1_branch2a', data=conv_stage2_block1_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block1_branch2a = bn_stage2_block1_branch2a
    relu_stage2_block1_branch2a = mx.symbol.Activation(name='relu_stage2_block1_branch2a',
                                                       data=scale_stage2_block1_branch2a, act_type='relu')
    conv_stage2_block1_branch2b = mx.symbol.Convolution(name='conv_stage2_block1_branch2b',
                                                        data=relu_stage2_block1_branch2a, num_filter=128, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage2_block1_branch2b = mx.symbol.BatchNorm(name='bn_stage2_block1_branch2b', data=conv_stage2_block1_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block1_branch2b = bn_stage2_block1_branch2b
    relu_stage2_block1_branch2b = mx.symbol.Activation(name='relu_stage2_block1_branch2b',
                                                       data=scale_stage2_block1_branch2b, act_type='relu')
    conv_stage2_block1_branch2c = mx.symbol.Convolution(name='conv_stage2_block1_branch2c',
                                                        data=relu_stage2_block1_branch2b, num_filter=512, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage2_block1_branch2c = mx.symbol.BatchNorm(name='bn_stage2_block1_branch2c', data=conv_stage2_block1_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block1_branch2c = bn_stage2_block1_branch2c
    eltwise_stage2_block1 = mx.symbol.broadcast_add(name='eltwise_stage2_block1',
                                                    *[relu_stage2_block0, scale_stage2_block1_branch2c])
    relu_stage2_block1 = mx.symbol.Activation(name='relu_stage2_block1', data=eltwise_stage2_block1, act_type='relu')
    conv_stage2_block2_branch2a = mx.symbol.Convolution(name='conv_stage2_block2_branch2a', data=relu_stage2_block1,
                                                        num_filter=128, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                        no_bias=False)
    bn_stage2_block2_branch2a = mx.symbol.BatchNorm(name='bn_stage2_block2_branch2a', data=conv_stage2_block2_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block2_branch2a = bn_stage2_block2_branch2a
    relu_stage2_block2_branch2a = mx.symbol.Activation(name='relu_stage2_block2_branch2a',
                                                       data=scale_stage2_block2_branch2a, act_type='relu')
    conv_stage2_block2_branch2b = mx.symbol.Convolution(name='conv_stage2_block2_branch2b',
                                                        data=relu_stage2_block2_branch2a, num_filter=128, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage2_block2_branch2b = mx.symbol.BatchNorm(name='bn_stage2_block2_branch2b', data=conv_stage2_block2_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block2_branch2b = bn_stage2_block2_branch2b
    relu_stage2_block2_branch2b = mx.symbol.Activation(name='relu_stage2_block2_branch2b',
                                                       data=scale_stage2_block2_branch2b, act_type='relu')
    conv_stage2_block2_branch2c = mx.symbol.Convolution(name='conv_stage2_block2_branch2c',
                                                        data=relu_stage2_block2_branch2b, num_filter=512, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage2_block2_branch2c = mx.symbol.BatchNorm(name='bn_stage2_block2_branch2c', data=conv_stage2_block2_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block2_branch2c = bn_stage2_block2_branch2c
    eltwise_stage2_block2 = mx.symbol.broadcast_add(name='eltwise_stage2_block2',
                                                    *[relu_stage2_block1, scale_stage2_block2_branch2c])
    relu_stage2_block2 = mx.symbol.Activation(name='relu_stage2_block2', data=eltwise_stage2_block2, act_type='relu')
    conv_stage2_block3_branch2a = mx.symbol.Convolution(name='conv_stage2_block3_branch2a', data=relu_stage2_block2,
                                                        num_filter=128, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                        no_bias=False)
    bn_stage2_block3_branch2a = mx.symbol.BatchNorm(name='bn_stage2_block3_branch2a', data=conv_stage2_block3_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block3_branch2a = bn_stage2_block3_branch2a
    relu_stage2_block3_branch2a = mx.symbol.Activation(name='relu_stage2_block3_branch2a',
                                                       data=scale_stage2_block3_branch2a, act_type='relu')
    conv_stage2_block3_branch2b = mx.symbol.Convolution(name='conv_stage2_block3_branch2b',
                                                        data=relu_stage2_block3_branch2a, num_filter=128, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage2_block3_branch2b = mx.symbol.BatchNorm(name='bn_stage2_block3_branch2b', data=conv_stage2_block3_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block3_branch2b = bn_stage2_block3_branch2b
    relu_stage2_block3_branch2b = mx.symbol.Activation(name='relu_stage2_block3_branch2b',
                                                       data=scale_stage2_block3_branch2b, act_type='relu')
    conv_stage2_block3_branch2c = mx.symbol.Convolution(name='conv_stage2_block3_branch2c',
                                                        data=relu_stage2_block3_branch2b, num_filter=512, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage2_block3_branch2c = mx.symbol.BatchNorm(name='bn_stage2_block3_branch2c', data=conv_stage2_block3_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block3_branch2c = bn_stage2_block3_branch2c
    eltwise_stage2_block3 = mx.symbol.broadcast_add(name='eltwise_stage2_block3',
                                                    *[relu_stage2_block2, scale_stage2_block3_branch2c])
    relu_stage2_block3 = mx.symbol.Activation(name='relu_stage2_block3', data=eltwise_stage2_block3, act_type='relu')
    conv_stage2_block4_branch2a = mx.symbol.Convolution(name='conv_stage2_block4_branch2a', data=relu_stage2_block3,
                                                        num_filter=128, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                        no_bias=False)
    bn_stage2_block4_branch2a = mx.symbol.BatchNorm(name='bn_stage2_block4_branch2a', data=conv_stage2_block4_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block4_branch2a = bn_stage2_block4_branch2a
    relu_stage2_block4_branch2a = mx.symbol.Activation(name='relu_stage2_block4_branch2a',
                                                       data=scale_stage2_block4_branch2a, act_type='relu')
    conv_stage2_block4_branch2b = mx.symbol.Convolution(name='conv_stage2_block4_branch2b',
                                                        data=relu_stage2_block4_branch2a, num_filter=128, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage2_block4_branch2b = mx.symbol.BatchNorm(name='bn_stage2_block4_branch2b', data=conv_stage2_block4_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block4_branch2b = bn_stage2_block4_branch2b
    relu_stage2_block4_branch2b = mx.symbol.Activation(name='relu_stage2_block4_branch2b',
                                                       data=scale_stage2_block4_branch2b, act_type='relu')
    conv_stage2_block4_branch2c = mx.symbol.Convolution(name='conv_stage2_block4_branch2c',
                                                        data=relu_stage2_block4_branch2b, num_filter=512, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage2_block4_branch2c = mx.symbol.BatchNorm(name='bn_stage2_block4_branch2c', data=conv_stage2_block4_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block4_branch2c = bn_stage2_block4_branch2c
    eltwise_stage2_block4 = mx.symbol.broadcast_add(name='eltwise_stage2_block4',
                                                    *[relu_stage2_block3, scale_stage2_block4_branch2c])
    relu_stage2_block4 = mx.symbol.Activation(name='relu_stage2_block4', data=eltwise_stage2_block4, act_type='relu')
    conv_stage2_block5_branch2a = mx.symbol.Convolution(name='conv_stage2_block5_branch2a', data=relu_stage2_block4,
                                                        num_filter=128, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                        no_bias=False)
    bn_stage2_block5_branch2a = mx.symbol.BatchNorm(name='bn_stage2_block5_branch2a', data=conv_stage2_block5_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block5_branch2a = bn_stage2_block5_branch2a
    relu_stage2_block5_branch2a = mx.symbol.Activation(name='relu_stage2_block5_branch2a',
                                                       data=scale_stage2_block5_branch2a, act_type='relu')
    conv_stage2_block5_branch2b = mx.symbol.Convolution(name='conv_stage2_block5_branch2b',
                                                        data=relu_stage2_block5_branch2a, num_filter=128, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage2_block5_branch2b = mx.symbol.BatchNorm(name='bn_stage2_block5_branch2b', data=conv_stage2_block5_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block5_branch2b = bn_stage2_block5_branch2b
    relu_stage2_block5_branch2b = mx.symbol.Activation(name='relu_stage2_block5_branch2b',
                                                       data=scale_stage2_block5_branch2b, act_type='relu')
    conv_stage2_block5_branch2c = mx.symbol.Convolution(name='conv_stage2_block5_branch2c',
                                                        data=relu_stage2_block5_branch2b, num_filter=512, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage2_block5_branch2c = mx.symbol.BatchNorm(name='bn_stage2_block5_branch2c', data=conv_stage2_block5_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage2_block5_branch2c = bn_stage2_block5_branch2c
    eltwise_stage2_block5 = mx.symbol.broadcast_add(name='eltwise_stage2_block5',
                                                    *[relu_stage2_block4, scale_stage2_block5_branch2c])
    relu_stage2_block5 = mx.symbol.Activation(name='relu_stage2_block5', data=eltwise_stage2_block5, act_type='relu')
    drop_stage2 = mx.symbol.Dropout(name='drop_stage2', data=relu_stage2_block5, p=0.200000)
    conv_stage3_block0_proj_shortcut = mx.symbol.Convolution(name='conv_stage3_block0_proj_shortcut', data=drop_stage2,
                                                             num_filter=1024, pad=(0, 0), kernel=(1, 1), stride=(2, 2),
                                                             no_bias=False)
    bn_stage3_block0_proj_shortcut = mx.symbol.BatchNorm(name='bn_stage3_block0_proj_shortcut',
                                                         data=conv_stage3_block0_proj_shortcut, use_global_stats=False,
                                                         fix_gamma=False, eps=0.000100)
    scale_stage3_block0_proj_shortcut = bn_stage3_block0_proj_shortcut
    conv_stage3_block0_branch2a = mx.symbol.Convolution(name='conv_stage3_block0_branch2a', data=drop_stage2,
                                                        num_filter=256, pad=(0, 0), kernel=(1, 1), stride=(2, 2),
                                                        no_bias=False)
    bn_stage3_block0_branch2a = mx.symbol.BatchNorm(name='bn_stage3_block0_branch2a', data=conv_stage3_block0_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage3_block0_branch2a = bn_stage3_block0_branch2a
    relu_stage3_block0_branch2a = mx.symbol.Activation(name='relu_stage3_block0_branch2a',
                                                       data=scale_stage3_block0_branch2a, act_type='relu')
    conv_stage3_block0_branch2b = mx.symbol.Convolution(name='conv_stage3_block0_branch2b',
                                                        data=relu_stage3_block0_branch2a, num_filter=256, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage3_block0_branch2b = mx.symbol.BatchNorm(name='bn_stage3_block0_branch2b', data=conv_stage3_block0_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage3_block0_branch2b = bn_stage3_block0_branch2b
    relu_stage3_block0_branch2b = mx.symbol.Activation(name='relu_stage3_block0_branch2b',
                                                       data=scale_stage3_block0_branch2b, act_type='relu')
    conv_stage3_block0_branch2c = mx.symbol.Convolution(name='conv_stage3_block0_branch2c',
                                                        data=relu_stage3_block0_branch2b, num_filter=1024, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage3_block0_branch2c = mx.symbol.BatchNorm(name='bn_stage3_block0_branch2c', data=conv_stage3_block0_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage3_block0_branch2c = bn_stage3_block0_branch2c
    eltwise_stage3_block0 = mx.symbol.broadcast_add(name='eltwise_stage3_block0',
                                                    *[scale_stage3_block0_proj_shortcut, scale_stage3_block0_branch2c])
    relu_stage3_block0 = mx.symbol.Activation(name='relu_stage3_block0', data=eltwise_stage3_block0, act_type='relu')
    conv_stage3_block1_branch2a = mx.symbol.Convolution(name='conv_stage3_block1_branch2a', data=relu_stage3_block0,
                                                        num_filter=256, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                        no_bias=False)
    bn_stage3_block1_branch2a = mx.symbol.BatchNorm(name='bn_stage3_block1_branch2a', data=conv_stage3_block1_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage3_block1_branch2a = bn_stage3_block1_branch2a
    relu_stage3_block1_branch2a = mx.symbol.Activation(name='relu_stage3_block1_branch2a',
                                                       data=scale_stage3_block1_branch2a, act_type='relu')
    conv_stage3_block1_branch2b = mx.symbol.Convolution(name='conv_stage3_block1_branch2b',
                                                        data=relu_stage3_block1_branch2a, num_filter=256, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage3_block1_branch2b = mx.symbol.BatchNorm(name='bn_stage3_block1_branch2b', data=conv_stage3_block1_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage3_block1_branch2b = bn_stage3_block1_branch2b
    relu_stage3_block1_branch2b = mx.symbol.Activation(name='relu_stage3_block1_branch2b',
                                                       data=scale_stage3_block1_branch2b, act_type='relu')
    conv_stage3_block1_branch2c = mx.symbol.Convolution(name='conv_stage3_block1_branch2c',
                                                        data=relu_stage3_block1_branch2b, num_filter=1024, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage3_block1_branch2c = mx.symbol.BatchNorm(name='bn_stage3_block1_branch2c', data=conv_stage3_block1_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage3_block1_branch2c = bn_stage3_block1_branch2c
    eltwise_stage3_block1 = mx.symbol.broadcast_add(name='eltwise_stage3_block1',
                                                    *[relu_stage3_block0, scale_stage3_block1_branch2c])
    relu_stage3_block1 = mx.symbol.Activation(name='relu_stage3_block1', data=eltwise_stage3_block1, act_type='relu')
    conv_stage3_block2_branch2a = mx.symbol.Convolution(name='conv_stage3_block2_branch2a', data=relu_stage3_block1,
                                                        num_filter=256, pad=(0, 0), kernel=(1, 1), stride=(1, 1),
                                                        no_bias=False)
    bn_stage3_block2_branch2a = mx.symbol.BatchNorm(name='bn_stage3_block2_branch2a', data=conv_stage3_block2_branch2a,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage3_block2_branch2a = bn_stage3_block2_branch2a
    relu_stage3_block2_branch2a = mx.symbol.Activation(name='relu_stage3_block2_branch2a',
                                                       data=scale_stage3_block2_branch2a, act_type='relu')
    conv_stage3_block2_branch2b = mx.symbol.Convolution(name='conv_stage3_block2_branch2b',
                                                        data=relu_stage3_block2_branch2a, num_filter=256, pad=(1, 1),
                                                        kernel=(3, 3), stride=(1, 1), no_bias=False)
    bn_stage3_block2_branch2b = mx.symbol.BatchNorm(name='bn_stage3_block2_branch2b', data=conv_stage3_block2_branch2b,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage3_block2_branch2b = bn_stage3_block2_branch2b
    relu_stage3_block2_branch2b = mx.symbol.Activation(name='relu_stage3_block2_branch2b',
                                                       data=scale_stage3_block2_branch2b, act_type='relu')
    conv_stage3_block2_branch2c = mx.symbol.Convolution(name='conv_stage3_block2_branch2c',
                                                        data=relu_stage3_block2_branch2b, num_filter=1024, pad=(0, 0),
                                                        kernel=(1, 1), stride=(1, 1), no_bias=False)
    bn_stage3_block2_branch2c = mx.symbol.BatchNorm(name='bn_stage3_block2_branch2c', data=conv_stage3_block2_branch2c,
                                                    use_global_stats=False, fix_gamma=False, eps=0.000100)
    scale_stage3_block2_branch2c = bn_stage3_block2_branch2c
    eltwise_stage3_block2 = mx.symbol.broadcast_add(name='eltwise_stage3_block2',
                                                    *[relu_stage3_block1, scale_stage3_block2_branch2c])
    relu_stage3_block2 = mx.symbol.Activation(name='relu_stage3_block2', data=eltwise_stage3_block2, act_type='relu')
    pool = mx.symbol.Pooling(name='pool', data=relu_stage3_block2, pooling_convention='full', pad=(0, 0), kernel=(7, 7),
                             stride=(1, 1), pool_type='avg')
    drop_pool = mx.symbol.Dropout(name='drop_pool', data=pool, p=0.400000)
    flatten_0 = mx.symbol.Flatten(name='flatten_0', data=drop_pool)
    fc_2c = mx.symbol.FullyConnected(name='fc_2c', data=flatten_0, num_hidden=2, no_bias=False)
    if is_train:
        return mx.symbol.SoftmaxOutput(name='prob', data=fc_2c, label=labels[0])

    return mx.symbol.SoftmaxActivation(name='prob', data=fc_2c)
