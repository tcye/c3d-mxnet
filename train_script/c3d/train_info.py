import mxnet as mx


gpus = '0,1,2,3'

def get_train_param(begin_num_update=0):
    '''
    get the training model
    '''
    param = dict()

    param['begin_epoch'] = 15
    param['num_epoch'] = 20000
    param['kvstore'] = mx.kvstore.create('local')

    # param['optimizer'] = 'rmsprop'
    # param['optimizer_params'] = {
    # 'begin_num_update': begin_num_update,
    # 'learning_rate' : 0.01,
    # 'lr_scheduler': mx.lr_scheduler.InvScheduler(gamma=0.0001, power=0.75),
    # 'wd': 0.0005,
    # 'gamma1': 0.9,
    # 'gamma2': 0.5,
    # 'clip_gradient' : 5,
    # 'global_clip': True,
    # }

    param['optimizer'] = 'sgd'
    param['optimizer_params'] = {
        'begin_num_update': begin_num_update,
        'learning_rate' : 0.001,
        'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=100000, factor=0.1),
        'wd': 0.00001,
        'momentum': 0.9,
        # 'clip_gradient' : 5,
        # 'global_clip': True,
    }

    # param['optimizer'] = 'adam'
    # param['optimizer_params'] = {
        # # 'begin_num_update': begin_num_update,
        # 'learning_rate': 0.0001,
        # # 'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=100000, factor=0.1),
        # 'wd': 0.0001,
    # }

    # initializer
    param['initializer'] = mx.initializer.Xavier(  # rnd_type='gaussian',
        rnd_type='uniform',
        factor_type='avg',
        magnitude=2.34)

    return param
