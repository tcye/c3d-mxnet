import mxnet as mx

gpus = '0,1,2,3'


def get_train_param(begin_num_update=0):
    '''
    get the training model
    '''
    param = dict()
    param['num_epoch'] = 500
    param['optimizer'] = 'sgd'
    param['optimizer_params'] = {
        'begin_num_update': begin_num_update,
        'learning_rate': 0.001,
        'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=100000, factor=0.1),
        'wd': 0.00001,
        'momentum': 0.9,
    }
    param['initializer'] = mx.initializer.Xavier(factor_type='in', magnitude=2.34)
    param['kvstore'] = mx.kvstore.create('local')
    return param
