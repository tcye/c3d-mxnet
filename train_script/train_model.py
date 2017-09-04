import os
import sys
import importlib
import logging
import mxnet as mx
import argparse
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('network_dir', type=str, help='the network dir')
    parser.add_argument('--test', action='store_true', help='test network')
    return parser.parse_args()


def _load_model(model_prefix, begin_epoch, rank=0):
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % rank
    logging.info('loading model from %s-%04d...' % (model_prefix, begin_epoch))
    return mx.model.load_checkpoint(model_prefix, begin_epoch)


if __name__ == '__main__':
    args = parse_args()
    sys.path.insert(0, args.network_dir)
    train_info = importlib.import_module('train_info')
    train_symbol = importlib.import_module('train_symbol')
    train_data_iter = importlib.import_module('train_data_iter')
    train_params = train_info.get_train_param()

    try:
        symbol = train_symbol.get_symbol(not args.test)
    except AttributeError:
        symbol = None
        if 'begin_epoch' not in train_params:
            raise Exception('symbol definition can only load from checkpoint and symbol.get_symbol')

    fixed_param_names = train_symbol.fixed_param_names if hasattr(train_symbol, 'fixed_param_names') else []

    # logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    head = '%(asctime)-15s Node[' + str(train_params['kvstore'].rank) + '] %(message)s'
    formatter = logging.Formatter(head)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logging.info('start with arguments:')
    logging.info(train_params)

    # save model
    model_prefix = sys.argv[1] + '/model/'
    if not os.path.exists(model_prefix):
        os.makedirs(model_prefix)
    model_prefix += '/model'
    checkpoint = mx.callback.do_checkpoint(model_prefix)

    # load model
    if 'begin_epoch' in train_params:
        dummy_sym, arg_params, aux_params = _load_model(model_prefix, train_params['begin_epoch'],
                                                        train_params['kvstore'].rank)
        if symbol is not None:
            if symbol.tojson() != dummy_sym.tojson():
                warnings.warn('load symbol is not equal to pre-define symbol')
        else:
            symbol = dummy_sym
    elif hasattr(train_symbol, 'pretrained_model'):
        logging.info('loading pretrained model from %s' % train_symbol.pretrained_model)
        save_dict = mx.nd.load(train_symbol.pretrained_model)
        arg_params = {}
        aux_params = {}
        loaded_params = []
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            loaded_params.append(name)
            if tp == 'arg':
                arg_params[name] = v
            if tp == 'aux':
                aux_params[name] = v
        logging.info('')
        logging.info('loaded_params:')
        logging.info(loaded_params)
        # list match and unmatch params
        match_params = []
        unmatch_params = []
        for p in symbol.list_arguments():
            if p in arg_params or p in aux_params:
                match_params.append(p)
            else:
                unmatch_params.append(p)

        logging.info('')
        logging.info('match_params:')
        logging.info(match_params)

        logging.info('')
        logging.info('unmatch_params:')
        logging.info(unmatch_params)

        if hasattr(train_symbol, 'fix_pretrained_model_params') and train_symbol.fix_pretrained_model_params:
            for k in arg_params:
                if k not in fixed_param_names:
                    fixed_param_names.append(k)
    else:
        arg_params = None
        aux_params = None

    logging.info('')
    logging.info('fixed_param_names')
    logging.info(fixed_param_names)
    trainable_param_names = set(symbol.list_arguments()) - set(fixed_param_names)
    logging.info('')
    logging.info('trainable_param_names')
    logging.info(trainable_param_names)

    devs = mx.cpu()
    if hasattr(train_info, 'gpus') and train_info.gpus:
        devs = [mx.gpu(int(i)) for i in train_info.gpus.split(',')]
    # create model
    model = mx.mod.Module(context=devs, symbol=symbol, data_names=train_symbol.data_names,
                          label_names=train_symbol.label_names,
                          fixed_param_names=fixed_param_names, logger=logger)

    eval_metrics = ['accuracy']

    if args.test:
        eval_data = train_data_iter.get_test_data_iter(data_names=train_symbol.data_names,
                                                       label_names=train_symbol.label_names)
        model.bind(data_shapes=eval_data.provide_data, label_shapes=eval_data.provide_label, for_training=False)
        model.init_params(arg_params=arg_params, aux_params=aux_params, allow_missing=False)
        logging.info(model.score(eval_data, eval_metrics))
    else:
        data_iters = train_data_iter.get_data_iter()
        batch_end_callbacks = [mx.callback.Speedometer(train_data_iter.batch_size, 50)]
        model.fit(train_data=data_iters[0],
                  eval_data=None if len(data_iters) == 1 else data_iters[1],
                  eval_metric=eval_metrics,
                  batch_end_callback=batch_end_callbacks,
                  # epoch_end_callback=checkpoint,
                  arg_params=arg_params,
                  aux_params=aux_params,
                  **train_params)
