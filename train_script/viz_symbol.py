import argparse
import sys
import importlib
import mxnet as mx


def parse_args(args):
    parser = argparse.ArgumentParser(description='save network symbol and visual it')
    parser.add_argument('network_dir', type=str, help='the network dir')
    parser.add_argument('-m', '--mode', type=str, default='test', choices=['train', 'test'],
                        help='train symbol or test symbol, default is test')
    parser.add_argument('-s', '--shapes', type=str, action='append',
                        help='shape for arguments, format: arg_name,s1,s2,...,sn, example: data,1,3,224,224')
    parser.add_argument('-p', '--print_arguments', action='store_true')
    return parser.parse_args(args)


def viz_symbol(_args):
    args = parse_args(_args)

    sys.path.insert(0, args.network_dir)
    train_symbol = importlib.import_module('train_symbol')

    is_train = (args.mode == 'train')
    symbol = train_symbol.get_symbol(is_train=is_train)
    symbol.save(args.network_dir + '/symbol_%s.json' % ('train' if is_train else 'test'))
    if args.print_arguments:
        print('arg_names are: ')
        print(symbol.list_arguments())
    viz_shape = None
    if args.shapes is not None and len(args.shapes) > 0:
        viz_shape = {}
        for shape in args.shapes:
            items = shape.replace('\'', '').replace('"', '').split(',')
            viz_shape[items[0]] = tuple([int(s) for s in items[1:]])
    mx.viz.plot_network(symbol, shape=viz_shape).view()


if __name__ == '__main__':
    viz_symbol(sys.argv[1:])
