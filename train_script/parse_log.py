"""
parse mxnet output log into a markdown table
"""
import argparse
import sys
import re

parser = argparse.ArgumentParser(description='Parse mxnet output log')
parser.add_argument('logfile', type=str,
                    help='the log file for parsing')
parser.add_argument('--format', type=str, default='markdown',
                    choices=['markdown', 'none'],
                    help='the format of the parsed outout')
args = parser.parse_args()

with open(args.logfile) as f:
    lines = f.readlines()

loss_id = 0
res = [re.compile('.*Epoch\[(\d+)\] Train-accuracy=([.\d]+)'),
       re.compile('.*Epoch\[(\d+)\] Validation-accuracy=([.\d]+)'),
       re.compile('.*Epoch\[(\d+)\] Time cost=([.\d]+)')]

data = {}
max_val = None
max_train = None
for l in lines:
    i = 0
    for r in res:
        m = r.match(l)
        if m is not None:
            break
        i += 1
    if m is None:
        continue

    assert len(m.groups()) == 2
    epoch = int(m.groups()[0]) + 1
    val = float(m.groups()[1])
    if i != 2:
        val *= 100

    if epoch not in data:
        data[epoch] = [0] * len(res)

    data[epoch][i] = val

    if i == 0 and (max_train is None or max_train[1] < val):
        max_train = [epoch, val]

    if i == 1 and (max_val is None or max_val[1] < val):
        max_val = [epoch, val]

out_f = open(args.logfile + ".acc", 'w')
if args.format == 'markdown':
    out_f.write("| epoch | train-accuracy | val-accuracy | time |\n")
    out_f.write("| --- | --- | --- | --- |\n")
    for k, v in data.items():
        out_f.write("| %2d | %.2f | %.2f | % .1f |\n" % (k, v[0], v[1], v[2]))
elif args.format == 'none':
    out_f.write("epoch\ttrain-accuracy\tvalid-accuracy\ttime\n")
    for k, v in data.items():
        out_f.write("%2d\t%.2f\t%.2f\t%.1f\n" % (k, v[0], v[1], v[2]))

print("max train accuracy: epoch %2d -- %.2f" % (max_train[0], max_train[1]))
print("max val accuracy: epoch %2d -- %.2f" % (max_val[0], max_val[1]))

rm_str = 'ls model/model-*.params | grep -v %04d | xargs rm' % max_val[0]
print(rm_str)

import getpass
import os
import socket

rsync_str = 'rsync -auvl %s@%s:%s/model/model-%04d.params model/' % (
    getpass.getuser(), socket.gethostname(), os.path.dirname(os.path.abspath(args.logfile)), max_val[0])
print(rsync_str)
