
import os
import argparse
import multiprocessing
import logging
import avi2jpg

num_processed = 0

def cvt_dataset(src_dir, dest_dir):
    if not os.path.exists(src_dir):
        raise Exception('cvt_dataset: src_dir does not exists!')

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if os.listdir(dest_dir):
        raise Exception('cvt_dataset: dest_dir must be an empty dir!')

    for node in os.listdir(src_dir):
        if os.path.isdir(os.path.join(src_dir, node)):
            cvt_dataset(os.path.join(src_dir, node), os.path.join(dest_dir, node))
        else:
            name, ext = os.path.splitext(node)
            if ext != '.avi':
                continue
            frames = avi2jpg.extract_frames(os.path.join(src_dir, node))
            avi2jpg.save_frames(os.path.join(dest_dir, name), frames)

            global num_processed
            num_processed += 1
            if num_processed % 30 == 0:
                # logging.info('video(%s) has been processed', num_processed)
                print('video(%s) has been processed' % num_processed)

def process(src_dirs, dest_dirs):
    for src_dir, dest_dir in zip(src_dirs, dest_dirs):
        cvt_dataset(src_dir, dest_dir)

def parse_args():
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('src_dir', type=str, help='the dataset top dir')
    parser.add_argument('dest_dir', type=str, help='the destination dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print('start convert')

    pool = multiprocessing.Pool(processes=10)
    for node in os.listdir(args.src_dir):
        if os.path.isdir(os.path.join(args.src_dir, node)):
            pool.apply_async(process, ([os.path.join(args.src_dir, node), ], [os.path.join(args.dest_dir, node)]))
        else:
            print('error')
    pool.close()
    pool.join()

    print('end convert')
