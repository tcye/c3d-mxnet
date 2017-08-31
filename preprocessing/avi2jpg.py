# -*- coding: <encoding name> -*-

import cv2
import os
import logging

def extract_frames_v1(path, num_to_extract=16):
    video = cv2.VideoCapture(path)
    frame_cnt = video.get(cv2.CAP_PROP_FRAME_COUNT)
    step = int(frame_cnt/num_to_extract)
    frames = []
    for i in range(num_to_extract):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        succ, frame = video.read()
        if not succ:
            logging.warning('failed to read frame(%d) from video(%s)' % (i * step, path))
            break
        else:
            frames.append(frame)
    video.release()
    return frames

def extract_frames_v2(path, num_to_extract=16):
    video = cv2.VideoCapture(path)
    frames = []
    while 1:
        succ, frame = video.read()
        if not succ or frame is None:
            break
        frames.append(frame)
    video.release()

    frame_cnt = len(frames)
    if frame_cnt < num_to_extract:
        logging.warning('failed to read enough frames for video(%s)', path)
        return frames

    step = int(frame_cnt/num_to_extract)
    res = []
    for i in range(num_to_extract):
        res.append(frames[i * step])
    return res

extract_frames = extract_frames_v2

def save_frames(savedir, frames):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(savedir, '%d.jpg' % i), frame)
