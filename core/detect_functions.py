import os
import cv2
import numpy as np
import json
import time as Time

import config
from core.utils.datasets import get_labels
from core.utils.inference import draw_text
from exceptions import APIException
from collections import Counter


class FrameEmotion:
    def __init__(self, time, emotion, rate):
        self.time = time
        self.emotion = emotion
        self.rate = rate


class EmotionStreamResponse:
    def __init__(self, emotion_stream, pure_color_rate):
        self.emotion_stream = emotion_stream
        self.pure_color_rate = pure_color_rate


# 根据固定时长抽帧并分析
def get_emotion_stream(video_path, detector, frame_interval_ms):
    return get_emotion_stream_cut(video_path, detector, frame_interval_ms, 0, -1)


# 只分析指定范围内的视频，end_ms小于零代表分析到末尾
def get_emotion_stream_cut(video_path, detector, frame_interval_ms, start_ms, end_ms):
    video_capture = cv2.VideoCapture(video_path)
    try:
        video_capture.isOpened()
    except Exception as ex:
        raise ex
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    start_frame_no = ms2frame(fps, start_ms) + 1  # 加一防止为0的情况
    end_frame_no = ms2frame(fps, end_ms)

    if end_ms < 0 or end_frame_no > frame_count:
        end_frame_no = frame_count
    if start_frame_no < 1:
        start_frame_no = 1
    if start_frame_no > end_frame_no:
        return []

    interval_frame_num = ms2frame(fps, frame_interval_ms)
    emotion_stream = []
    if interval_frame_num < 1:
        interval_frame_num = 1  # 防止帧间隔为0的情况
    while start_frame_no <= end_frame_no:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)
        frame = video_capture.read()[1]
        if frame is None or np.size(frame) is 0:
            start_frame_no += interval_frame_num
            continue
        time = int(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        prediction, _ = detector.detect_biggest(frame)
        if prediction is None:
            start_frame_no += interval_frame_num
            continue
        # 计算最最大表情值所占百分比，再乘以1000，转为整数，再除以1000，变为float相当于保留三位小数
        rate = int(np.max(prediction) / np.sum(prediction) * 1000) / 1000.0
        emotion_text = get_labels('fer2013')[np.argmax(prediction)]
        frame_emotion = FrameEmotion(time, emotion_text, rate)
        # 转换为便于转成json的字典格式
        emotion_stream.append(frame_emotion.__dict__)
        start_frame_no += interval_frame_num
    video_capture.release()
    return emotion_stream


# 只分析指定范围内的视频，end_ms小于零代表分析到末尾，增添超时控制
def get_emotion_stream_cut_with_timeout(video_path, detector, frame_interval_ms, start_ms, end_ms, time_limit_second):
    video_capture = cv2.VideoCapture(video_path)
    try:
        video_capture.isOpened()
    except Exception as ex:
        raise ex
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    start_frame_no = ms2frame(fps, start_ms) + 1  # 加一防止为0的情况
    end_frame_no = ms2frame(fps, end_ms)

    if end_ms < 0 or end_frame_no > frame_count:
        end_frame_no = frame_count
    if start_frame_no < 1:
        start_frame_no = 1
    if start_frame_no > end_frame_no:
        return []

    interval_frame_num = ms2frame(fps, frame_interval_ms)
    emotion_stream = []
    if interval_frame_num < 1:
        interval_frame_num = 1  # 防止帧间隔为0的情况

    start_time = Time.time()
    while start_frame_no <= end_frame_no:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)
        frame = video_capture.read()[1]
        if frame is None or np.size(frame) is 0:
            start_frame_no += interval_frame_num
            continue
        time = int(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        prediction, _ = detector.detect_biggest(frame)
        now_time = Time.time()
        if now_time - start_time > time_limit_second:
            raise APIException(500, config.TIMEOUT)

        if prediction is None:
            start_frame_no += interval_frame_num
            continue
        # 计算最最大表情值所占百分比，再乘以1000，转为整数，再除以1000，变为float相当于保留三位小数
        rate = int(np.max(prediction) / np.sum(prediction) * 1000) / 1000.0
        emotion_text = get_labels('fer2013')[np.argmax(prediction)]
        frame_emotion = FrameEmotion(time, emotion_text, rate)
        # 转换为便于转成json的字典格式
        emotion_stream.append(frame_emotion.__dict__)
        start_frame_no += interval_frame_num
    video_capture.release()
    return emotion_stream


# 只分析指定范围内的视频，end_ms小于零代表分析到末尾，增添超时控制,黑屏检测
def get_emotion_stream_cut_with_timeout_pure_color_detect(video_path, detector, frame_interval_ms, start_ms, end_ms,
                                                          time_limit_second):
    video_capture = cv2.VideoCapture(video_path)
    try:
        video_capture.isOpened()
    except Exception as ex:
        raise ex
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    start_frame_no = ms2frame(fps, start_ms) + 1  # 加一防止为0的情况
    end_frame_no = ms2frame(fps, end_ms)

    if end_ms < 0 or end_frame_no > frame_count:
        end_frame_no = frame_count
    if start_frame_no < 1:
        start_frame_no = 1
    if start_frame_no > end_frame_no:
        return []

    interval_frame_num = ms2frame(fps, frame_interval_ms)
    emotion_stream = []
    if interval_frame_num < 1:
        interval_frame_num = 1  # 防止帧间隔为0的情况

    pure_color_count = 0
    start_time = Time.time()
    while start_frame_no <= end_frame_no:
        # 超时检测
        time = int(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        now_time = Time.time()
        if now_time - start_time > time_limit_second:
            raise APIException(500, config.TIMEOUT)

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_no)
        frame = video_capture.read()[1]
        if frame is None or np.size(frame) is 0:
            start_frame_no += interval_frame_num
            continue
        count_pair = Counter(frame.flatten()).most_common(1)
        color_rate = count_pair[0][1] / (len(frame) * len(frame[0]))
        if color_rate > 0.86:
            frame_emotion = FrameEmotionWithBlack(time, "", rate)
            # 转换为便于转成json的字典格式
            pure_color_count = pure_color_count + 1
            emotion_stream.append(frame_emotion.__dict__)
            continue

        prediction, _ = detector.detect_biggest(frame)
        if prediction is None:
            start_frame_no += interval_frame_num
            continue
        # 计算最最大表情值所占百分比，再乘以1000，转为整数，再除以1000，变为float相当于保留三位小数
        rate = int(np.max(prediction) / np.sum(prediction) * 1000) / 1000.0
        emotion_text = get_labels('fer2013')[np.argmax(prediction)]
        frame_emotion = FrameEmotion(time, emotion_text, rate)
        # 转换为便于转成json的字典格式
        emotion_stream.append(frame_emotion.__dict__)
        start_frame_no += interval_frame_num
    video_capture.release()
    return EmotionStreamResponse(emotion_stream, ((pure_color_count * 1000) / end_frame_no) / 1000.0)


def ms2frame(fps, time_ms):
    interval_frame_num = int(time_ms / 1000 * fps)  # 间隔帧数
    return interval_frame_num


def get_image_emotion(image, detector):
    return detector.detect_biggest(image)


def save_biggest_emotion_images(video_path, save_path, detector, frame_interval_ms):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        return
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    # 创建视频存储文件夹
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    frame_no = 1  # 抽取帧的序号
    interval_frame_num = int(frame_interval_ms / 1000 * fps)  # 间隔帧数
    if interval_frame_num < 1:
        interval_frame_num = 1  # 防止帧间隔为0的情况
    while frame_no <= frame_count:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        frame = video_capture.read()[1]
        if frame is None or np.size(frame) is 0:
            frame_no += interval_frame_num
            continue
        time = int(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        prediction, coord = detector.detect_biggest(frame)
        if prediction is None:
            frame_no += interval_frame_num
            continue
        emotion_probability = np.max(prediction)
        frame_no += interval_frame_num
        emotion_label_arg = np.argmax(prediction)
        emotion_text = detector.labels[emotion_label_arg]
        # 根据情绪选择颜色
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        text = emotion_text + ' ' + str(time / 1000) + 's'
        cv2.rectangle(frame, (coord[0], coord[2]), (coord[1], coord[3]), color, 2)
        draw_text(coord, frame, text,
                  color, 0, -45, 1, 1)
        image_path = save_path + '/' + str(frame_no) + '.png'
        cv2.imwrite(image_path, frame)

    video_capture.release()
    return


def save_biggest_emotion_images_cut(video_path, save_path, detector, frame_interval_ms, start_ms, end_ms):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        return
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    start_frame_no = int(start_ms / 1000 * fps + 1)
    end_frame_no = int(end_ms / 1000 * fps)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if start_frame_no < 0 or end_frame_no > frame_count:
        return
    # 创建视频存储文件夹
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    frame_no = 1  # 抽取帧的序号
    interval_frame_num = int(frame_interval_ms / 1000 * fps)  # 间隔帧数
    if interval_frame_num < 1:
        interval_frame_num = 1  # 防止帧间隔为0的情况
    while frame_no <= end_frame_no:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        frame = video_capture.read()[1]
        if frame is None or np.size(frame) is 0:
            frame_no += interval_frame_num
            continue
        time = int(video_capture.get(cv2.CAP_PROP_POS_MSEC))
        prediction, coord = detector.detect_biggest(frame)
        if prediction is None:
            frame_no += interval_frame_num
            continue
        emotion_probability = np.max(prediction)
        frame_no += interval_frame_num
        emotion_label_arg = np.argmax(prediction)
        emotion_text = detector.labels[emotion_label_arg]
        # 根据情绪选择颜色
        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        text = emotion_text + ' ' + str(time / 1000) + 's'
        cv2.rectangle(frame, (coord[0], coord[2]), (coord[1], coord[3]), color, 2)
        draw_text(coord, frame, text,
                  color, 0, -45, 1, 1)
        image_path = save_path + '/' + str(frame_no) + '.png'
        cv2.imwrite(image_path, frame)

    video_capture.release()
    return


# 根据固定时长抽帧并分析,返回为json格式
def get_emotion_stream_json(video_path, detector, frame_interval_ms):
    emotion_stream = get_emotion_stream(video_path, detector, frame_interval_ms)
    emotion_stream_json = json.dumps(emotion_stream)
    return emotion_stream_json


# 只分析指定范围内的视频，返回json
def get_emotion_stream_cut_json(video_path, detector, frame_interval_ms, start_ms, end_ms):
    emotion_stream = get_emotion_stream_cut(video_path, detector, frame_interval_ms, start_ms, end_ms)
    emotion_stream_json = json.dumps(emotion_stream)
    return emotion_stream_json


# 只分析指定范围内的视频，返回json
def get_emotion_stream_cut_json_with_timeout(video_path, detector, frame_interval_ms, start_ms, end_ms, timeout_second):
    emotion_stream = get_emotion_stream_cut_with_timeout(video_path, detector, frame_interval_ms, start_ms, end_ms,
                                                         timeout_second)
    emotion_stream_json = json.dumps(emotion_stream)
    return emotion_stream_json


# 只分析指定范围内的视频，返回json
def get_emotion_stream_cut_json_with_timeout_and_pure(video_path, detector, frame_interval_ms, start_ms, end_ms,
                                                      timeout_second):
    emotion_stream = get_emotion_stream_cut_with_timeout_pure_color_detect(video_path, detector, frame_interval_ms,
                                                                           start_ms, end_ms,
                                                                           timeout_second)
    response_json = json.dumps(emotion_stream)
    return response_json
