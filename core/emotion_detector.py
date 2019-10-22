import cv2
import numpy as np


from core.utils.preprocessor import preprocess_input


class EmotionDetector:
    def __init__(self, labels, face_detector,
                 emotion_classifier, emotion_offsets):
        self.labels = labels
        self.face_detector = face_detector
        self.classifier = emotion_classifier
        self.emotion_offsets = emotion_offsets  # 扩大表情相框

    def detect_biggest(self, bgr_image):
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray_face, coord = self.face_detector.get_biggest_face(gray_image, self.emotion_offsets)
        emotion_target_size = self.classifier.input_shape[1:3]
        if np.size(gray_face) is 0:
            return None, coord
        gray_face = cv2.resize(gray_face, emotion_target_size)
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        # 返回预测值，返回形式为数组，值是emotion_labels中标签对应下标的值
        emotion_prediction = self.classifier.predict(gray_face)
        return emotion_prediction, coord
