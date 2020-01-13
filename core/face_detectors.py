from core.utils.inference import make_face_coordinates, apply_offsets, load_detection_model
import config

class CvFaceDetector:
    def __init__(self):
        self.detector = load_detection_model(config.FACE_DETECTOR_PATH) # 只构建一次模型对象，保存下来重复使用

    def detect_faces(self, gray_image_array):
        return self.detector.detectMultiScale(gray_image_array, 1.3, 5)

    def get_biggest_face(self, gray_image, emotion_offsets):
        detected_faces = self.detect_faces(gray_image)
        size = 0
        coord = [0, 0, 0, 0]
        # 选出最大的人脸
        for detected_face in detected_faces:
            face_coordinates = make_face_coordinates(detected_face)
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            new_size = abs((x1 - x2) * (y1 - y2))
            if new_size > size:
                coord[0] = x1
                coord[1] = x2
                coord[2] = y1
                coord[3] = y2
                size = new_size

        return gray_image[coord[2]:coord[3], coord[0]:coord[1]], coord
