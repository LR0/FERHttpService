import psutil
import os
import logging
import json
from flask import Flask, request, Response, jsonify
from logging.handlers import TimedRotatingFileHandler
from flasgger import Swagger
from keras.engine.saving import load_model

import config
from core import detect_functions
from core.emotion_detector import EmotionDetector
from core.face_detectors import DlibFaceDetector
from core.utils.datasets import get_labels

# 抑制cpu占用率
from exceptions import APIException

app = Flask(__name__)
fileTimeHandler = TimedRotatingFileHandler(
    config.LOG_FILE_PATH, when="D", interval=1, backupCount=60,
    encoding="UTF-8", delay=False, utc=True)
formatter = logging.Formatter(config.LOG_FORMATTER)  # 每行日志的前缀设置
fileTimeHandler.suffix = "%Y%m%d.log"
fileTimeHandler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO)
app.logger.addHandler(fileTimeHandler)

# 获取表情探测器，探测器整个进程应只初始化一次
model = load_model(config.EMOTION_MODEL_PATH, compile=False)
# web处理异步线程引起的玄学bug，预测必须执行此条才能正确运行，同样，训练得执行model._make_train_function()才可以
model._make_predict_function()
emotion_detector = EmotionDetector(labels=get_labels('fer2013'), face_detector=DlibFaceDetector(),
                                   emotion_classifier=model, emotion_offsets=(10, 10))

# swagger配置
swagger_config = Swagger.DEFAULT_CONFIG
swagger_config['title'] = config.SWAGGER_TITLE  # 配置大标题
swagger_config['description'] = config.SWAGGER_DESC  # 配置公共描述内容
swagger_config['host'] = config.SWAGGER_HOST  # 请求域名

# swagger_config['swagger_ui_bundle_js'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui-bundle.js'
# swagger_config['swagger_ui_standalone_preset_js'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui-standalone-preset.js'
# swagger_config['jquery_js'] = '//unpkg.com/jquery@2.2.4/dist/jquery.min.js'
# swagger_config['swagger_ui_css'] = '//unpkg.com/swagger-ui-dist@3/swagger-ui.css'
Swagger(app, config=swagger_config)


class MyResponse(Response):
    @classmethod
    def force_type(cls, response, environ=None):
        if isinstance(response, (list, dict)):
            response = jsonify(response)
            response.status_code = 200
        return super(Response, cls).force_type(response, environ)


class MyFlask(Flask):
    response_class = MyResponse


@app.errorhandler(APIException)
def handle_api_exception(error):
    from flask import jsonify
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/')
def hello_world():
    return 'Hello World!'


# @app.route('/getEmotionStreamJson/', methods=['GET'])
# def get_emotion_stream_json():
#     """
#     获取表情流
#     ---
#     tags:
#       - 调用模型函数接口
#     description:
#         获取表情流接口，json格式
#     parameters:
#       - name: video_path
#         in: query
#         type: string
#         required: true
#         description: 视频路径
#       - name: interval_ms
#         in: query
#         type: integer
#         description: 帧采样间隔
#       - name: cpu_num
#         in: query
#         type: integer
#         description: 调用cpu个数
#     responses:
#       200:
#         description: 成功返回结果
#       406:
#         description: 注册有误，参数有误等
#
#     """
#     fun = 'get_emotion_stream_json()--->'
#
#     video_path = str(request.args.get('video_path'))
#     interval_ms = int(request.args.get('interval_ms'))
#     cpu_num = int(request.args.get('cpu_num'))
#     p = psutil.Process()
#     p.cpu_affinity(range(0, cpu_num))
#     app.logger.info(config.API_INFO_FORMAT_GET.format(fun, video_path, interval_ms))
#     if os.path.exists(video_path) is False:
#         app.logger.warning(
#             config.API_WARNING_FORMAT_GET.format(fun, request.url, config.ERROR_MSG[config.FILE_DOESNT_EXIST]))
#         raise APIException(406, config.FILE_DOESNT_EXIST)
#     if interval_ms <= 0:
#         app.logger.warning(
#             config.API_WARNING_FORMAT_GET.format(fun, request.url, config.ERROR_MSG[config.PARAMETER_ERROR]))
#         raise APIException(406, config.PARAMETER_ERROR)
#     try:
#         emotion_stream_json = detect_functions.get_emotion_stream_json(video_path, emotion_detector, interval_ms)
#     except APIException as aex:
#         app.logger.warning(config.API_WARNING_FORMAT_GET.format(fun, request.url, config.ERROR_MSG[aex.error_code]))
#         raise aex
#     except Exception as ex:
#         app.logger.error(config.API_ERROR_FORMAT_GET.format(fun, request.url, ex.args))
#         raise APIException(500, config.FUNCTION_RUN_ERROR, ex.args)
#     # if emotion_stream_json == '[]':
#     #     app.logger.warning(
#     #         config.API_WARNING_FORMAT_GET.format(fun, request.url, config.ERROR_MSG[config.FUNCTION_RETURN_VOID]))
#     #     raise APIException(204, config.FUNCTION_RETURN_VOID)
#     return emotion_stream_json


@app.route('/getEmotionStreamJson/', methods=['POST'])
def get_emotion_stream_json():
    """
    获取表情流
    ---
    tags:
      - 调用模型函数接口
    description:
        获取表情流接口，json格式
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: 表情流请求参数
          properties:
            videoPath:
              type: string
            intervalMs:
              type: integer
            beginMs:
              type: integer
            endMs:
              type: integer
            modelNum:
              type: integer
            cpuNum:
              type: integer
    responses:
      200:
        description: 成功返回结果
      406:
        description: 注册有误，参数有误等

    """
    fun = 'get_emotion_stream_json()--->'
    data = request.get_data()
    parameters = json.loads(data)
    video_path = parameters['videoPath']
    interval_ms = parameters['intervalMs']
    begin_ms = parameters['beginMs']
    end_ms = parameters['endMs']
    model_num = parameters['modelNum']
    cpu_num = parameters['cpuNum']
    p = psutil.Process()
    p.cpu_affinity(range(0, cpu_num))
    app.logger.info(config.API_INFO_FORMAT_POST.format(fun, request.url, str(data)))
    if os.path.exists(video_path) is False:
        app.logger.warning(
            config.API_WARNING_FORMAT_POST.format(fun, request.url, str(data), config.ERROR_MSG[config.FILE_DOESNT_EXIST]))
        raise APIException(406, config.FILE_DOESNT_EXIST)
    if interval_ms <= 0:
        app.logger.warning(
            config.API_WARNING_FORMAT_POST.format(fun, request.url, str(data), config.ERROR_MSG[config.PARAMETER_ERROR]))
        raise APIException(406, config.PARAMETER_ERROR)
    try:
        emotion_stream_json = detect_functions.get_tiny_emotion_stream_cut_json(video_path, emotion_detector, interval_ms, begin_ms, end_ms)
    except APIException as aex:
        app.logger.warning(config.API_WARNING_FORMAT_POST.format(fun, request.url, str(data), config.ERROR_MSG[aex.error_code]))
        raise aex
    except Exception as ex:
        app.logger.error(config.API_ERROR_FORMAT_POST.format(fun, request.url, str(data), ex.args))
        raise APIException(500, config.FUNCTION_RUN_ERROR, ex.args)
    return emotion_stream_json


if __name__ == '__main__':
    app.run()
