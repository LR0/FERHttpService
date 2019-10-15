from logging.handlers import TimedRotatingFileHandler

SWAGGER_TITLE = "FERHttpService"  # 配置大标题
SWAGGER_DESC = ""  # 配置公共描述内容
SWAGGER_HOST = ""  # 请求域名
# 默认模型地址
EMOTION_MODEL_PATH = './core/trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5'

# 日志配置
API_INFO_FORMAT_GET = '{0} url:{1}'
API_WARNING_FORMAT_GET = '{0} url:{1} msg:{2}'
API_ERROR_FORMAT_GET = '{0} url:{1} error:{2}'
# 返回错误码，1000~1999日志级别为warn(只影响该次请求)，2000~2999为error(可能影响后续功能的运行)
FILE_DOESNT_EXIST = 1000
FILE_FORMAT_ERROR = 1001
PARAMETER_ERROR = 1002
FUNCTION_RETURN_VOID = 1003  # 分析结果为空
FUNCTION_VIDEO_UNDONE = 1004  # 视频未分析完
FUNCTION_RUN_ERROR = 2000
ERROR_MSG = {FILE_DOESNT_EXIST: '文件不存在', FILE_FORMAT_ERROR: '文件格式错误', FUNCTION_RUN_ERROR: '函数运行出错',
             FUNCTION_RETURN_VOID: '分析结果为空', FUNCTION_VIDEO_UNDONE: '视频未分析完', PARAMETER_ERROR: '参数有错'}
