import app
import config


class APIException(Exception):
    # 自己定义了一个 error_code，作为更细颗粒度的错误代码
    def __init__(self, status_code=None, error_code=None, error=None, payload=None):
        Exception.__init__(self)
        self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
        self.error = error

    # 构造要返回的错误代码和错误信息的 dict
    def to_dict(self):
        result = {
            "message": config.ERROR_MSG[self.error_code],
            "error": self.error
        }
        return result
