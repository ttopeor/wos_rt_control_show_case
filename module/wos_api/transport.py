import json
import ssl
import threading
from .wos_type import WOSAPIMessage
from websocket._app import WebSocketApp
from typing import Callable
from .logger import logger


class WOSTransport:
    """Abstract Transport Class"""

    def send(self, message: WOSAPIMessage):
        pass

    def bind(self, func):
        pass

    def run(self):
        pass

    def stop(self):
        pass

    def wait_connected(self) -> bool:
        return False


class WSTransport(WOSTransport):
    """websocket transport layer using threading model"""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.conn = WebSocketApp(
            self.endpoint,
            on_open=self.on_open,
            on_message=self._on_message,
            on_close=self.on_close,
        )
        self.connected = False
        self.connectionEvent = threading.Event()
        self.cb = None

    def send(self, message: WOSAPIMessage):
        if self.connected:
            data = message.to_json()
            logger.debug("send message: %s", data)
            self.conn.send(data)
        else:
            logger.warn("failed to send: websocket conn not opened")

    def bind(self, cb: Callable[[WOSAPIMessage], None]):
        self.cb = cb

    def on_open(self, ws):
        self.conn = ws
        self.connected = True
        self.connectionEvent.set()
        logger.info("websocket connected")

    def _on_message(self, ws, msg):
        logger.debug("receive msg: %s", msg)
        api = WOSAPIMessage.from_json(json.loads(msg))
        if self.cb != None:
            self.cb(api)

    def on_error(self, ws, error):
        logger.warn("websocket error %s", error)
        pass

    def on_close(self, ws, status_code, close_msg):
        self.connected = False
        self.connectionEvent.set()
        logger.info("websocket closed")

    def run(self):
        logger.info("websocket connecting: %s", self.endpoint)
        self.conn.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    def stop(self):
        self.conn.close()

    def wait_connected(self) -> bool:
        self.connectionEvent.wait()
        return self.connected
