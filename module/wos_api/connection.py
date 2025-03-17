from .client import WOSClient
from .transport import WSTransport


def CreateWSClient(endpoint) -> WOSClient:
    """
    Factory function to create a wos client using websocket transports

    Parameters:
      - Endpoint: The endpoint to connect. default: WOS_ENDPOINT env variable or localhost:15117

    Return:
      - WOSClient object
    """
    wsEndpoint = "ws://" + endpoint + "/api/ws"
    transport = WSTransport(wsEndpoint)
    return WOSClient(transport)
