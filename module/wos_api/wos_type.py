from typing import Any, List
import json


class WOSAPIFeedback:
    @staticmethod
    def from_json(obj):
        return WOSAPIFeedback(obj["progress"], obj["status"])

    def __init__(self, progress: float, status: str):
        self.progress = progress
        self.status = status

    def to_json(self):
        return {"progress": self.progress, "status": self.status}


class WOSAPIResult:
    @staticmethod
    def from_json(obj):
        result = WOSAPIResult("", None)
        if "result" in obj:
            result.result = obj["result"]
        if "error" in obj:
            result.error = obj["error"]

        return result

    def __init__(self, error: str, result: Any):
        self.error = error
        self.result = result

    def to_json(self):
        return {"error": self.error, "result": self.result}


class WOSAPIRequest:
    @staticmethod
    def from_json(obj):
        return WOSAPIRequest(obj["action"], obj["arguments"])

    def __init__(self, action: str, args: Any):
        self.action = action
        self.arguments = args

    def to_json(self):
        return {"action": self.action, "arguments": self.arguments}


class WOSPublishMessage:
    @staticmethod
    def from_json(obj):
        return WOSPublishMessage(obj["resource"], obj["topic"], obj["message"])

    def __init__(self, resource: str, topic: str, message: Any):
        self.resource = resource
        self.topic = topic
        self.message = message

    def to_json(self):
        return {"resource": self.resource, "topic": self.topic, "message": self.message}


class WOSServiceInfo:

    @staticmethod
    def from_json(obj):
        return WOSServiceInfo(obj["topics"], obj["requests"], obj["actions"])

    def __init__(
        self, topics: List[str], requests: List[str], actions: List[str]
    ) -> None:
        self.topics = topics
        self.requests = requests
        self.actions = actions

    def to_json(self):
        return {
            "topics": self.topics,
            "requests": self.requests,
            "actions": self.actions,
        }


class WOSAPIMessage:
    @staticmethod
    def from_json(obj: Any):
        return WOSAPIMessage(obj["id"], obj["op"], obj["resource"], obj["data"])

    def __init__(
        self,
        id: str,
        op: str,
        resource: str,
        data={},
    ) -> None:
        self.op = op
        self.id = id
        self.data = data
        self.resource = resource

    def to_json(self):
        data = self.data
        if "to_json" in dir(self.data):
            data = self.data.to_json()

        return json.dumps(
            {"id": self.id, "op": self.op, "resource": self.resource, "data": data}
        )

    def get_publish_message(self):
        return WOSPublishMessage.from_json(self.data)

    def get_api_request(self):
        return WOSAPIRequest.from_json(self.data)

    def get_data_string(self) -> str:
        return self.data

    def get_api_feedback(self):
        return WOSAPIFeedback.from_json(self.data)

    def get_api_result(self):
        return WOSAPIResult.from_json(self.data)

    def get_service_info(self):
        return WOSServiceInfo.from_json(self.data)
