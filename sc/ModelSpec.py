from pydantic import BaseModel

# {"payloadId": payloadId,
# "orgid": orgid,
# "requestKey": requestKey,
# "requestId": requestId,
# "entity": entity,
# "refdoc": refdoc,
# "payloadconfig": payloadConfig,
# "X-TIKOS-MODEL": llmmodel}

class ModelSpec(BaseModel):
    payloadId: str = ""
    orgid: str = ""
    requestKey: str = ""
    requestId: str = ""
    entity: str = ""
    refdoc: str = ""
    payloadconfig: str = ""
    X_TIKOS_MODEL: int = 1