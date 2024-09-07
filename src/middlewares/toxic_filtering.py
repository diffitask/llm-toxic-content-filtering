from starlette.middleware.base import BaseHTTPMiddleware
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
import random

class ToxicFiltering(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    @traceable
    async def toxic_classification(self, input: str):
        # TODO ML classification
        classification_result = 0
        return classification_result

    async def dispatch(self, request, call_next):
        encoded_request_body = await request.body()
        request_body = encoded_request_body.decode()
        print(request_body)
        if request_body != "":
            res = await self.toxic_classification(request_body)
        response = await call_next(request)
        # async for chunk in response.body_iterator:
        #     print(chunk)
        print(response)
        return response
