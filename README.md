# LLM Toxic Content Filtering

# Project Setup
```
docker compose build
docker compose up
```

## API Usage
Our classifiers can be called like `RemoteRunnable` or using `requests` module

```python
# RemoteRunnable example

from langserve import RemoteRunnable

baseline = RemoteRunnable("http://localhost:8080/baseline/")

baseline.invoke({'text': 'привет'})
```
```
Output: {'predicted_class': 'vanilla_harmful'}
```

```python
# requests example

import requests

response = requests.post(
    "http://localhost:8080/baseline/invoke",
    json={'input': {'text': 'привет'}}
)
response.json()
```

## Playground for models
Our API also provides playground endpoints to interact with models using simple UI - http://localhost:8080/baseline/invoke/playground