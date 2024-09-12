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

classifier = RemoteRunnable("http://localhost:8080/classifiers/mistral/")

classifier.invoke('привет')
```
```
Output: {'predicted_class': 'vanilla_harmful'}
```

```python
# requests example

import requests

response = requests.post(
    "http://localhost:8080/classifiers/mistral/invoke",
    json={'input': 'привет'}
)
response.json()
```