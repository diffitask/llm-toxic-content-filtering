# LLM Toxic Content Filtering

# Project Setup
First you need to configure `.env` file. Rename `.public_env` to `.env` and specify your credentials (LANGCHAIN_API_KEY, HF_TOKEN).

Now install all the dependencies in your virutal environment (first create one):
```
pip install -r requirements.txt
```

Next you need to create `artifacts` folder and download model weights:

#### peft_mistral
- Unpack `.zip` file from the [link](https://drive.google.com/drive/folders/1zIKR60AxLkSYbr3TMDC6b9nfVFje4spf?usp=sharing) and place it in `artifacts/peft_mistral`
- Go to `download-models/download_mistral.ipynb` notebook and run it (directories with the needed files should appear in `artifacts`)

## Run the application
Start the application:
- locally
    ```
    uvicorn src.main:app
    ```
- in Docker (currently not working):
    ```
    docker compose build
    docker compose up
    ```
This may take some time because it will be loading checkpoint shards

## API Usage
You can check how to use our API in `examples/demo_with_local_llm.ipynb` file

## LangSmith
Our API is traceable via LangSmith if you specify variables in `.env` file:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<YOUR_KEY>
```