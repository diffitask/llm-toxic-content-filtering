# LLM Toxic Content Filtering

# Project Description
Project goals, values, target audience, uniqueness, realization details -- see all these aspects in [our presentation](https://docs.google.com/presentation/d/1Bbjp2RH65IX8I-KE-Y8YoMusPbc44nX22xBPZQnRlvQ/edit?usp=sharing).

# Project Setup
First you need to configure `.env` file. Rename `.public_env` to `.env` and specify your credentials (LANGCHAIN_API_KEY, HF_TOKEN).

Next you need to create `artifacts` folder and download model weights.

###### peft_mistral
- Unpack `.zip` file from the [link](https://drive.google.com/drive/folders/1zIKR60AxLkSYbr3TMDC6b9nfVFje4spf?usp=sharing) and place it in `artifacts/peft_mistral`

Start the application:
- locally
    ```
    uvicorn src.main:app
    ```
- in Docker:
    ```
    docker compose build
    docker compose up
    ```

## API Usage
You can check how to use our API in `examples/demo_with_local_llm.ipynb` file

## LangSmith
Our API is traceable via LangSmith if you specify variables in `.env` file:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<YOUR_KEY>
```
