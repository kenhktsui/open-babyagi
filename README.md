<h1 align="center">
 open-babyagi

</h1>

## Motivation 
This is a re-implementation of idea of [babyagi](https://github.com/yoheinakajima/babyagi), with open source in mind:
- any huggingface [transformers](https://github.com/huggingface/transformers) model, which does count GPU hours instead of token usage.
- [Qdrant](https://github.com/qdrant/qdrant) vector DB, which is a fast and scalable vector DB written in rust, free of charge for self host.


This is ideal for people:
- who have trained LLM, and want to test the LLM capability of acting as an autonomous agent
- who knows some well-trained assistant


## Example

#### running locally
```shell
# vectorDB
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant

# install requirements
pip install -r requirements.txt

# running models
python huggingface_app.py stabilityai/stablelm-tuned-alpha-3b --user_prefix "<|USER|>" --assistant_prefix "<|ASSISTANT|>" --port 5000

# running agents
python main.py config.yaml
```

#### running with docker
```shell
docker compose up
```