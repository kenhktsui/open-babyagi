FROM python:3.8

WORKDIR /model
COPY requirements.txt /model
RUN pip install --no-cache -r requirements.txt

COPY huggingface_app.py /model


CMD python huggingface_app.py stabilityai/stablelm-tuned-alpha-3b --user_prefix "<|USER|>" --assistant_prefix "<|ASSISTANT|>" --port 5000
