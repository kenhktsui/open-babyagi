FROM python:3.8

WORKDIR /autonomous_agent
COPY requirements.txt /autonomous_agent
RUN pip install --no-cache -r requirements.txt

COPY autonomous_agent /autonomous_agent/autonomous_agent
COPY main.py /autonomous_agent
COPY config.yaml /autonomous_agent


CMD python main.py config.yaml