FROM python:3.11 AS builder

WORKDIR /webapp

COPY . . 

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]