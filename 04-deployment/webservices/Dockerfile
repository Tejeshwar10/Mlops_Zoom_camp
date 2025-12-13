FROM python:3.9-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY Pipfile Pipfile.lock /

RUN pipenv install --system --deploy

# Copy application files
COPY predict.py .
COPY lin_reg.bin .

EXPOSE 9696


ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]