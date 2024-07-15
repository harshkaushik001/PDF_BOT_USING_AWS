FROM python:3.11
EXPOSE 8080
WORKDIR /app

# Add environment variables here
ENV AWS_ACCESS_KEY_ID "YOUR AWS_ACCESS_KEY_ID"

ENV AWS_SECRET_ACCESS_KEY "YOUR AWS_SECRET_ACCESS_KEY"

ENV AWS_DEFAULT_REGION us-west-2

COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . ./
ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
