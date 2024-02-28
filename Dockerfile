FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    ffmpeg \
    graphviz \
    graphviz-dev \
    libgraphviz-dev \
    pkg-config  \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/MatteoFasulo/BayesianClassifier . \
    && pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
