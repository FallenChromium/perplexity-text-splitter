# Perplexity-based text chunker

## Prerequisites

- Python 3.10+
- PyTorch 2.0+

## Installation
```sh
copy .env.example .env
# pytorch with cuda support if you have an Nvidia GPU and CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

docker compose up
```


todo: bootstrap script to сreate an .env file, execute "CREATE EXTENSION vector;" on postgres and create an access key on minio (and install nltk models)
```py
from nltk import download
download('punkt_tab')
```