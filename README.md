# Ollama Python Library

The Ollama Python library provides the easiest way to integrate Python 3.8+ projects with [Ollama](https://github.com/ollama/ollama).

## Prerequisites

- [Ollama](https://ollama.com/download) should be installed and running
- Pull a model to use with the library: `ollama pull <model>` e.g. `ollama pull gemma3`
  - See [Ollama.com](https://ollama.com/search) for more information on the models available.

## Install

```sh
pip install ollama
```

## Usage

```python
from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='gemma3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
# or access fields directly from the response object
print(response.message.content)
```

See [_types.py](ollama/_types.py) for more information on the response types.

## Streaming responses

Response streaming can be enabled by setting `stream=True`.

```python
from ollama import chat

stream = chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
```

## Cloud Models

Run larger models by offloading to Ollama’s cloud while keeping your local workflow.

- Supported models: `deepseek-v3.1:671b-cloud`, `gpt-oss:20b-cloud`, `gpt-oss:120b-cloud`, `kimi-k2:1t-cloud`, `qwen3-coder:480b-cloud`, `kimi-k2-thinking` See [Ollama Models - Cloud](https://ollama.com/search?c=cloud) for more information

### Run via local Ollama

1) Sign in (one-time):

```
ollama signin
```

2) Pull a cloud model:

```
ollama pull gpt-oss:120b-cloud
```

3) Make a request:

```python
from ollama import Client

client = Client()

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b-cloud', messages=messages, stream=True):
  print(part.message.content, end='', flush=True)
```

### Cloud API (ollama.com)

Access cloud models directly by pointing the client at `https://ollama.com`.

1) Create an API key from [ollama.com](https://ollama.com/settings/keys) , then set:

```
export OLLAMA_API_KEY=your_api_key
```

2) (Optional) List models available via the API:

```
curl https://ollama.com/api/tags
```

3) Generate a response via the cloud API:

```python
import os
from ollama import Client

client = Client(
    host='https://ollama.com',
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

messages = [
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
]

for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
  print(part.message.content, end='', flush=True)
```

## Custom client
A custom client can be created by instantiating `Client` or `AsyncClient` from `ollama`.

All extra keyword arguments are passed into the [`httpx.Client`](https://www.python-httpx.org/api/#client).

```python
from ollama import Client
client = Client(
  host='http://localhost:11434',
  headers={'x-some-header': 'some-value'}
)
response = client.chat(model='gemma3', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
```

## Async client

The `AsyncClient` class is used to make asynchronous requests. It can be configured with the same fields as the `Client` class.

```python
import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  response = await AsyncClient().chat(model='gemma3', messages=[message])

asyncio.run(chat())
```

Setting `stream=True` modifies functions to return a Python asynchronous generator:

```python
import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  async for part in await AsyncClient().chat(model='gemma3', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)

asyncio.run(chat())
```

## API

The Ollama Python library's API is designed around the [Ollama REST API](https://github.com/ollama/ollama/blob/main/docs/api.md)

### Chat

```python
ollama.chat(model='gemma3', messages=[{'role': 'user', 'content': 'Why is the sky blue?'}])
```

### Generate

```python
ollama.generate(model='gemma3', prompt='Why is the sky blue?')
```

### List

```python
ollama.list()
```

### Show

```python
ollama.show('gemma3')
```

### Create

```python
ollama.create(model='example', from_='gemma3', system="You are Mario from Super Mario Bros.")
```

### Copy

```python
ollama.copy('gemma3', 'user/gemma3')
```

### Delete

```python
ollama.delete('gemma3')
```

### Pull

```python
ollama.pull('gemma3')
```

### Push

```python
ollama.push('user/gemma3')
```

### Embed

```python
ollama.embed(model='gemma3', input='The sky is blue because of rayleigh scattering')
```

### Embed (batch)

```python
ollama.embed(model='gemma3', input=['The sky is blue because of rayleigh scattering', 'Grass is green because of chlorophyll'])
```

### Ps

```python
ollama.ps()
```

### Save

Export a locally-available model to a portable `.tar.gz` archive:

```python
ollama.save('llama3.2:latest', 'llama3.2-latest.tar.gz')
```

### Load

Import a model from an archive previously created by `save()`:

```python
ollama.load('llama3.2:latest', 'llama3.2-latest.tar.gz')
```

## Airgapped transfer

Models can be exported on a network-connected machine, physically transferred to an airgapped (network-disconnected) environment, and imported into a locally-running Ollama instance — no internet access required on the destination.

### Archive format

The `.tar.gz` archive produced by `save()` contains:

| Entry | Description |
|---|---|
| `meta.json` | Model name, tag, and export timestamp |
| `manifest.json` | Ollama manifest (config digest, layer digests, media types) |
| `blobs/<digest>` | All model blobs (weights, tokeniser, config) |

Every blob is verified against its SHA-256 digest on import.

### Workflow

**1. Export on the connected machine**

```python
import ollama

# Pull the model first if not already present
# ollama.pull('llama3.2:latest')

ollama.save('llama3.2:latest', 'llama3.2-latest.tar.gz')
```

**2. Transfer the archive**

Copy `llama3.2-latest.tar.gz` to the airgapped machine via USB drive, SCP, or any out-of-band mechanism available in your environment.

**3. Import on the airgapped machine**

```python
import ollama

ollama.load('llama3.2:latest', 'llama3.2-latest.tar.gz')

# The model is immediately available
response = ollama.generate(model='llama3.2:latest', prompt='Why is the sky blue?')
print(response.response)
```

### Async usage

```python
import asyncio
from ollama import AsyncClient

async def transfer():
    client = AsyncClient()
    await client.save('llama3.2:latest', 'llama3.2-latest.tar.gz')
    await client.load('llama3.2:latest', 'llama3.2-latest.tar.gz')

asyncio.run(transfer())
```

### Custom models directory

If Ollama stores its models in a non-default location (e.g. set via the `OLLAMA_MODELS` environment variable), pass `models_dir` explicitly:

```python
ollama.save('llama3.2:latest', 'llama3.2-latest.tar.gz', models_dir='/mnt/data/ollama/models')
ollama.load('llama3.2:latest', 'llama3.2-latest.tar.gz', models_dir='/mnt/data/ollama/models')
```

The models directory is resolved in this priority order:
1. The `models_dir` argument
2. The `OLLAMA_MODELS` environment variable
3. The running Ollama process's home directory (detected via `/proc`)
4. Well-known service paths (`/usr/share/ollama/.ollama/models`, `/var/lib/ollama/.ollama/models`)
5. `~/.ollama/models`

### Idempotent import

`load()` only copies blobs that are not already present in the destination models directory. Re-importing the same archive (or a different archive that shares blobs, e.g. a fine-tune of the same base model) is safe and will not overwrite existing files.

### Disk space planning

The archive size is roughly equal to the total size of the model's blobs (weights + tokeniser + config), compressed with gzip. Allow for the following temporary space during each operation:

| Operation | Space required |
|---|---|
| `save()` | 1× model size (the archive itself) |
| `load()` | 1× model size (temporary extraction) + 1× model size (blobs in models dir) |

To check the size of a model before exporting:

```python
import ollama

info = ollama.show('llama3.2:latest')
total = sum(d.size or 0 for d in (info.details and []) or [])
# Or simply inspect the blobs directory beforehand:
# ls -lh ~/.ollama/models/blobs/
```

### Error handling

Both `save()` and `load()` raise standard Python exceptions; no special Ollama exception types are used for these operations.

```python
import ollama

# Export
try:
    ollama.save('llama3.2:latest', 'llama3.2-latest.tar.gz')
except FileNotFoundError as exc:
    # Model has not been pulled, or a blob is missing from the models directory
    print(f'Export failed — model not found locally: {exc}')
    # Pull the model first: ollama.pull('llama3.2:latest')

# Import
try:
    ollama.load('llama3.2:latest', 'llama3.2-latest.tar.gz')
except FileNotFoundError as exc:
    # Archive path does not exist
    print(f'Import failed — archive not found: {exc}')
except ValueError as exc:
    # Archive is missing required entries, contains unsafe paths or symlinks,
    # or a blob's SHA-256 digest does not match its filename (corrupt archive)
    print(f'Import failed — archive invalid or corrupt: {exc}')
```

### Security

Every archive member is validated before extraction:

- Absolute paths are rejected.
- Path-traversal entries (e.g. `../../etc/passwd`) are rejected.
- Symlinks and hard links are rejected.
- Each blob's SHA-256 digest is verified against its filename after extraction. A mismatch raises `ValueError` before any blob is written to the models directory.

### CLI helper

The `examples/airgap-transfer.py` script provides a command-line interface:

```sh
# Connected machine — export
python examples/airgap-transfer.py export llama3.2:latest llama3.2-latest.tar.gz

# Airgapped machine — import
python examples/airgap-transfer.py import llama3.2:latest llama3.2-latest.tar.gz

# Override the models directory
python examples/airgap-transfer.py export llama3.2:latest llama3.2-latest.tar.gz --models-dir /mnt/data/ollama/models
```

## Errors

Errors are raised if requests return an error status or if an error is detected while streaming.

```python
model = 'does-not-yet-exist'

try:
  ollama.chat(model)
except ollama.ResponseError as e:
  print('Error:', e.error)
  if e.status_code == 404:
    ollama.pull(model)
```
