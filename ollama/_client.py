import contextlib
import io
import ipaddress
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.parse
from datetime import datetime, timezone
from hashlib import sha256
from os import PathLike
from pathlib import Path
from typing import (
  Any,
  Callable,
  Dict,
  List,
  Literal,
  Mapping,
  Optional,
  Sequence,
  Type,
  TypeVar,
  Union,
  overload,
)

import anyio
from pydantic.json_schema import JsonSchemaValue

from ollama._utils import convert_function_to_tool

if sys.version_info < (3, 9):
  from typing import AsyncIterator, Iterator
else:
  from collections.abc import AsyncIterator, Iterator

from importlib import metadata

try:
  __version__ = metadata.version('ollama')
except metadata.PackageNotFoundError:
  __version__ = '0.0.0'

import httpx

from ollama._types import (
  ChatRequest,
  ChatResponse,
  CopyRequest,
  CreateRequest,
  DeleteRequest,
  EmbeddingsRequest,
  EmbeddingsResponse,
  EmbedRequest,
  EmbedResponse,
  GenerateRequest,
  GenerateResponse,
  Image,
  ListResponse,
  Message,
  Options,
  ProcessResponse,
  ProgressResponse,
  PullRequest,
  PushRequest,
  ResponseError,
  ShowRequest,
  ShowResponse,
  StatusResponse,
  Tool,
  WebFetchRequest,
  WebFetchResponse,
  WebSearchRequest,
  WebSearchResponse,
)

T = TypeVar('T')


class BaseClient(contextlib.AbstractContextManager, contextlib.AbstractAsyncContextManager):
  def __init__(
    self,
    client,
    host: Optional[str] = None,
    *,
    follow_redirects: bool = True,
    timeout: Any = None,
    headers: Optional[Mapping[str, str]] = None,
    **kwargs,
  ) -> None:
    """
    Creates a httpx client. Default parameters are the same as those defined in httpx
    except for the following:
    - `follow_redirects`: True
    - `timeout`: None
    `kwargs` are passed to the httpx client.
    """

    headers = {
      k.lower(): v
      for k, v in {
        **(headers or {}),
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': f'ollama-python/{__version__} ({platform.machine()} {platform.system().lower()}) Python/{platform.python_version()}',
      }.items()
      if v is not None
    }
    api_key = os.getenv('OLLAMA_API_KEY', None)
    if not headers.get('authorization') and api_key:
      headers['authorization'] = f'Bearer {api_key}'

    self._client = client(
      base_url=_parse_host(host or os.getenv('OLLAMA_HOST')),
      follow_redirects=follow_redirects,
      timeout=timeout,
      headers=headers,
      **kwargs,
    )

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()


CONNECTION_ERROR_MESSAGE = 'Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download'


class Client(BaseClient):
  def __init__(self, host: Optional[str] = None, **kwargs) -> None:
    super().__init__(httpx.Client, host, **kwargs)

  def close(self):
    self._client.close()

  def _request_raw(self, *args, **kwargs):
    try:
      r = self._client.request(*args, **kwargs)
      r.raise_for_status()
      return r
    except httpx.HTTPStatusError as e:
      raise ResponseError(e.response.text, e.response.status_code) from None
    except httpx.ConnectError:
      raise ConnectionError(CONNECTION_ERROR_MESSAGE) from None

  @overload
  def _request(
    self,
    cls: Type[T],
    *args,
    stream: Literal[False] = False,
    **kwargs,
  ) -> T: ...

  @overload
  def _request(
    self,
    cls: Type[T],
    *args,
    stream: Literal[True] = True,
    **kwargs,
  ) -> Iterator[T]: ...

  @overload
  def _request(
    self,
    cls: Type[T],
    *args,
    stream: bool = False,
    **kwargs,
  ) -> Union[T, Iterator[T]]: ...

  def _request(
    self,
    cls: Type[T],
    *args,
    stream: bool = False,
    **kwargs,
  ) -> Union[T, Iterator[T]]:
    if stream:

      def inner():
        with self._client.stream(*args, **kwargs) as r:
          try:
            r.raise_for_status()
          except httpx.HTTPStatusError as e:
            e.response.read()
            raise ResponseError(e.response.text, e.response.status_code) from None

          for line in r.iter_lines():
            part = json.loads(line)
            if err := part.get('error'):
              raise ResponseError(err)
            yield cls(**part)

      return inner()

    return cls(**self._request_raw(*args, **kwargs).json())

  @overload
  def generate(
    self,
    model: str = '',
    prompt: str = '',
    suffix: str = '',
    *,
    system: str = '',
    template: str = '',
    context: Optional[Sequence[int]] = None,
    stream: Literal[False] = False,
    think: Optional[bool] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    raw: bool = False,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    images: Optional[Sequence[Union[str, bytes, Image]]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
  ) -> GenerateResponse: ...

  @overload
  def generate(
    self,
    model: str = '',
    prompt: str = '',
    suffix: str = '',
    *,
    system: str = '',
    template: str = '',
    context: Optional[Sequence[int]] = None,
    stream: Literal[True] = True,
    think: Optional[bool] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    raw: bool = False,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    images: Optional[Sequence[Union[str, bytes, Image]]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
  ) -> Iterator[GenerateResponse]: ...

  def generate(
    self,
    model: str = '',
    prompt: Optional[str] = None,
    suffix: Optional[str] = None,
    *,
    system: Optional[str] = None,
    template: Optional[str] = None,
    context: Optional[Sequence[int]] = None,
    stream: bool = False,
    think: Optional[bool] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    raw: Optional[bool] = None,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    images: Optional[Sequence[Union[str, bytes, Image]]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
  ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
    """
    Create a response using the requested model.

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `GenerateResponse` if `stream` is `False`, otherwise returns a `GenerateResponse` generator.
    """

    return self._request(
      GenerateResponse,
      'POST',
      '/api/generate',
      json=GenerateRequest(
        model=model,
        prompt=prompt,
        suffix=suffix,
        system=system,
        template=template,
        context=context,
        stream=stream,
        think=think,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        raw=raw,
        format=format,
        images=list(_copy_images(images)) if images else None,
        options=options,
        keep_alive=keep_alive,
        width=width,
        height=height,
        steps=steps,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  @overload
  def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
    stream: Literal[False] = False,
    think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> ChatResponse: ...

  @overload
  def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
    stream: Literal[True] = True,
    think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Iterator[ChatResponse]: ...

  def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
    stream: bool = False,
    think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Union[ChatResponse, Iterator[ChatResponse]]:
    """
    Create a chat response using the requested model.

    Args:
      tools:
        A JSON schema as a dict, an Ollama Tool or a Python Function.
        Python functions need to follow Google style docstrings to be converted to an Ollama Tool.
        For more information, see: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
      stream: Whether to stream the response.
      format: The format of the response.

    Example:
      def add_two_numbers(a: int, b: int) -> int:
        '''
        Add two numbers together.

        Args:
          a: First number to add
          b: Second number to add

        Returns:
          int: The sum of a and b
        '''
        return a + b

      client.chat(model='llama3.2', tools=[add_two_numbers], messages=[...])

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ChatResponse` if `stream` is `False`, otherwise returns a `ChatResponse` generator.
    """
    return self._request(
      ChatResponse,
      'POST',
      '/api/chat',
      json=ChatRequest(
        model=model,
        messages=list(_copy_messages(messages)),
        tools=list(_copy_tools(tools)),
        stream=stream,
        think=think,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        format=format,
        options=options,
        keep_alive=keep_alive,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  def embed(
    self,
    model: str = '',
    input: Union[str, Sequence[str]] = '',
    truncate: Optional[bool] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
    dimensions: Optional[int] = None,
  ) -> EmbedResponse:
    return self._request(
      EmbedResponse,
      'POST',
      '/api/embed',
      json=EmbedRequest(
        model=model,
        input=input,
        truncate=truncate,
        options=options,
        keep_alive=keep_alive,
        dimensions=dimensions,
      ).model_dump(exclude_none=True),
    )

  def embeddings(
    self,
    model: str = '',
    prompt: Optional[str] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> EmbeddingsResponse:
    """
    Deprecated in favor of `embed`.
    """
    return self._request(
      EmbeddingsResponse,
      'POST',
      '/api/embeddings',
      json=EmbeddingsRequest(
        model=model,
        prompt=prompt,
        options=options,
        keep_alive=keep_alive,
      ).model_dump(exclude_none=True),
    )

  @overload
  def pull(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[False] = False,
  ) -> ProgressResponse: ...

  @overload
  def pull(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[True] = True,
  ) -> Iterator[ProgressResponse]: ...

  def pull(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: bool = False,
  ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return self._request(
      ProgressResponse,
      'POST',
      '/api/pull',
      json=PullRequest(
        model=model,
        insecure=insecure,
        stream=stream,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  @overload
  def push(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[False] = False,
  ) -> ProgressResponse: ...

  @overload
  def push(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[True] = True,
  ) -> Iterator[ProgressResponse]: ...

  def push(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: bool = False,
  ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return self._request(
      ProgressResponse,
      'POST',
      '/api/push',
      json=PushRequest(
        model=model,
        insecure=insecure,
        stream=stream,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  @overload
  def create(
    self,
    model: str,
    quantize: Optional[str] = None,
    from_: Optional[str] = None,
    files: Optional[Dict[str, str]] = None,
    adapters: Optional[Dict[str, str]] = None,
    template: Optional[str] = None,
    license: Optional[Union[str, List[str]]] = None,
    system: Optional[str] = None,
    parameters: Optional[Union[Mapping[str, Any], Options]] = None,
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    stream: Literal[False] = False,
  ) -> ProgressResponse: ...

  @overload
  def create(
    self,
    model: str,
    quantize: Optional[str] = None,
    from_: Optional[str] = None,
    files: Optional[Dict[str, str]] = None,
    adapters: Optional[Dict[str, str]] = None,
    template: Optional[str] = None,
    license: Optional[Union[str, List[str]]] = None,
    system: Optional[str] = None,
    parameters: Optional[Union[Mapping[str, Any], Options]] = None,
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    stream: Literal[True] = True,
  ) -> Iterator[ProgressResponse]: ...

  def create(
    self,
    model: str,
    quantize: Optional[str] = None,
    from_: Optional[str] = None,
    files: Optional[Dict[str, str]] = None,
    adapters: Optional[Dict[str, str]] = None,
    template: Optional[str] = None,
    license: Optional[Union[str, List[str]]] = None,
    system: Optional[str] = None,
    parameters: Optional[Union[Mapping[str, Any], Options]] = None,
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    stream: bool = False,
  ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return self._request(
      ProgressResponse,
      'POST',
      '/api/create',
      json=CreateRequest(
        model=model,
        stream=stream,
        quantize=quantize,
        from_=from_,
        files=files,
        adapters=adapters,
        license=license,
        template=template,
        system=system,
        parameters=parameters,
        messages=messages,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  def create_blob(self, path: Union[str, Path]) -> str:
    sha256sum = sha256()
    with open(path, 'rb') as r:
      while True:
        chunk = r.read(32 * 1024)
        if not chunk:
          break
        sha256sum.update(chunk)

    digest = f'sha256:{sha256sum.hexdigest()}'

    with open(path, 'rb') as r:
      self._request_raw('POST', f'/api/blobs/{digest}', content=r)

    return digest

  def save(self, model: str, path: Union[str, Path], models_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Export a locally-available Ollama model to a portable .tar.gz archive.

    The archive can be physically transferred to an airgapped machine and
    imported into a locally-running Ollama instance using ``load()``.

    Args:
      model: The model to export, e.g. ``'llama3:latest'``.
      path: Destination file path for the ``.tar.gz`` archive.
      models_dir: Override the Ollama models directory. Defaults to the value
                  of the ``OLLAMA_MODELS`` env var, or ``~/.ollama/models``.

    Raises:
      FileNotFoundError: If the model has not been pulled locally or a blob
                         is missing from the models directory.
    """
    _save_model(model, path, models_dir)

  def load(self, model: str, path: Union[str, Path], models_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Import a model from a .tar.gz archive previously created by ``save()``
    into the local Ollama models directory.

    Ollama discovers the imported model automatically without requiring a
    restart. The model is registered under the name given by ``model``.

    Args:
      model: The model name to register, e.g. ``'llama3:latest'``.
      path: Path to the ``.tar.gz`` archive.
      models_dir: Override the Ollama models directory. Defaults to the value
                  of the ``OLLAMA_MODELS`` env var, or ``~/.ollama/models``.

    Raises:
      FileNotFoundError: If the archive path does not exist.
      ValueError: If the archive contains unsafe paths, symlinks, or blobs
                  whose SHA-256 digest does not match their filename.
    """
    _load_model(model, path, models_dir)

  def list(self) -> ListResponse:
    return self._request(
      ListResponse,
      'GET',
      '/api/tags',
    )

  def delete(self, model: str) -> StatusResponse:
    r = self._request_raw(
      'DELETE',
      '/api/delete',
      json=DeleteRequest(
        model=model,
      ).model_dump(exclude_none=True),
    )
    return StatusResponse(
      status='success' if r.status_code == 200 else 'error',
    )

  def copy(self, source: str, destination: str) -> StatusResponse:
    r = self._request_raw(
      'POST',
      '/api/copy',
      json=CopyRequest(
        source=source,
        destination=destination,
      ).model_dump(exclude_none=True),
    )
    return StatusResponse(
      status='success' if r.status_code == 200 else 'error',
    )

  def show(self, model: str) -> ShowResponse:
    return self._request(
      ShowResponse,
      'POST',
      '/api/show',
      json=ShowRequest(
        model=model,
      ).model_dump(exclude_none=True),
    )

  def ps(self) -> ProcessResponse:
    return self._request(
      ProcessResponse,
      'GET',
      '/api/ps',
    )

  def web_search(self, query: str, max_results: int = 3) -> WebSearchResponse:
    """
    Performs a web search

    Args:
      query: The query to search for
      max_results: The maximum number of results to return (default: 3)

    Returns:
      WebSearchResponse with the search results
    Raises:
      ValueError: If OLLAMA_API_KEY environment variable is not set
    """
    if not self._client.headers.get('authorization', '').startswith('Bearer '):
      raise ValueError('Authorization header with Bearer token is required for web search')

    return self._request(
      WebSearchResponse,
      'POST',
      'https://ollama.com/api/web_search',
      json=WebSearchRequest(
        query=query,
        max_results=max_results,
      ).model_dump(exclude_none=True),
    )

  def web_fetch(self, url: str) -> WebFetchResponse:
    """
    Fetches the content of a web page for the provided URL.

    Args:
      url: The URL to fetch

    Returns:
      WebFetchResponse with the fetched result
    """
    if not self._client.headers.get('authorization', '').startswith('Bearer '):
      raise ValueError('Authorization header with Bearer token is required for web fetch')

    return self._request(
      WebFetchResponse,
      'POST',
      'https://ollama.com/api/web_fetch',
      json=WebFetchRequest(
        url=url,
      ).model_dump(exclude_none=True),
    )


class AsyncClient(BaseClient):
  def __init__(self, host: Optional[str] = None, **kwargs) -> None:
    super().__init__(httpx.AsyncClient, host, **kwargs)

  async def close(self):
    await self._client.aclose()

  async def _request_raw(self, *args, **kwargs):
    try:
      r = await self._client.request(*args, **kwargs)
      r.raise_for_status()
      return r
    except httpx.HTTPStatusError as e:
      raise ResponseError(e.response.text, e.response.status_code) from None
    except httpx.ConnectError:
      raise ConnectionError(CONNECTION_ERROR_MESSAGE) from None

  @overload
  async def _request(
    self,
    cls: Type[T],
    *args,
    stream: Literal[False] = False,
    **kwargs,
  ) -> T: ...

  @overload
  async def _request(
    self,
    cls: Type[T],
    *args,
    stream: Literal[True] = True,
    **kwargs,
  ) -> AsyncIterator[T]: ...

  @overload
  async def _request(
    self,
    cls: Type[T],
    *args,
    stream: bool = False,
    **kwargs,
  ) -> Union[T, AsyncIterator[T]]: ...

  async def _request(
    self,
    cls: Type[T],
    *args,
    stream: bool = False,
    **kwargs,
  ) -> Union[T, AsyncIterator[T]]:
    if stream:

      async def inner():
        async with self._client.stream(*args, **kwargs) as r:
          try:
            r.raise_for_status()
          except httpx.HTTPStatusError as e:
            await e.response.aread()
            raise ResponseError(e.response.text, e.response.status_code) from None

          async for line in r.aiter_lines():
            part = json.loads(line)
            if err := part.get('error'):
              raise ResponseError(err)
            yield cls(**part)

      return inner()

    return cls(**(await self._request_raw(*args, **kwargs)).json())

  async def web_search(self, query: str, max_results: int = 3) -> WebSearchResponse:
    """
    Performs a web search

    Args:
      query: The query to search for
      max_results: The maximum number of results to return (default: 3)

    Returns:
      WebSearchResponse with the search results
    """
    return await self._request(
      WebSearchResponse,
      'POST',
      'https://ollama.com/api/web_search',
      json=WebSearchRequest(
        query=query,
        max_results=max_results,
      ).model_dump(exclude_none=True),
    )

  async def web_fetch(self, url: str) -> WebFetchResponse:
    """
    Fetches the content of a web page for the provided URL.

    Args:
      url: The URL to fetch

    Returns:
      WebFetchResponse with the fetched result
    """
    return await self._request(
      WebFetchResponse,
      'POST',
      'https://ollama.com/api/web_fetch',
      json=WebFetchRequest(
        url=url,
      ).model_dump(exclude_none=True),
    )

  @overload
  async def generate(
    self,
    model: str = '',
    prompt: str = '',
    suffix: str = '',
    *,
    system: str = '',
    template: str = '',
    context: Optional[Sequence[int]] = None,
    stream: Literal[False] = False,
    think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    raw: bool = False,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    images: Optional[Sequence[Union[str, bytes, Image]]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
  ) -> GenerateResponse: ...

  @overload
  async def generate(
    self,
    model: str = '',
    prompt: str = '',
    suffix: str = '',
    *,
    system: str = '',
    template: str = '',
    context: Optional[Sequence[int]] = None,
    stream: Literal[True] = True,
    think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    raw: bool = False,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    images: Optional[Sequence[Union[str, bytes, Image]]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
  ) -> AsyncIterator[GenerateResponse]: ...

  async def generate(
    self,
    model: str = '',
    prompt: Optional[str] = None,
    suffix: Optional[str] = None,
    *,
    system: Optional[str] = None,
    template: Optional[str] = None,
    context: Optional[Sequence[int]] = None,
    stream: bool = False,
    think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    raw: Optional[bool] = None,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    images: Optional[Sequence[Union[str, bytes, Image]]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    steps: Optional[int] = None,
  ) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
    """
    Create a response using the requested model.

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `GenerateResponse` if `stream` is `False`, otherwise returns an asynchronous `GenerateResponse` generator.
    """
    return await self._request(
      GenerateResponse,
      'POST',
      '/api/generate',
      json=GenerateRequest(
        model=model,
        prompt=prompt,
        suffix=suffix,
        system=system,
        template=template,
        context=context,
        stream=stream,
        think=think,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        raw=raw,
        format=format,
        images=list(_copy_images(images)) if images else None,
        options=options,
        keep_alive=keep_alive,
        width=width,
        height=height,
        steps=steps,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  @overload
  async def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
    stream: Literal[False] = False,
    think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> ChatResponse: ...

  @overload
  async def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
    stream: Literal[True] = True,
    think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> AsyncIterator[ChatResponse]: ...

  async def chat(
    self,
    model: str = '',
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None,
    stream: bool = False,
    think: Optional[Union[bool, Literal['low', 'medium', 'high']]] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    format: Optional[Union[Literal['', 'json'], JsonSchemaValue]] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
    """
    Create a chat response using the requested model.

    Args:
      tools:
        A JSON schema as a dict, an Ollama Tool or a Python Function.
        Python functions need to follow Google style docstrings to be converted to an Ollama Tool.
        For more information, see: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
      stream: Whether to stream the response.
      format: The format of the response.

    Example:
      def add_two_numbers(a: int, b: int) -> int:
        '''
        Add two numbers together.

        Args:
          a: First number to add
          b: Second number to add

        Returns:
          int: The sum of a and b
        '''
        return a + b

      await client.chat(model='llama3.2', tools=[add_two_numbers], messages=[...])

    Raises `RequestError` if a model is not provided.

    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ChatResponse` if `stream` is `False`, otherwise returns an asynchronous `ChatResponse` generator.
    """

    return await self._request(
      ChatResponse,
      'POST',
      '/api/chat',
      json=ChatRequest(
        model=model,
        messages=list(_copy_messages(messages)),
        tools=list(_copy_tools(tools)),
        stream=stream,
        think=think,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        format=format,
        options=options,
        keep_alive=keep_alive,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  async def embed(
    self,
    model: str = '',
    input: Union[str, Sequence[str]] = '',
    truncate: Optional[bool] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
    dimensions: Optional[int] = None,
  ) -> EmbedResponse:
    return await self._request(
      EmbedResponse,
      'POST',
      '/api/embed',
      json=EmbedRequest(
        model=model,
        input=input,
        truncate=truncate,
        options=options,
        keep_alive=keep_alive,
        dimensions=dimensions,
      ).model_dump(exclude_none=True),
    )

  async def embeddings(
    self,
    model: str = '',
    prompt: Optional[str] = None,
    options: Optional[Union[Mapping[str, Any], Options]] = None,
    keep_alive: Optional[Union[float, str]] = None,
  ) -> EmbeddingsResponse:
    """
    Deprecated in favor of `embed`.
    """
    return await self._request(
      EmbeddingsResponse,
      'POST',
      '/api/embeddings',
      json=EmbeddingsRequest(
        model=model,
        prompt=prompt,
        options=options,
        keep_alive=keep_alive,
      ).model_dump(exclude_none=True),
    )

  @overload
  async def pull(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[False] = False,
  ) -> ProgressResponse: ...

  @overload
  async def pull(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[True] = True,
  ) -> AsyncIterator[ProgressResponse]: ...

  async def pull(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: bool = False,
  ) -> Union[ProgressResponse, AsyncIterator[ProgressResponse]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return await self._request(
      ProgressResponse,
      'POST',
      '/api/pull',
      json=PullRequest(
        model=model,
        insecure=insecure,
        stream=stream,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  @overload
  async def push(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[False] = False,
  ) -> ProgressResponse: ...

  @overload
  async def push(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: Literal[True] = True,
  ) -> AsyncIterator[ProgressResponse]: ...

  async def push(
    self,
    model: str,
    *,
    insecure: bool = False,
    stream: bool = False,
  ) -> Union[ProgressResponse, AsyncIterator[ProgressResponse]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """
    return await self._request(
      ProgressResponse,
      'POST',
      '/api/push',
      json=PushRequest(
        model=model,
        insecure=insecure,
        stream=stream,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  @overload
  async def create(
    self,
    model: str,
    quantize: Optional[str] = None,
    from_: Optional[str] = None,
    files: Optional[Dict[str, str]] = None,
    adapters: Optional[Dict[str, str]] = None,
    template: Optional[str] = None,
    license: Optional[Union[str, List[str]]] = None,
    system: Optional[str] = None,
    parameters: Optional[Union[Mapping[str, Any], Options]] = None,
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    stream: Literal[False] = False,
  ) -> ProgressResponse: ...

  @overload
  async def create(
    self,
    model: str,
    quantize: Optional[str] = None,
    from_: Optional[str] = None,
    files: Optional[Dict[str, str]] = None,
    adapters: Optional[Dict[str, str]] = None,
    template: Optional[str] = None,
    license: Optional[Union[str, List[str]]] = None,
    system: Optional[str] = None,
    parameters: Optional[Union[Mapping[str, Any], Options]] = None,
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    stream: Literal[True] = True,
  ) -> AsyncIterator[ProgressResponse]: ...

  async def create(
    self,
    model: str,
    quantize: Optional[str] = None,
    from_: Optional[str] = None,
    files: Optional[Dict[str, str]] = None,
    adapters: Optional[Dict[str, str]] = None,
    template: Optional[str] = None,
    license: Optional[Union[str, List[str]]] = None,
    system: Optional[str] = None,
    parameters: Optional[Union[Mapping[str, Any], Options]] = None,
    messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
    *,
    stream: bool = False,
  ) -> Union[ProgressResponse, AsyncIterator[ProgressResponse]]:
    """
    Raises `ResponseError` if the request could not be fulfilled.

    Returns `ProgressResponse` if `stream` is `False`, otherwise returns a `ProgressResponse` generator.
    """

    return await self._request(
      ProgressResponse,
      'POST',
      '/api/create',
      json=CreateRequest(
        model=model,
        stream=stream,
        quantize=quantize,
        from_=from_,
        files=files,
        adapters=adapters,
        license=license,
        template=template,
        system=system,
        parameters=parameters,
        messages=messages,
      ).model_dump(exclude_none=True),
      stream=stream,
    )

  async def create_blob(self, path: Union[str, Path]) -> str:
    sha256sum = sha256()
    async with await anyio.open_file(path, 'rb') as r:
      while True:
        chunk = await r.read(32 * 1024)
        if not chunk:
          break
        sha256sum.update(chunk)

    digest = f'sha256:{sha256sum.hexdigest()}'

    async def upload_bytes():
      async with await anyio.open_file(path, 'rb') as r:
        while True:
          chunk = await r.read(32 * 1024)
          if not chunk:
            break
          yield chunk

    await self._request_raw('POST', f'/api/blobs/{digest}', content=upload_bytes())

    return digest

  async def save(self, model: str, path: Union[str, Path], models_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Export a locally-available Ollama model to a portable .tar.gz archive.

    See ``Client.save`` for full documentation.
    """
    await anyio.to_thread.run_sync(lambda: _save_model(model, path, models_dir))

  async def load(self, model: str, path: Union[str, Path], models_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Import a model from a .tar.gz archive previously created by ``save()``
    into the local Ollama models directory.

    See ``Client.load`` for full documentation.
    """
    await anyio.to_thread.run_sync(lambda: _load_model(model, path, models_dir))

  async def list(self) -> ListResponse:
    return await self._request(
      ListResponse,
      'GET',
      '/api/tags',
    )

  async def delete(self, model: str) -> StatusResponse:
    r = await self._request_raw(
      'DELETE',
      '/api/delete',
      json=DeleteRequest(
        model=model,
      ).model_dump(exclude_none=True),
    )
    return StatusResponse(
      status='success' if r.status_code == 200 else 'error',
    )

  async def copy(self, source: str, destination: str) -> StatusResponse:
    r = await self._request_raw(
      'POST',
      '/api/copy',
      json=CopyRequest(
        source=source,
        destination=destination,
      ).model_dump(exclude_none=True),
    )
    return StatusResponse(
      status='success' if r.status_code == 200 else 'error',
    )

  async def show(self, model: str) -> ShowResponse:
    return await self._request(
      ShowResponse,
      'POST',
      '/api/show',
      json=ShowRequest(
        model=model,
      ).model_dump(exclude_none=True),
    )

  async def ps(self) -> ProcessResponse:
    return await self._request(
      ProcessResponse,
      'GET',
      '/api/ps',
    )


def _copy_images(images: Optional[Sequence[Union[Image, Any]]]) -> Iterator[Image]:
  for image in images or []:
    yield image if isinstance(image, Image) else Image(value=image)


def _copy_messages(messages: Optional[Sequence[Union[Mapping[str, Any], Message]]]) -> Iterator[Message]:
  for message in messages or []:
    yield Message.model_validate(
      {k: list(_copy_images(v)) if k == 'images' else v for k, v in dict(message).items() if v},
    )


def _copy_tools(tools: Optional[Sequence[Union[Mapping[str, Any], Tool, Callable]]] = None) -> Iterator[Tool]:
  for unprocessed_tool in tools or []:
    yield convert_function_to_tool(unprocessed_tool) if callable(unprocessed_tool) else Tool.model_validate(unprocessed_tool)


def _as_path(s: Optional[Union[str, PathLike]]) -> Union[Path, None]:
  if isinstance(s, (str, Path)):
    try:
      if (p := Path(s)).exists():
        return p
    except Exception:
      ...
  return None


def _get_ollama_models_dir() -> Path:
  """Return the Ollama models directory.

  Resolution order:
  1. ``OLLAMA_MODELS`` environment variable (explicit override).
  2. The running ``ollama serve`` process's HOME, read from ``/proc`` on Linux.
  3. Common service-user locations (``/usr/share/ollama``, ``/var/lib/ollama``).
  4. Current user's ``~/.ollama/models`` (fallback / non-systemd installs).
  """
  env = os.getenv('OLLAMA_MODELS')
  if env:
    return Path(env)

  # Try to read HOME from the running ollama server process via /proc.
  if platform.system() != 'Windows':
    try:
      import glob as _glob
      for comm_path in _glob.glob('/proc/*/comm'):
        try:
          if Path(comm_path).read_text().strip() == 'ollama':
            pid_dir = Path(comm_path).parent
            environ = (pid_dir / 'environ').read_bytes()
            for entry in environ.split(b'\x00'):
              if entry.startswith(b'HOME='):
                home = Path(entry[5:].decode())
                candidate = home / '.ollama' / 'models'
                if candidate.is_dir():
                  return candidate
        except (PermissionError, FileNotFoundError, ValueError):
          continue
    except Exception:
      pass

    # Well-known service-user home locations (systemd installs on Linux).
    for base in (Path('/usr/share/ollama'), Path('/var/lib/ollama')):
      candidate = base / '.ollama' / 'models'
      try:
        if candidate.is_dir():
          return candidate
      except PermissionError:
        # Directory exists but is owned by the service user; return the path
        # anyway so the caller can decide whether they have access.
        return candidate

  if platform.system() == 'Windows':
    base = Path(os.environ.get('USERPROFILE', str(Path.home())))
  else:
    base = Path.home()
  return base / '.ollama' / 'models'


def _parse_model_ref(model: str):
  """
  Parse a model reference into (registry, namespace, name, tag).

  Examples::

    'llama3'               -> ('registry.ollama.ai', 'library', 'llama3', 'latest')
    'llama3:7b'            -> ('registry.ollama.ai', 'library', 'llama3', '7b')
    'user/model'           -> ('registry.ollama.ai', 'user', 'model', 'latest')
    'user/model:tag'       -> ('registry.ollama.ai', 'user', 'model', 'tag')
    'reg.io/ns/model:tag'  -> ('reg.io', 'ns', 'model', 'tag')
  """
  if '/' in model:
    last_slash = model.rfind('/')
    last_segment = model[last_slash + 1:]
    path_prefix = model[:last_slash]
    if ':' in last_segment:
      colon = last_segment.rfind(':')
      name_no_tag = last_segment[:colon]
      tag = last_segment[colon + 1:]
    else:
      name_no_tag = last_segment
      tag = 'latest'
    full_path = f'{path_prefix}/{name_no_tag}'
  else:
    if ':' in model:
      colon = model.rfind(':')
      full_path = model[:colon]
      tag = model[colon + 1:]
    else:
      full_path = model
      tag = 'latest'

  parts = full_path.split('/')
  if len(parts) == 1:
    return 'registry.ollama.ai', 'library', parts[0], tag
  elif len(parts) == 2:
    return 'registry.ollama.ai', parts[0], parts[1], tag
  else:
    return parts[0], parts[1], '/'.join(parts[2:]), tag


def _validate_tar_member(member: tarfile.TarInfo, target_dir: Path) -> None:
  """Reject tar members that could escape the extraction directory."""
  if os.path.isabs(member.name):
    raise ValueError(f'Unsafe tar member with absolute path: {member.name!r}')
  target_resolved = target_dir.resolve()
  resolved = (target_dir / member.name).resolve()
  if not str(resolved).startswith(str(target_resolved) + os.sep):
    raise ValueError(f'Unsafe tar member that escapes target directory: {member.name!r}')
  if member.issym() or member.islnk():
    raise ValueError(f'Tar member is a symlink, which is not permitted: {member.name!r}')


def _save_model(model: str, path: Union[str, Path], models_dir: Optional[Union[str, Path]] = None) -> None:
  """Core implementation of model export — used by both Client.save and AsyncClient.save."""
  models_path = Path(models_dir) if models_dir else _get_ollama_models_dir()
  registry, namespace, name, tag = _parse_model_ref(model)

  manifest_path = models_path / 'manifests' / registry / namespace / name / tag
  if not manifest_path.exists():
    raise FileNotFoundError(f'Model manifest not found: {manifest_path}. Has the model been pulled?')

  manifest_data = manifest_path.read_bytes()
  manifest = json.loads(manifest_data)

  # Collect all blob digests referenced by the manifest.
  digests: List[str] = []
  if config_digest := manifest.get('config', {}).get('digest'):
    digests.append(config_digest)
  for layer in manifest.get('layers', []):
    if d := layer.get('digest'):
      digests.append(d)

  # Verify blobs exist before opening the output archive.
  blob_paths: Dict[str, Path] = {}
  for digest in digests:
    blob_filename = digest.replace(':', '-')
    blob_path = models_path / 'blobs' / blob_filename
    if not blob_path.exists():
      raise FileNotFoundError(f'Blob not found: {blob_path}')
    blob_paths[digest] = blob_path

  meta = {
    'model': model,
    'registry': registry,
    'namespace': namespace,
    'name': name,
    'tag': tag,
    'exported_at': datetime.now(timezone.utc).isoformat(),
  }
  meta_bytes = json.dumps(meta, indent=2).encode()

  with tarfile.open(path, 'w:gz') as tf:
    # meta.json — written from memory, no temp file required.
    info = tarfile.TarInfo(name='meta.json')
    info.size = len(meta_bytes)
    tf.addfile(info, io.BytesIO(meta_bytes))

    # manifest.json
    info = tarfile.TarInfo(name='manifest.json')
    info.size = len(manifest_data)
    tf.addfile(info, io.BytesIO(manifest_data))

    # blobs/
    for digest, blob_path in blob_paths.items():
      blob_filename = digest.replace(':', '-')
      tf.add(blob_path, arcname=f'blobs/{blob_filename}')


def _load_model(model: str, path: Union[str, Path], models_dir: Optional[Union[str, Path]] = None) -> None:
  """Core implementation of model import — used by both Client.load and AsyncClient.load."""
  archive_path = Path(path)
  if not archive_path.exists():
    raise FileNotFoundError(f'Archive not found: {archive_path}')

  models_path = Path(models_dir) if models_dir else _get_ollama_models_dir()
  registry, namespace, name, tag = _parse_model_ref(model)

  with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)

    # Validate every member before extraction to prevent path traversal.
    with tarfile.open(archive_path, 'r:gz') as tf:
      members = tf.getmembers()
      for member in members:
        _validate_tar_member(member, tmp)
      # extractall is safe here: all members have been validated above.
      try:
        tf.extractall(tmp, filter='data')
      except TypeError:
        # filter parameter was added in Python 3.12; fall back on older versions.
        tf.extractall(tmp)  # noqa: S202

    manifest_file = tmp / 'manifest.json'
    if not manifest_file.exists():
      raise ValueError('Archive is missing manifest.json')
    manifest_data = manifest_file.read_bytes()
    manifest = json.loads(manifest_data)

    # Collect expected blob digests.
    digests: List[str] = []
    if config_digest := manifest.get('config', {}).get('digest'):
      digests.append(config_digest)
    for layer in manifest.get('layers', []):
      if d := layer.get('digest'):
        digests.append(d)

    # Verify each blob's SHA-256 checksum, then copy into the models directory.
    blobs_dst = models_path / 'blobs'
    blobs_dst.mkdir(parents=True, exist_ok=True)

    for digest in digests:
      algo, expected_hex = digest.split(':', 1)
      if algo != 'sha256':
        raise ValueError(f'Unsupported digest algorithm: {algo!r}')

      blob_filename = digest.replace(':', '-')
      src = tmp / 'blobs' / blob_filename
      if not src.exists():
        raise ValueError(f'Archive is missing blob: {blob_filename!r}')

      actual = sha256()
      with open(src, 'rb') as f:
        while True:
          chunk = f.read(32 * 1024)
          if not chunk:
            break
          actual.update(chunk)
      if actual.hexdigest() != expected_hex:
        raise ValueError(f'Digest mismatch for blob {blob_filename!r}: archive may be corrupt')

      dst = blobs_dst / blob_filename
      if not dst.exists():
        shutil.copy2(src, dst)

    # Write the manifest last; Ollama detects it without a restart.
    manifest_dir = models_path / 'manifests' / registry / namespace / name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / tag).write_bytes(manifest_data)





def _parse_host(host: Optional[str]) -> str:
  """
  >>> _parse_host(None)
  'http://127.0.0.1:11434'
  >>> _parse_host('')
  'http://127.0.0.1:11434'
  >>> _parse_host('1.2.3.4')
  'http://1.2.3.4:11434'
  >>> _parse_host(':56789')
  'http://127.0.0.1:56789'
  >>> _parse_host('1.2.3.4:56789')
  'http://1.2.3.4:56789'
  >>> _parse_host('http://1.2.3.4')
  'http://1.2.3.4:80'
  >>> _parse_host('https://1.2.3.4')
  'https://1.2.3.4:443'
  >>> _parse_host('https://1.2.3.4:56789')
  'https://1.2.3.4:56789'
  >>> _parse_host('example.com')
  'http://example.com:11434'
  >>> _parse_host('example.com:56789')
  'http://example.com:56789'
  >>> _parse_host('http://example.com')
  'http://example.com:80'
  >>> _parse_host('https://example.com')
  'https://example.com:443'
  >>> _parse_host('https://example.com:56789')
  'https://example.com:56789'
  >>> _parse_host('example.com/')
  'http://example.com:11434'
  >>> _parse_host('example.com:56789/')
  'http://example.com:56789'
  >>> _parse_host('example.com/path')
  'http://example.com:11434/path'
  >>> _parse_host('example.com:56789/path')
  'http://example.com:56789/path'
  >>> _parse_host('https://example.com:56789/path')
  'https://example.com:56789/path'
  >>> _parse_host('example.com:56789/path/')
  'http://example.com:56789/path'
  >>> _parse_host('[0001:002:003:0004::1]')
  'http://[0001:002:003:0004::1]:11434'
  >>> _parse_host('[0001:002:003:0004::1]:56789')
  'http://[0001:002:003:0004::1]:56789'
  >>> _parse_host('http://[0001:002:003:0004::1]')
  'http://[0001:002:003:0004::1]:80'
  >>> _parse_host('https://[0001:002:003:0004::1]')
  'https://[0001:002:003:0004::1]:443'
  >>> _parse_host('https://[0001:002:003:0004::1]:56789')
  'https://[0001:002:003:0004::1]:56789'
  >>> _parse_host('[0001:002:003:0004::1]/')
  'http://[0001:002:003:0004::1]:11434'
  >>> _parse_host('[0001:002:003:0004::1]:56789/')
  'http://[0001:002:003:0004::1]:56789'
  >>> _parse_host('[0001:002:003:0004::1]/path')
  'http://[0001:002:003:0004::1]:11434/path'
  >>> _parse_host('[0001:002:003:0004::1]:56789/path')
  'http://[0001:002:003:0004::1]:56789/path'
  >>> _parse_host('https://[0001:002:003:0004::1]:56789/path')
  'https://[0001:002:003:0004::1]:56789/path'
  >>> _parse_host('[0001:002:003:0004::1]:56789/path/')
  'http://[0001:002:003:0004::1]:56789/path'
  """

  host, port = host or '', 11434
  scheme, _, hostport = host.partition('://')
  if not hostport:
    scheme, hostport = 'http', host
  elif scheme == 'http':
    port = 80
  elif scheme == 'https':
    port = 443

  split = urllib.parse.urlsplit(f'{scheme}://{hostport}')
  host = split.hostname or '127.0.0.1'
  port = split.port or port

  try:
    if isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address):
      # Fix missing square brackets for IPv6 from urlsplit
      host = f'[{host}]'
  except ValueError:
    ...

  if path := split.path.strip('/'):
    return f'{scheme}://{host}:{port}/{path}'

  return f'{scheme}://{host}:{port}'
