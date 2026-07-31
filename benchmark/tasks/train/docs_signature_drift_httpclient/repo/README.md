# minihttp

A small HTTP client with the network taken out of it. `HTTPClient` owns the retry loop,
the URL joining and the status policy; a *transport* owns the socket and is passed in.

```python
from src import HTTPClient, StubTransport, Response

client = HTTPClient(StubTransport([Response(200, {}, "pong")]), base_url="https://api.example.com")
client.get("/ping").body
```

| module | what it does |
| --- | --- |
| `src/httpclient.py` | the client, the response object, the errors, URL building |
| `src/transports.py` | a transport that answers from a script, for tests and local dev |

## Docstring house style

Reference documentation is generated straight from these docstrings, so they follow one
fixed shape and the generator is strict about it:

* one `:param <name>:` line per parameter, **in signature order**, using the name the
  function actually takes;
* where a parameter has a default, the line ends with ``(default ``<value>``)`` and the
  value is written as a Python literal — ``(default ``None``)``, ``(default ``3``)``,
  ``(default ``''``)``;
* one `:raises <ExceptionType>:` line per exception the call can really produce, naming
  a type that exists in the module;
* a closing `:returns:` line.

Every public callable gets one. A parameter's description may wrap onto further indented
lines; the shape above is what the generator parses, so keep to it rather than inventing
a second convention alongside it.
