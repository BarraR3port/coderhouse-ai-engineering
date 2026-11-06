# Unified Async LLM Client

Pre-entrega 1 de **AI Engineering (Coderhouse)**: un cliente asíncrono en Python 3.12 que habla con
**OpenAI** y **Anthropic** a través de una sola interfaz, con streaming de tokens, validación con
Pydantic y manejo de errores que no corta el programa.

```python
from llm_client import AsyncLLMManager, Provider, Settings

async with AsyncLLMManager(Settings()) as llm:
    con_el_default = await llm.generate("¿Qué es la entropía?")
    con_claude = await llm.generate("¿Qué es la entropía?", provider=Provider.ANTHROPIC)

    async for token in llm.stream("¿Qué es la entropía?"):
        print(token, end="", flush=True)
```

## Qué resuelve

| Requisito | Dónde está |
|---|---|
| **Intercambiabilidad**: elegir OpenAI o Anthropic bajo una interfaz común | `AsyncLLMManager` (`llm_client/manager.py`) crea el cliente según `LLM_PROVIDER` o el argumento `provider=`. Los dos clientes heredan de `BaseLLMClient` y devuelven el mismo `ModelResponse`. |
| **Asincronía**: ninguna llamada bloquea el event loop | Se usan `AsyncOpenAI` y `AsyncAnthropic`; todo es `async`/`await`. |
| **Streaming**: generador asíncrono de tokens | `stream()` es un `async def` con `yield` que recorre el stream del SDK con `async for`. |
| **Validación con Pydantic** | `llm_client/schemas.py`: `ChatMessage`, `ChatRequest`, `LLMConfig`, `RetryPolicy`, `ModelResponse`, `ErrorInfo`. La configuración del `.env` se valida con `pydantic-settings`. |
| **Resiliencia**: fallos de red y rate limiting sin romper el flujo | Reintentos con backoff exponencial y jitter, respeto de `Retry-After`, errores normalizados en `ErrorInfo`. |

## Estructura

```
llm_client/
├── schemas.py              # modelos Pydantic: mensajes, configuración, respuesta y error
├── settings.py             # lectura y validación del .env (pydantic-settings)
├── errors.py               # LLMError y traducción de excepciones de los SDK a ErrorInfo
├── manager.py              # AsyncLLMManager: elige proveedor, reintenta y normaliza
└── clients/
    ├── base.py             # BaseLLMClient (ABC): generate(), stream(), aclose()
    ├── openai_client.py    # OpenAIClient sobre AsyncOpenAI
    └── anthropic_client.py # AnthropicClient sobre AsyncAnthropic
main.py                     # script de validación: modo normal y streaming
tests/                      # 79 tests sin red (HTTP simulado)
```

## Cómo correrlo

Requisitos: Python 3.12+ y [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # crea el entorno e instala dependencias
cp .env.example .env         # completá al menos una API key
uv run main.py               # pregunta "¿Qué es la entropía?" a cada proveedor con key
uv run main.py --provider anthropic
```

Sin uv: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python main.py`.

### Variables de entorno

| Variable | Default | Validación |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` o `anthropic` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | vacía | se guardan como `SecretStr`; vacía = proveedor no configurado |
| `OPENAI_MODEL` | `gpt-4o-mini` | texto no vacío |
| `OPENAI_TEMPERATURE` | `0.7` | entre 0 y 2 |
| `ANTHROPIC_MODEL` | `claude-opus-5-5` | texto no vacío (para pruebas más baratas: `claude-haiku-5-5`) |
| `ANTHROPIC_EFFORT` | `low` | `low`, `medium`, `high`, `xhigh` o `max` |
| `LLM_MAX_TOKENS` | `16000` | entre 1 y 128 000 |
| `LLM_TIMEOUT_SECONDS` | `60` | mayor a 0 y hasta 600 |
| `LLM_MAX_ATTEMPTS` | `3` | entre 1 y 10 |
| `LLM_BASE_DELAY_SECONDS` / `LLM_MAX_DELAY_SECONDS` | `1` / `20` | la base no puede superar al máximo |

Un valor fuera de rango corta el arranque con un mensaje claro, antes de llamar a cualquier API.
El `.env` se lee siempre desde la raíz del proyecto, sin importar desde qué carpeta se ejecute.

### Tests

```bash
uv run pytest           # 79 tests, sin red ni API keys
uv run ruff check .
```

Los tests de los clientes usan los SDK reales contra un transporte HTTP simulado: así se prueba que
el request que sale (mensajes, `system`, `temperature`, `effort`) y la respuesta que vuelve se
traducen bien, incluidos el streaming por SSE, los códigos 401, 402, 429 y 529, los eventos
`error` que llegan dentro del stream y las conexiones que se cortan a mitad de la respuesta.
`tests/test_main.py` comprueba que `main.py` termina bien aunque el streaming falle.

## Decisiones de diseño

**Una interfaz, dos proveedores.** Las diferencias entre SDK quedan dentro de cada cliente:

| | OpenAI | Anthropic |
|---|---|---|
| Prompt de sistema | va como mensaje con `role: system` | se saca de la lista y va en el parámetro `system` |
| Texto de la respuesta | `choices[0].message.content` | concatenación de los bloques `text` de `content` (se ignoran los de `thinking`) |
| Streaming | `chat.completions.create(stream=True)` | `messages.stream()` → `text_stream` |
| Creatividad | `temperature` | `effort` (desde la API `2026-10-21`, Anthropic rechaza `temperature`) |

`LLMConfig` valida esa última fila: pedir `temperature` para Anthropic o `effort` para OpenAI es
un error de configuración.

**Errores con forma conocida.** Cada cliente atrapa las excepciones de su SDK y las de red
(`httpx2`, TLS y cierre de conexión), y las convierte en `LLMError` con un `ErrorInfo` (`kind`, `status_code`,
`retry_after_seconds`, `attempts`, `retryable`). El resto del código nunca ve una excepción de
`openai`, `anthropic` ni de la conexión. Si el error llega como evento dentro de un stream que ya
respondió 200, se clasifica por su tipo (`overloaded_error`, `rate_limit_error`, etc.) y queda con
`status_code: null`.

| `kind` | Origen | ¿Se reintenta? |
|---|---|---|
| `rate_limit` | HTTP 429 o `rate_limit_error` | sí, esperando lo que pide `Retry-After`* |
| `timeout`, `connection` | timeout, red caída o conexión cortada | sí |
| `provider_unavailable` | HTTP 5xx, 529 o `overloaded_error` | sí |
| `quota_exceeded` | 429 `insufficient_quota` de OpenAI, 402 de Anthropic | no: reintentar no recupera crédito |
| `authentication` | HTTP 401/403 (key inválida) | no |
| `invalid_request` | otros 4xx | no |

\* Si `Retry-After` pide esperar más que `LLM_MAX_DELAY_SECONDS`, no se reintenta: se devuelve el
error con `retry_after_seconds` para que quien llama decida cuándo volver a intentar.

**Reintentos en un solo lugar.** Los SDK se crean con `max_retries=0` y el `AsyncLLMManager`
reintenta con backoff exponencial y jitter (`RetryPolicy`). Así hay una sola política y queda
registrada en los logs.

**El programa no se cae.**
- `generate()` nunca lanza por un fallo del proveedor: devuelve un `ModelResponse` con `ok=False` y
  el `error` estructurado.
- `stream()` reintenta solo si el fallo ocurre **antes del primer token** (después repetiría texto
  ya mostrado). Si no se recupera, lanza `LLMError` con el mismo `ErrorInfo`, y `main.py` lo atrapa.
- Lo único que lanza `ValueError` es un error del programador: un prompt vacío, una conversación que
  no termina en un mensaje del usuario, o un proveedor sin API key (`ConfigurationError`).

## Salida de ejemplo

Con una key inválida, el programa informa y sigue con el siguiente modo y proveedor:

```
=== anthropic · modo normal ===
La llamada falló, pero el programa sigue. Error estructurado:
{
  "kind": "authentication",
  "provider": "anthropic",
  "message": "Error code: 401 - ... invalid x-api-key ...",
  "status_code": 401,
  "retry_after_seconds": null,
  "attempts": 1,
  "retryable": false
}
```

Sin red, los errores de conexión se reintentan antes de informar:

```
INFO llm_client.manager: anthropic respondió connection (intento 1/3); reintento en 0.9 s
INFO llm_client.manager: anthropic respondió connection (intento 2/3); reintento en 1.6 s
```
