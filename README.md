# AI Engineering · Coderhouse

Pre-entregas del curso **AI Engineering (Coderhouse)**, en Python 3.12.

| Pre-entrega | Qué es | Código | Script de prueba |
|---|---|---|---|
| [1 · Cliente de LLM robusto y asíncrono](#pre-entrega-1-cliente-de-llm-robusto-y-asíncrono) | Una interfaz asíncrona común para OpenAI y Anthropic, con streaming y errores controlados | `llm_client/` | `main.py` |
| [2 · Pipeline de procesamiento validado](#pre-entrega-2-pipeline-de-procesamiento-validado) | Una cadena LCEL que extrae tecnologías, criticidad y un resumen técnico de un texto, validados con Pydantic | `entity_pipeline/` | `run_pipeline.py` |

La Pre-entrega 2 reutiliza la configuración (`Settings`) y la clasificación de errores de la 1.

## Instalación

Requisitos: Python 3.12+ y [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # crea el entorno e instala dependencias, incluidas las de desarrollo
cp .env.example .env         # completá al menos una API key
```

Sin uv:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # solo para ejecutar los scripts
pip install -r requirements-dev.txt    # además, pytest y ruff para correr los tests
```

### Variables de entorno

| Variable | Default | Validación |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` o `anthropic` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | vacía | se guardan como `SecretStr`; vacía = proveedor no configurado |
| `OPENAI_MODEL` | `gpt-4o-mini` | texto no vacío |
| `OPENAI_TEMPERATURE` | `0.7` | entre 0 y 2 (solo Pre-entrega 1; la extracción usa 0) |
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
uv run pytest           # 128 tests, sin red ni API keys
uv run ruff check .
```

Con pip, después de `pip install -r requirements-dev.txt`: `pytest` y `ruff check .`.

Los tests usan los SDK reales contra respuestas HTTP simuladas: así se prueba que el request que
sale y la respuesta que vuelve se traducen bien, incluidos los errores 401, 402, 429, 500 y 529,
el streaming, las conexiones que se cortan, las respuestas cortadas por límite de tokens y los
rechazos del modelo.

### Estructura

```
llm_client/                 # Pre-entrega 1
├── schemas.py              # modelos Pydantic: mensajes, configuración, respuesta y error
├── settings.py             # lectura y validación del .env (pydantic-settings)
├── errors.py               # LLMError y traducción de excepciones de los SDK a ErrorInfo
├── manager.py              # AsyncLLMManager: elige proveedor, reintenta y normaliza
└── clients/
    ├── base.py             # BaseLLMClient (ABC): generate(), stream(), aclose()
    ├── openai_client.py    # OpenAIClient sobre AsyncOpenAI
    └── anthropic_client.py # AnthropicClient sobre AsyncAnthropic
entity_pipeline/            # Pre-entrega 2
├── schemas.py              # contrato Pydantic: ExtraccionTecnica, Criticidad, SolicitudDeExtraccion
└── chain.py                # prompt + LLM con salida estructurada + validación + reintentos + logs
main.py                     # script de la Pre-entrega 1: modo normal y streaming
run_pipeline.py             # script de la Pre-entrega 2: extracción sobre textos de ejemplo
tests/
```

---

## Pre-entrega 1: Cliente de LLM robusto y asíncrono

Un cliente asíncrono que habla con **OpenAI** y **Anthropic** a través de una sola interfaz, con
streaming de tokens, validación con Pydantic y manejo de errores que no corta el programa.

```python
from llm_client import AsyncLLMManager, Provider, Settings

async with AsyncLLMManager(Settings()) as llm:
    con_el_default = await llm.generate("¿Qué es la entropía?")
    con_claude = await llm.generate("¿Qué es la entropía?", provider=Provider.ANTHROPIC)

    async for token in llm.stream("¿Qué es la entropía?"):
        print(token, end="", flush=True)
```

```bash
uv run main.py                      # pregunta "¿Qué es la entropía?" a cada proveedor con key
uv run main.py --provider anthropic
```

### Qué resuelve

| Requisito | Dónde está |
|---|---|
| **Intercambiabilidad**: elegir OpenAI o Anthropic bajo una interfaz común | `AsyncLLMManager` (`llm_client/manager.py`) crea el cliente según `LLM_PROVIDER` o el argumento `provider=`. Los dos clientes heredan de `BaseLLMClient` y devuelven el mismo `ModelResponse`. |
| **Asincronía**: ninguna llamada bloquea el event loop | Se usan `AsyncOpenAI` y `AsyncAnthropic`; todo es `async`/`await`. |
| **Streaming**: generador asíncrono de tokens | `stream()` es un `async def` con `yield` que recorre el stream del SDK con `async for`. |
| **Validación con Pydantic** | `llm_client/schemas.py`: `ChatMessage`, `ChatRequest`, `LLMConfig`, `RetryPolicy`, `ModelResponse`, `ErrorInfo`. La configuración del `.env` se valida con `pydantic-settings`. |
| **Resiliencia**: fallos de red y rate limiting sin romper el flujo | Reintentos con backoff exponencial y jitter, respeto de `Retry-After`, errores normalizados en `ErrorInfo`. |

### Decisiones de diseño

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
(`httpx2`, TLS y cierre de conexión), y las convierte en `LLMError` con un `ErrorInfo` (`kind`,
`status_code`, `retry_after_seconds`, `attempts`, `retryable`). El resto del código nunca ve una
excepción de `openai`, `anthropic` ni de la conexión. Si el error llega como evento dentro de un
stream que ya respondió 200, se clasifica por su tipo (`overloaded_error`, `rate_limit_error`,
etc.) y queda con `status_code: null`.

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

### Salida de ejemplo

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

---

## Pre-entrega 2: Pipeline de procesamiento validado

Un pipeline que recibe un párrafo técnico (una descripción de arquitectura o un log de error) y
devuelve un objeto `ExtraccionTecnica` validado: las tecnologías que nombra, el nivel de
criticidad y un resumen técnico.

```python
from entity_pipeline import build_extraction_chain, extract
from llm_client import Settings

chain = build_extraction_chain(Settings())
resultado = await extract(chain, "ERROR payment-service: FATAL: too many clients already")
print(resultado.model_dump_json(indent=2))
```

```bash
uv run run_pipeline.py                         # procesa en paralelo los 3 textos de ejemplo
uv run run_pipeline.py "Texto propio a analizar"
uv run run_pipeline.py --provider anthropic
```

Sin `--provider`, usa `LLM_PROVIDER` si tiene key, o el otro proveedor configurado.

### Qué resuelve

| Componente | Dónde está |
|---|---|
| **Esquema Pydantic** con `tecnologias` (lista), `nivel_de_criticidad` (enum `baja`/`media`/`alta`) y `resumen_tecnico` | `entity_pipeline/schemas.py`: `ExtraccionTecnica` y `Criticidad`. El texto de entrada también se valida (`SolicitudDeExtraccion`). |
| **Prompt template** modular con el texto y las instrucciones de formato | `build_prompt()` en `chain.py`: un `ChatPromptTemplate` con las variables `{texto}` y `{format_instructions}`. Las instrucciones se generan desde las descripciones de los campos del esquema. |
| **Cadena LCEL** Prompt + LLM + Output Parser | `prompt \| llm.with_structured_output(ExtraccionTecnica) \| validar_extraccion` en `build_extraction_chain()`. |
| **Resiliencia**: reintento si el JSON llega mal formado o incompleto | `.with_retry(...)` sobre el modelo y la validación, con backoff exponencial y jitter; hasta `LLM_MAX_ATTEMPTS` intentos. |
| **Ejecución asíncrona y logs** | `extract()` usa `await chain.ainvoke(...)`; `run_pipeline.py` procesa los textos en paralelo con `asyncio.gather`. Un callback de LangChain (`PipelineLogger`) registra cada llamada, cada respuesta, cada validación y cada reintento. |

### Restricciones del contrato

- `tecnologias`: al menos una; se recortan espacios y se quitan los duplicados sin importar
  mayúsculas (`Kafka` y `kafka` cuentan como una).
- `nivel_de_criticidad`: solo `baja`, `media` o `alta`. La regla para elegir va en la descripción
  del campo, así que el modelo la recibe en el esquema y en el prompt.
- `resumen_tecnico`: no vacío y de hasta 600 caracteres.
- No se aceptan campos que no estén en el contrato (`extra="forbid"`).
- El texto de entrada no puede estar vacío ni superar los 20 000 caracteres.

### Salida de ejemplo

Para el log de error de los ejemplos (los valores exactos dependen del modelo):

```json
{
  "tecnologias": [
    "psycopg",
    "PostgreSQL"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El servicio de pagos no puede abrir conexiones a PostgreSQL porque la base alcanzó su límite de clientes, y el checkout responde 503."
}
```

Si la primera respuesta del modelo no cumple el esquema, el log muestra la validación fallida y
el reintento:

```
INFO entity_pipeline.chain: Llamando a gpt-4o-mini
WARNING entity_pipeline.chain: La salida no cumple el esquema: tecnologias: Value error, el texto tiene que nombrar al menos una tecnología
INFO entity_pipeline.chain: Reintentando la extracción: intento 2
INFO entity_pipeline.chain: Llamando a gpt-4o-mini
INFO entity_pipeline.chain: Respuesta recibida: fin=stop, tokens=… entrada / … salida
INFO entity_pipeline.chain: Salida validada: 2 tecnologías, criticidad alta
```

El tercer ejemplo es un texto ambiguo, sin tecnologías claras: es la prueba de estrés. Si el
modelo devuelve una lista de tecnologías vacía, el validador la rechaza y la cadena reintenta. Como
el request se repite igual (en OpenAI, con `temperature=0`), lo más probable es que el error se
repita; al agotar los intentos, el script muestra el error estructurado y sigue con el resto:

```
=== texto ambiguo ===
La extracción falló, pero el programa sigue:
{
  "error": "salida_invalida",
  "detalle": "tecnologias: Value error, el texto tiene que nombrar al menos una tecnología"
}
```

### Decisiones de diseño

**Salida estructurada nativa en los dos proveedores.** `with_structured_output` usa
`method="json_schema"`: OpenAI recibe el esquema en `response_format` con `strict: true`, y
Anthropic en `output_config.format`. En Anthropic se evita el modo por tool calling porque obliga
a forzar `tool_choice`, y eso devuelve un 400 en Claude Opus 5.5.

**No se ignora el `finish_reason`.** Si el modelo cortó la respuesta por límite de tokens, el
objeto se descarta como incompleto aunque el JSON parcial pareciera válido, y se reintenta:

| | OpenAI | Anthropic |
|---|---|---|
| Respuesta cortada | `finish_reason: "length"`; el SDK lanza `LengthFinishReasonError` al parsear | `stop_reason: "max_tokens"`; el paso `validar_extraccion` lo revisa antes de aceptar el objeto y lanza `IncompleteOutputError` |
| JSON mal formado | `ValidationError` | `OutputParserException` |
| JSON válido que no cumple el esquema | `ValidationError` | `OutputParserException`, con el `ValidationError` como causa |
| El modelo se niega a responder | `ModelRefusalError` o `ContentFilterFinishReasonError` | `ModelRefusalError` (`stop_reason: "refusal"`) |

Cuando el JSON no cumple el esquema, los logs y la salida del script muestran el campo que falló y
por qué, con el mismo texto en los dos proveedores. Cualquier otro error inesperado en un texto se
informa como `"error": "inesperado"` y no frena el procesamiento de los demás.

**Qué se reintenta y qué no.** Se reintentan los errores de formato (JSON mal formado, campos que
no cumplen el contrato, respuesta incompleta) y los errores transitorios del proveedor (red,
timeout, 429 de rate limit, 5xx, 529). Para decidir sobre los errores del proveedor se reutiliza
la clasificación de la Pre-entrega 1 (`classify_sdk_error`). Una key inválida o la falta de cuota
no se reintentan: repetir la llamada no los arregla. Tampoco se reintenta cuando el modelo se
niega a responder; el script lo informa como `"error": "sin_respuesta"`. Los SDK se crean con
`max_retries=0`, así que los reintentos ocurren en un solo lugar, el `.with_retry()` de la cadena.

**Temperatura 0 en OpenAI.** Para extraer datos conviene que el modelo varíe lo menos posible, así
que la cadena usa `temperature=0` aunque `OPENAI_TEMPERATURE` diga otra cosa. En Anthropic no se
envía `temperature` (la API ya no la acepta) y se usa `ANTHROPIC_EFFORT`.

**Prompt sin f-strings.** El texto del usuario entra por la variable `{texto}` del
`ChatPromptTemplate` y las instrucciones por `{format_instructions}` (con `.partial()`), así
LangChain gestiona las variables y el prompt se puede cambiar sin tocar la cadena.
