# AI Engineering · Coderhouse

Pre-entregas del curso **AI Engineering (Coderhouse)**, en Python 3.11+.

| Pre-entrega | Qué es | Código | Script de prueba |
|---|---|---|---|
| [1 · Cliente de LLM robusto y asíncrono](#pre-entrega-1-cliente-de-llm-robusto-y-asíncrono) | Una interfaz asíncrona común para OpenAI y Anthropic, con streaming y errores controlados | `llm_client/` | `main.py` |
| [2 · Pipeline de procesamiento validado](#pre-entrega-2-pipeline-de-procesamiento-validado) | Una cadena LCEL que extrae tecnologías, criticidad y un resumen técnico de un texto, validados con Pydantic | `entity_pipeline/` | `run_pipeline.py` |
| [3 · Sistema de recuperación semántica local (RAG)](#pre-entrega-3-sistema-de-recuperación-semántica-local-rag) | Un RAG que responde con la documentación de `data/`, cita sus fuentes y dice "No lo sé" si la respuesta no está | `rag/` | `ingest.py` y `run_rag.py` |

Cada pre-entrega reutiliza piezas de las anteriores: la configuración (`Settings`) y la
clasificación de errores de la 1, y el armado del modelo y los logs de la 2.

## Instalación

Requisitos: Python 3.11+ y [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # crea el entorno e instala dependencias, incluidas las de desarrollo
cp .env.example .env         # completá al menos una API key
```

Sin uv:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
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
| `RAG_DATA_DIR` | `data` | carpeta con los `.md` / `.txt` a indexar |
| `RAG_VECTORSTORE_DIR` | `vectorstore` | carpeta donde ChromaDB guarda la base (no se versiona) |
| `RAG_EMBEDDING_CACHE_DIR` | `.cache/fastembed` | dónde queda el modelo de embeddings descargado (no se versiona) |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | modelo de embeddings local (fastembed) |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `120` / `20` | en tokens del modelo de embeddings, que lee hasta 128; el solapamiento tiene que ser menor que el tamaño |
| `RAG_TOP_K` | `4` | entre 3 y 5 fragmentos por pregunta |
| `RAG_MIN_RELEVANCE` | `0.45` | similitud coseno mínima, entre 0 y 1, para pasarle un fragmento al LLM |

Un valor fuera de rango corta el arranque con un mensaje claro, antes de llamar a cualquier API.
El `.env` se lee siempre desde la raíz del proyecto, sin importar desde qué carpeta se ejecute.

### Tests

```bash
uv run pytest           # 166 tests, sin red ni API keys
uv run ruff check .
```

Salida en Python 3.11, la versión del entorno de corrección (`.python-version` la fija):

```text
$ uv run python --version
Python 3.11.15
$ uv run pytest -q
........................................................................ [ 43%]
........................................................................ [ 86%]
......................                                                   [100%]
166 passed in 19.77s
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
rag/                        # Pre-entrega 3
├── settings.py             # RagSettings: carpetas, modelo de embeddings, chunking, top_k (3 a 5)
├── embeddings.py           # LocalEmbeddings: embeddings ONNX en la máquina, sin API key
├── ingest.py               # carga, limpieza, chunking, persistencia en ChromaDB, control de embeddings
├── schemas.py              # contrato Pydantic: SolicitudRAG, RespuestaDelModelo, RespuestaRAG, Fuente
└── chain.py                # cadena LCEL asíncrona y get_rag_response
data/                       # dataset de la Pre-entrega 3: 4 documentos .md
ingest.py                   # script de la Pre-entrega 3: indexa data/ en ./vectorstore
run_rag.py                  # script de la Pre-entrega 3: responde preguntas con RAG
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
INFO entity_pipeline.chain: Reintentando: intento 2
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

---

## Pre-entrega 3: Sistema de recuperación semántica local (RAG)

Un sistema RAG de punta a punta sobre la documentación interna de una plataforma ficticia,
**Nimbus**: cuatro documentos de `data/` sobre despliegues, incidentes, la API pública y
observabilidad. Recibe una pregunta, busca los fragmentos relevantes en una base vectorial
**ChromaDB** persistida en `./vectorstore` y genera una respuesta que usa solo esa información,
con sus fuentes. Si la respuesta no está en los documentos, responde **"No lo sé"**.

```python
from rag import get_rag_response

respuesta = await get_rag_response(chain, "¿Se puede desplegar un viernes?")
respuesta.respuesta  # texto basado solo en los documentos, o "No lo sé"
respuesta.fuentes  # [Fuente(archivo="despliegues.md", seccion="Ventanas de despliegue", ...)]
```

### Cómo se ejecuta

```bash
uv run ingest.py                       # fragmenta data/ y lo guarda en ./vectorstore
uv run run_rag.py                      # responde 3 preguntas de ejemplo (2 del tema, 1 fuera de tema)
uv run run_rag.py "¿Cuánto duran los tokens de acceso?"
uv run run_rag.py --solo-recuperar     # muestra qué fragmentos se recuperan, sin LLM ni API key
uv run ingest.py --rebuild             # reindexa aunque nada haya cambiado
```

`run_rag.py` corre la ingesta antes de responder, así que con una base ya indexada no repite
el trabajo. Para generar respuestas hace falta una API key de OpenAI o de Anthropic en el `.env`.
Los embeddings no necesitan ninguna: el modelo corre en la máquina y se descarga una sola vez
(unos 220 MB) en `.cache/fastembed`.

### Cómo funciona

| Paso | Dónde | Qué hace |
|---|---|---|
| 1. Carga | `rag/ingest.py` · `load_documents` | Lee los `.md` y `.txt` de `data/` y guarda el nombre del archivo como metadato |
| 2. Limpieza | `clean_text` | Saca caracteres basura, espacios repetidos y saltos de línea de más |
| 3. Chunking | `split_documents` | Corta primero por sección de Markdown (`#`, `##`), que queda como metadato. Si una sección supera `RAG_CHUNK_SIZE`, la parte con `RecursiveCharacterTextSplitter` (párrafo → línea → oración → palabra), con `RAG_CHUNK_OVERLAP` de solapamiento. Mide en **tokens del tokenizer del propio modelo de embeddings**, que trunca a 128: así ningún fragmento pierde su final al vectorizarse. Un `RAG_CHUNK_SIZE` mayor de lo que lee el modelo se rechaza |
| 4. Persistencia | `ingest` | Guarda los fragmentos en ChromaDB (`./vectorstore`, distancia coseno) con IDs deterministas (`despliegues.md#004`). Antes de indexar compara una huella de los documentos y de la configuración: si no cambió nada, no reindexa. La huella se escribe recién cuando todos los fragmentos quedaron guardados; si la vectorización falla, la colección se borra y la próxima ingesta empieza de cero |
| 5. Recuperación | `rag/chain.py` · `retrieve` | Convierte la pregunta en embedding con **el mismo modelo** que indexó y trae los `RAG_TOP_K` (3 a 5) fragmentos más parecidos. Descarta los que no llegan a `RAG_MIN_RELEVANCE` |
| 6. Generación | `build_rag_chain` | `prompt | llm | PydanticOutputParser` con los fragmentos numerados como CONTEXTO. El prompt prohíbe usar conocimiento propio y obliga a responder "No lo sé" si el contexto no alcanza. Reintenta con backoff exponencial ante salidas inválidas y ante errores temporales del proveedor (429, 5xx, timeouts), con la misma política que la Pre-entrega 2 |
| 7. Fuentes | `cite_sources` | El modelo devuelve qué fragmentos usó; las fuentes (archivo, sección, ID, relevancia) salen de los **metadatos** de esos fragmentos, no del texto del modelo |

Toda la cadena es asíncrona: `get_rag_response` hace `await chain.ainvoke(...)`, la búsqueda usa
`asimilarity_search_with_relevance_scores` y el modelo se llama por su vía async.

### Salida de ejemplo

Ingesta, dos veces seguidas:

```text
$ uv run ingest.py
INFO rag.ingest: Indexados 25 fragmentos de 4 documentos en .../vectorstore
Listo: 25 fragmentos de 4 documentos guardados en .../vectorstore
$ uv run ingest.py
INFO rag.ingest: La colección 'nimbus_docs' ya está al día (25 fragmentos): no se reindexa
La base vectorial ya estaba al día (25 fragmentos): no se reindexó
```

Recuperación con los embeddings reales (`--solo-recuperar`): relevancia, si pasa el umbral, ID y
sección de cada fragmento.

```text
=== ¿Qué pasa si durante el canario la tasa de errores 5xx supera el 1 %? ===
0.495  usado=sí  despliegues.md#004  Criterios de rollback
0.487  usado=sí  despliegues.md#003  Flujo de despliegue
0.447  usado=no  api-publica.md#003  Límites de uso (rate limiting)
0.437  usado=no  incidentes.md#000  Niveles de severidad

=== ¿Cuál es la capital de Australia? ===
0.058  usado=no  despliegues.md#001  Ventanas de despliegue
0.039  usado=no  despliegues.md#000  Ventanas de despliegue
0.032  usado=no  api-publica.md#001  Versionado
0.024  usado=no  despliegues.md#003  Flujo de despliegue
```

Respuestas de `run_rag.py`. Esta corrida usó un servidor local que imita la API de OpenAI y
devuelve textos fijos, así que **la redacción de las respuestas es simulada**. Todo lo demás es
real: la recuperación, el prompt que se envía, el parseo con Pydantic y las fuentes.

```json
{
  "pregunta": "¿Qué pasa si durante el canario la tasa de errores 5xx supera el 1 %?",
  "respuesta": "Se hace rollback automático: si durante el canario la tasa de errores 5xx supera el 1 %, el despliegue vuelve a la versión anterior.",
  "encontrada": true,
  "fuentes": [
    {
      "archivo": "despliegues.md",
      "seccion": "Criterios de rollback",
      "fragmento_id": "despliegues.md#004",
      "relevancia": 0.495
    }
  ]
}
{
  "pregunta": "¿Cuál es la capital de Australia?",
  "respuesta": "No lo sé",
  "encontrada": false,
  "fuentes": []
}
```

```text
INFO rag.chain: Recuperados 2 de 4 fragmentos con relevancia >= 0.45: despliegues.md#004 (0.495), despliegues.md#003 (0.487), api-publica.md#003 (0.447), incidentes.md#000 (0.437)
INFO rag.chain: Recuperados 0 de 4 fragmentos con relevancia >= 0.45: despliegues.md#001 (0.058), ...
INFO rag.chain: Ningún fragmento supera la relevancia mínima: se responde sin llamar al modelo
INFO entity_pipeline.chain: Llamando a gpt-4o-mini
```

Con una API key en el `.env`, `uv run run_rag.py` hace la misma corrida contra el modelo real.

### Preguntas fuera de contexto

Hay dos barreras:

1. **Antes del LLM.** Si ningún fragmento llega a `RAG_MIN_RELEVANCE`, la cadena responde
   "No lo sé" sin llamar al modelo: no gasta tokens y no hay riesgo de que invente. El umbral
   (0.45) se calibró con el modelo de embeddings real y 17 preguntas: las 10 del tema traen su
   mejor fragmento con 0.48 a 0.69. Las 7 ajenas quedan debajo: "¿Cuál es la capital de
   Australia?" (0.06), "¿Cómo configuro Kubernetes en Nimbus?" (0.22) y hasta "¿Cuál es el
   sueldo de un ingeniero de guardia?" (0.44), que comparte palabras con la sección de guardias.
2. **En el LLM.** Si pasa algún fragmento del tema pero no contesta lo que se pregunta, el prompt
   obliga a responder exactamente "No lo sé". La cadena acepta también "No lo sé." con punto, lo
   normaliza, deja `encontrada` en `false` y `fuentes` vacía.

### Contrato Pydantic

- **`SolicitudRAG`** valida la pregunta antes de buscar: sin espacios sobrantes, de 3 a 1000
  caracteres.
- **`RespuestaDelModelo`** es lo que parsea el `PydanticOutputParser`: `respuesta` y
  `fragmentos_usados`. Una respuesta que no es "No lo sé" tiene que citar al menos un fragmento.
  Si el JSON está mal formado, no cita o cita un fragmento que no existe, la cadena reintenta
  (`.with_retry`, hasta `LLM_MAX_ATTEMPTS`).
- **`RespuestaRAG`** es la salida final, con `pregunta`, `respuesta`, `encontrada` y `fuentes`
  (`Fuente`: `archivo`, `seccion`, `fragmento_id`, `relevancia`). Su validador impide una
  respuesta encontrada sin fuentes, o un "No lo sé" con fuentes.

### Decisiones de diseño

**Embeddings locales y el mismo modelo siempre.** Se usa
`paraphrase-multilingual-MiniLM-L12-v2` (fastembed, ONNX), que entiende español, no cuesta
tokens y permite probar la recuperación sin API key. El nombre del modelo queda en los metadatos
de la colección. Si se consulta con otro modelo, `open_vectorstore` corta con
`EmbeddingMismatchError` en lugar de devolver resultados al azar. Cambiar el modelo en el `.env`
cambia la huella, así que la próxima ingesta reindexa sola.

**Persistencia sin trabajo repetido.** La huella combina el contenido de cada documento, el
modelo de embeddings y el tamaño y el solapamiento de los fragmentos. Si coincide con la
guardada, no se vuelve a fragmentar ni a calcular embeddings. Si un documento cambia, la
colección se reconstruye entera, así no quedan fragmentos viejos.

**Las fuentes no dependen del modelo.** El LLM solo dice qué números de fragmento usó. El
archivo, la sección y el ID salen de los metadatos que guardó la ingesta, así una fuente no
puede ser inventada.

**Credenciales.** Las API keys se leen del `.env` como `SecretStr`, que no se imprime ni en los
logs ni en los errores. El `.env`, `./vectorstore` y `.cache/` están en `.gitignore`. Lo prueban
`test_the_chain_runs_against_the_openai_api_without_leaking_the_key` (la key viaja solo en el
encabezado `Authorization`: no aparece en los logs, en la respuesta ni en `repr(settings)`) y
`tests/test_schemas.py`.

### Pruebas

`tests/test_rag_ingest.py`, `tests/test_rag_chain.py` y `tests/test_rag_schemas.py` (38 tests)
corren sin red. Usan Chroma en una carpeta temporal, embeddings de palabras clave y un modelo
falso que solo responde por la vía asíncrona. Cubren:

- el chunking: secciones, IDs, límite de tamaño y solapamiento;
- que una segunda ingesta no reindexa, y que sí reindexa si cambia un documento, con `--rebuild`
  o después de una ingesta que falló;
- el error con otro modelo de embeddings, y los límites de `top_k` (3 a 5);
- las fuentes tomadas de los metadatos, y que el modelo recibe solo el contexto relevante;
- "No lo sé" sin llamar al modelo cuando la pregunta está fuera de contexto;
- que la búsqueda usa la API asíncrona del vector store;
- los reintentos ante JSON inválido, citas inexistentes o un 429 del proveedor;
- las validaciones de `SolicitudRAG` y `RespuestaRAG`;
- una corrida contra la API de OpenAI simulada, con el request que sale.
