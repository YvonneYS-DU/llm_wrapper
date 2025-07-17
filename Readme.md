# LLM Wrapper

A streamlined Python wrapper for LangChain that simplifies AI agent creation with multi-modal support.

## Features

- **Simple API**: Create AI chains with one function call
- **Multi-modal Support**: Handle text and image inputs seamlessly
- **Flexible Output**: String or JSON parsing options
- **Async/Sync Execution**: Both blocking and non-blocking calls
- **Streaming Support**: Real-time response streaming
- **Error Handling**: Built-in retry mechanisms

## Installation

```bash
pip install langchain-openai langchain-anthropic langchain-core
pip install PyMuPDF Pillow  # For image processing
```

## Quick Start

```python
from llm_wrapper import Agents
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Initialize model
llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0, api_key='your-api-key')

# Create chain
chain = Agents.chain_create(
    model=llm,
    text_prompt_template="Answer this question: {question}",
    output_parser=StrOutputParser()
)

# Generate response
response = Agents.chain_batch_generator(chain, {"question": "What is AI?"})
print(response)
```

## Usage Examples

### Text Processing

```python
# Basic text prompt
prompt = "Translate '{text}' to {language}"
chain = Agents.chain_create(llm, prompt)
result = Agents.chain_batch_generator(chain, {"text": "Hello", "language": "French"})
```

### Image Processing

```python
# Image analysis
import base64

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

chain = Agents.chain_create(
    model=llm,
    text_prompt_template="Describe this image: {base64_image}",
    image_prompt_template=True
)

image_b64 = image_to_base64("image.jpg")
result = Agents.chain_batch_generator(chain, {"base64_image": image_b64})
```

### JSON Output

```python
from langchain_core.output_parsers import JsonOutputParser

prompt = """Analyze sentiment and return JSON:
{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.95
}
Text: {text}"""

chain = Agents.chain_create(llm, prompt, output_parser=JsonOutputParser())
result = Agents.chain_batch_generator(chain, {"text": "Great product!"})
```

### Async Processing

```python
import asyncio

async def process_text():
    result = await Agents.chain_batch_generator_async(
        chain, 
        {"text": "Hello world"}, 
        delay=0.5
    )
    return result

response = asyncio.run(process_text())
```

### Streaming Responses

```python
for chunk in Agents.chain_stream_generator(chain, {"text": "Tell me a story"}):
    print(chunk, end="", flush=True)
```

## API Reference

### `Agents.chain_create()`

Creates a LangChain processing chain.

**Parameters:**
- `model`: LLM model instance
- `text_prompt_template`: String template with `{parameter}` placeholders
- `image_prompt_template`: Boolean, enable image processing
- `output_parser`: `StrOutputParser()` or `JsonOutputParser()`
- `image_list`: List of base64 images for multi-image processing

### `Agents.chain_batch_generator()`

Synchronous chain execution with retry logic.

**Parameters:**
- `chain`: Chain object from `chain_create()`
- `dic`: Dictionary of parameters to fill template
- `max_retries`: Maximum retry attempts (default: 2)

### `Agents.chain_batch_generator_async()`

Asynchronous chain execution.

**Parameters:**
- `chain`: Chain object
- `dic`: Parameter dictionary
- `delay`: Optional delay before execution
- `max_retries`: Maximum retry attempts

### `Agents.chain_stream_generator()`

Streaming response generator.

**Parameters:**
- `chain`: Chain object
- `dic`: Parameter dictionary

## Supported Models

- **OpenAI**: GPT-4, GPT-4o, GPT-4o-mini
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Haiku
- **Azure OpenAI**: Compatible with Azure endpoints

## Examples

See `example.ipynb` for comprehensive usage examples including:
- Basic text processing
- Image analysis
- Multi-image processing
- JSON output parsing
- Async/sync execution patterns
- Streaming responses
