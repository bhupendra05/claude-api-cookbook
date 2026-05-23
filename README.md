# Claude API Cookbook

Practical, copy-paste-ready examples for the Anthropic Claude API — from basic chat to tool use, vision, streaming, structured output, batch processing, and system prompt engineering.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

Get an API key at [console.anthropic.com](https://console.anthropic.com).

## Examples

| File | What it covers |
|------|---------------|
| `01_basic_chat.py` | Single-turn, multi-turn conversation |
| `02_tool_use.py` | Tool definition, execution loop, multi-tool |
| `03_vision.py` | Image analysis via URL and base64 |
| `04_streaming.py` | Real-time streaming, stats, async |
| `05_structured_output.py` | XML tags, Pydantic, prefill technique |
| `06_batch_api.py` | Message Batches for bulk async processing |
| `07_system_prompts.py` | Personas, CoT, few-shot, constrained output |

Run any example:
```bash
python examples/01_basic_chat.py
```

## Model Comparison

| Model | Best for | Input $/1M | Output $/1M | Context |
|-------|---------|-----------|------------|---------|
| `claude-opus-4-5` | Complex reasoning, vision, coding | $15 | $75 | 200K |
| `claude-sonnet-4-5` | Balanced quality/speed | $3 | $15 | 200K |
| `claude-haiku-4-5` | Fast, cheap, high volume | $0.25 | $1.25 | 200K |

Rule of thumb:
- Use **Haiku** for classification, extraction, simple Q&A
- Use **Sonnet** for most production tasks
- Use **Opus** for complex reasoning, code generation, nuanced analysis

## Common Patterns

### Basic message
```python
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.content[0].text)
print(f"Tokens used: {response.usage.input_tokens} in, {response.usage.output_tokens} out")
```

### Tool use loop
```python
while True:
    response = client.messages.create(model=..., tools=tools, messages=messages)
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
        break
    elif response.stop_reason == "tool_use":
        results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": results})
```

### Streaming
```python
with client.messages.stream(model=..., max_tokens=1024, messages=[...]) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    usage = stream.get_final_message().usage
```

### Image analysis
```python
import base64
with open("image.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}},
        {"type": "text", "text": "What's in this image?"}
    ]}]
)
```

### Batch API (async bulk)
```python
batch = client.beta.messages.batches.create(
    requests=[
        {"custom_id": f"item-{i}", "params": {"model": "claude-haiku-4-5", "max_tokens": 100,
         "messages": [{"role": "user", "content": texts[i]}]}}
        for i in range(len(texts))
    ]
)
# Poll + retrieve results (see 06_batch_api.py for full implementation)
```

## Cost Calculator

Estimate cost for a job:
```python
input_tokens = 500     # avg input tokens per request
output_tokens = 200    # avg output tokens per request
num_requests = 10_000  # total requests

# Claude Haiku prices
input_cost = (input_tokens * num_requests / 1_000_000) * 0.25
output_cost = (output_tokens * num_requests / 1_000_000) * 1.25
total = input_cost + output_cost
print(f"Estimated cost: ${total:.2f}")
# → $2.75 for 10K requests with Haiku
```

Use Batch API for 50% discount on asynchronous workloads.

## Resources

- [Anthropic API Docs](https://docs.anthropic.com)
- [Model Reference](https://docs.anthropic.com/en/docs/about-claude/models)
- [Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Prompt Library](https://docs.anthropic.com/en/prompt-library)
