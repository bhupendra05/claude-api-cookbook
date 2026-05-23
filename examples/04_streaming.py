"""Streaming with Claude API — real-time token output and token counting."""
import anthropic
import time

client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Method 1: Basic streaming
# ---------------------------------------------------------------------------

def stream_basic(prompt: str, model: str = "claude-haiku-4-5"):
    """Stream a response and print tokens as they arrive."""
    print(f"Streaming response for: '{prompt[:60]}...'")
    print("-" * 60)

    full_response = ""
    start = time.perf_counter()

    with client.messages.stream(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text

    elapsed = time.perf_counter() - start
    print(f"\n\n[Streamed {len(full_response)} chars in {elapsed:.2f}s]")
    return full_response


# ---------------------------------------------------------------------------
# Method 2: Stream with usage stats
# ---------------------------------------------------------------------------

def stream_with_stats(prompt: str, system: str = "", model: str = "claude-sonnet-4-5"):
    """Stream and collect usage statistics."""
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": model, "max_tokens": 1024, "messages": messages}
    if system:
        kwargs["system"] = system

    full_text = ""
    start = time.perf_counter()
    first_token_time = None

    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            if first_token_time is None:
                first_token_time = time.perf_counter() - start
            full_text += text
            print(text, end="", flush=True)

        # Access final message for usage stats
        final_message = stream.get_final_message()

    total_time = time.perf_counter() - start
    input_tokens = final_message.usage.input_tokens
    output_tokens = final_message.usage.output_tokens
    tokens_per_sec = output_tokens / total_time if total_time > 0 else 0

    print(f"\n\n{'='*60}")
    print(f"Time to first token: {first_token_time:.3f}s")
    print(f"Total time:          {total_time:.2f}s")
    print(f"Input tokens:        {input_tokens}")
    print(f"Output tokens:       {output_tokens}")
    print(f"Throughput:          {tokens_per_sec:.1f} tokens/sec")

    return full_text, final_message.usage


# ---------------------------------------------------------------------------
# Method 3: Stream with event hooks
# ---------------------------------------------------------------------------

def stream_with_events(prompt: str):
    """Stream using low-level event iteration for maximum control."""
    print("Streaming with event inspection:")
    print("-" * 60)

    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for event in stream:
            event_type = type(event).__name__

            if event_type == "MessageStartEvent":
                print(f"[START] model={event.message.model}")

            elif event_type == "ContentBlockStartEvent":
                print(f"[BLOCK START] type={event.content_block.type}")

            elif event_type == "ContentBlockDeltaEvent":
                if hasattr(event.delta, "text"):
                    print(event.delta.text, end="", flush=True)

            elif event_type == "MessageDeltaEvent":
                print(f"\n[DELTA] stop_reason={event.delta.stop_reason}")

            elif event_type == "MessageStopEvent":
                print("[STOP]")


# ---------------------------------------------------------------------------
# Method 4: Async streaming
# ---------------------------------------------------------------------------

async def stream_async(prompt: str):
    """Async streaming for use in async applications (FastAPI, etc.)."""
    import anthropic

    async_client = anthropic.AsyncAnthropic()
    full_text = ""

    async with async_client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            full_text += text
            print(text, end="", flush=True)

    print()
    return full_text


if __name__ == "__main__":
    import asyncio

    print("=== Basic Streaming ===")
    stream_basic("Explain Python generators in 3 bullet points.")

    print("\n\n=== Streaming with Stats ===")
    stream_with_stats(
        "Write a short poem about programming.",
        system="You are a creative poet. Keep it under 8 lines."
    )

    print("\n\n=== Stream with Events ===")
    stream_with_events("What is 2 + 2?")

    print("\n\n=== Async Streaming ===")
    asyncio.run(stream_async("Name 3 Python best practices."))
