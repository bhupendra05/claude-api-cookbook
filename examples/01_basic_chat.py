"""Basic Claude API usage — single turn and multi-turn conversations."""
import anthropic

client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

# Single turn
message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    system="You are a concise technical assistant.",
    messages=[{"role": "user", "content": "What is async/await in Python? 2 sentences."}],
)
print("Single turn:", message.content[0].text)
print(f"Tokens: input={message.usage.input_tokens}, output={message.usage.output_tokens}")

# Multi-turn conversation
conversation = []
topics = ["What is FastAPI?", "How does it compare to Flask?", "Give me a minimal example."]
for question in topics:
    conversation.append({"role": "user", "content": question})
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        messages=conversation,
    )
    answer = response.content[0].text
    conversation.append({"role": "assistant", "content": answer})
    print(f"\nQ: {question}\nA: {answer[:200]}...")
