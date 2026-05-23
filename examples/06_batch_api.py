"""Claude Message Batches API — process thousands of messages asynchronously."""
import anthropic
import time

client = anthropic.Anthropic()

# Create a batch
batch = client.beta.messages.batches.create(
    requests=[
        {"custom_id": f"task-{i}", "params": {
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": f"What is {i} * {i}? Answer with just the number."}]
        }}
        for i in range(1, 11)
    ]
)
print(f"Batch created: {batch.id}, status: {batch.processing_status}")

# Poll until done
while batch.processing_status == "in_progress":
    time.sleep(5)
    batch = client.beta.messages.batches.retrieve(batch.id)
    print(f"Status: {batch.processing_status}")

# Retrieve results
for result in client.beta.messages.batches.results(batch.id):
    if result.result.type == "succeeded":
        print(f"  {result.custom_id}: {result.result.message.content[0].text}")
