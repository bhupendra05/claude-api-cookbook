"""Analyze images with Claude vision — URL and base64 methods."""
import anthropic
import base64

client = anthropic.Anthropic()

# From URL
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "url", "url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"}},
            {"type": "text", "text": "What do you see? Be specific."}
        ]
    }]
)
print("URL image:", response.content[0].text)


# From file (base64)
def analyze_image(path: str, prompt: str = "Describe this image.") -> str:
    with open(path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    ext = path.split(".")[-1].lower()
    media_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }.get(ext, "image/jpeg")
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
            {"type": "text", "text": prompt}
        ]}]
    )
    return response.content[0].text


# Multi-image comparison
def compare_images(image_paths: list[str], question: str) -> str:
    """Compare multiple images in a single request."""
    content = []
    for i, path in enumerate(image_paths):
        with open(path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        ext = path.split(".")[-1].lower()
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

        content.append({"type": "text", "text": f"Image {i+1} ({path}):"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": image_data}
        })

    content.append({"type": "text", "text": question})

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        messages=[{"role": "user", "content": content}]
    )
    return response.content[0].text


# Extract structured data from image
def extract_from_image(path: str, schema_description: str) -> str:
    """Extract structured data from an image (e.g., table, form, receipt)."""
    with open(path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    ext = path.split(".")[-1].lower()
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system="You are a data extraction assistant. Extract structured data from images and return valid JSON.",
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
            {"type": "text", "text": f"Extract the following from this image as JSON: {schema_description}"}
        ]}]
    )
    return response.content[0].text


if __name__ == "__main__":
    import os

    # Demo: URL-based analysis (no local file needed)
    print("=== URL Image Analysis ===")
    url_response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
                    }
                },
                {"type": "text", "text": "What animal is this? Describe its appearance in one sentence."}
            ]
        }]
    )
    print(url_response.content[0].text)

    # Show function signatures
    print("\n=== Available Functions ===")
    print("analyze_image(path, prompt)     - Analyze a local image file")
    print("compare_images(paths, question) - Compare multiple images")
    print("extract_from_image(path, schema) - Extract structured data from image")
    print("\nUsage examples:")
    print('  result = analyze_image("screenshot.png", "What errors are shown?")')
    print('  json_data = extract_from_image("receipt.jpg", "total amount, date, vendor name")')
