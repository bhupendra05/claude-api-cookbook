"""Structured output from Claude — XML parsing and Pydantic extraction."""
import anthropic
import re
import json
from pydantic import BaseModel, Field
from typing import Optional

client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Method 1: XML tags for structured output
# ---------------------------------------------------------------------------

def extract_xml_tags(text: str, tag: str) -> str:
    """Extract content from XML-like tags."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def analyze_code_xml(code: str) -> dict:
    """Ask Claude to analyze code and return structured XML response."""
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system="""Analyze the provided code and respond in this exact XML format:
<analysis>
  <language>programming language name</language>
  <complexity>low|medium|high</complexity>
  <issues>
    <issue>description of issue 1</issue>
    <issue>description of issue 2</issue>
  </issues>
  <suggestions>
    <suggestion>improvement suggestion 1</suggestion>
    <suggestion>improvement suggestion 2</suggestion>
  </suggestions>
  <summary>one-sentence summary</summary>
</analysis>""",
        messages=[{"role": "user", "content": f"Analyze this code:\n```\n{code}\n```"}],
    )

    raw = response.content[0].text

    # Parse XML response
    return {
        "language": extract_xml_tags(raw, "language"),
        "complexity": extract_xml_tags(raw, "complexity"),
        "issues": re.findall(r"<issue>(.*?)</issue>", raw, re.DOTALL),
        "suggestions": re.findall(r"<suggestion>(.*?)</suggestion>", raw, re.DOTALL),
        "summary": extract_xml_tags(raw, "summary"),
        "raw_response": raw,
    }


# ---------------------------------------------------------------------------
# Method 2: JSON output with Pydantic validation
# ---------------------------------------------------------------------------

class PersonInfo(BaseModel):
    name: str = Field(description="Full name")
    age: Optional[int] = Field(None, description="Age if mentioned")
    occupation: Optional[str] = Field(None, description="Job or role")
    location: Optional[str] = Field(None, description="City or country")
    key_facts: list[str] = Field(default_factory=list, description="Other notable facts")


class ArticleMetadata(BaseModel):
    title: str
    topic: str
    sentiment: str = Field(description="positive, negative, or neutral")
    key_people: list[str] = Field(default_factory=list)
    key_organizations: list[str] = Field(default_factory=list)
    main_points: list[str] = Field(max_length=5)
    word_count_estimate: int


def extract_structured(text: str, model_class: type[BaseModel], context: str = "") -> BaseModel:
    """Extract structured data from text using Claude + Pydantic."""
    schema = model_class.model_json_schema()

    system = f"""Extract information from the provided text and return a JSON object matching this exact schema:
{json.dumps(schema, indent=2)}

Rules:
- Return ONLY valid JSON, no markdown, no explanation
- Use null for missing optional fields
- Be accurate — only include information explicitly stated in the text"""

    messages = [{"role": "user", "content": f"{context}\n\nText to analyze:\n{text}".strip()}]

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=system,
        messages=messages,
    )

    raw = response.content[0].text.strip()

    # Handle markdown code blocks
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?\n?", "", raw).strip().rstrip("`").strip()

    data = json.loads(raw)
    return model_class(**data)


# ---------------------------------------------------------------------------
# Method 3: Prefill technique for guaranteed JSON
# ---------------------------------------------------------------------------

def prefill_json_extraction(text: str, fields: list[str]) -> dict:
    """Use prefilling to force Claude to start with { for guaranteed JSON."""
    fields_desc = ", ".join(f'"{f}"' for f in fields)

    messages = [
        {
            "role": "user",
            "content": f"Extract these fields from the text: {fields_desc}\n\nText: {text}\n\nRespond with a JSON object only."
        },
        {
            "role": "assistant",
            "content": "{"  # Prefill forces Claude to start mid-JSON
        }
    ]

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        messages=messages,
    )

    # Reconstruct full JSON (we prefilled the opening brace)
    json_text = "{" + response.content[0].text.strip()
    return json.loads(json_text)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # XML extraction
    print("=== XML Structured Code Analysis ===")
    sample_code = """
def get_users(db):
    result = db.execute("SELECT * FROM users WHERE id = " + str(id))
    return result
"""
    analysis = analyze_code_xml(sample_code)
    print(f"Language: {analysis['language']}")
    print(f"Complexity: {analysis['complexity']}")
    print(f"Issues: {analysis['issues']}")
    print(f"Summary: {analysis['summary']}")

    # Pydantic extraction
    print("\n=== Pydantic Structured Extraction ===")
    bio = """
    Elon Musk (born 1971) is a South African-born American entrepreneur based in Austin, Texas.
    He is the CEO of Tesla and SpaceX, and owner of X (formerly Twitter).
    He founded The Boring Company and co-founded Neuralink and OpenAI.
    """
    person = extract_structured(bio, PersonInfo)
    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Occupation: {person.occupation}")
    print(f"Location: {person.location}")
    print(f"Key facts: {person.key_facts}")

    # Prefill technique
    print("\n=== Prefill JSON Extraction ===")
    product_text = "The iPhone 15 Pro costs $999. It has a titanium body and was released in September 2023."
    data = prefill_json_extraction(product_text, ["product_name", "price", "material", "release_date"])
    print(json.dumps(data, indent=2))
