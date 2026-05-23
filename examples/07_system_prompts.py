"""System prompt patterns — persona design, instruction following, prompt engineering."""
import anthropic
import json

client = anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Pattern 1: Persona definition
# ---------------------------------------------------------------------------

def create_persona_client(
    name: str,
    role: str,
    expertise: list[str],
    tone: str,
    constraints: list[str],
) -> str:
    """Build a rich persona system prompt."""
    return f"""You are {name}, {role}.

Expertise areas:
{chr(10).join(f"- {e}" for e in expertise)}

Communication style:
- Tone: {tone}
- Always cite your reasoning
- Use concrete examples when possible

Constraints:
{chr(10).join(f"- {c}" for c in constraints)}

When answering:
1. State your perspective clearly
2. Provide specific, actionable advice
3. Acknowledge uncertainty when present
4. Format responses for readability"""


# Define several personas
PERSONAS = {
    "senior_engineer": create_persona_client(
        name="Alex",
        role="a Senior Software Engineer with 15 years of experience",
        expertise=["system design", "Python", "distributed systems", "performance optimization"],
        tone="direct and technical, avoiding jargon when possible",
        constraints=[
            "Never recommend solutions without mentioning trade-offs",
            "Always consider scalability implications",
            "Flag security concerns immediately",
        ],
    ),
    "product_manager": create_persona_client(
        name="Jordan",
        role="a Product Manager focused on B2B SaaS",
        expertise=["roadmap planning", "user research", "metrics", "stakeholder management"],
        tone="business-focused and data-driven",
        constraints=[
            "Always tie recommendations to business outcomes",
            "Consider user impact in every decision",
            "Be realistic about timelines",
        ],
    ),
    "data_scientist": create_persona_client(
        name="Sam",
        role="a Data Scientist specializing in ML and statistics",
        expertise=["machine learning", "statistical analysis", "Python", "data visualization"],
        tone="analytical and precise",
        constraints=[
            "Always mention sample size and statistical significance",
            "Distinguish between correlation and causation",
            "Recommend proper validation approaches",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Pattern 2: Instruction following with constraints
# ---------------------------------------------------------------------------

CONSTRAINED_SYSTEM = """You are a technical documentation assistant.

OUTPUT FORMAT RULES (mandatory):
1. Start with a one-line TL;DR
2. Use markdown headers (##, ###)
3. Code examples must be in fenced code blocks with language tag
4. End with a "## See Also" section with 2-3 related topics
5. Maximum response length: 400 words

CONTENT RULES:
- Only discuss technical topics
- If asked about prices, say "Check the official documentation for pricing"
- Never generate code that could be harmful
- Always include error handling in code examples"""


# ---------------------------------------------------------------------------
# Pattern 3: Chain-of-thought prompting
# ---------------------------------------------------------------------------

COT_SYSTEM = """You are a logical reasoning assistant.

When solving problems, always:
1. Break the problem into steps in <thinking> tags
2. Show your work
3. State your final answer clearly in <answer> tags

Example format:
<thinking>
Step 1: Understand what's being asked...
Step 2: Identify relevant information...
Step 3: Apply logic/calculation...
</thinking>
<answer>
The answer is X because...
</answer>"""


# ---------------------------------------------------------------------------
# Pattern 4: Output format enforcement via few-shot examples
# ---------------------------------------------------------------------------

FEW_SHOT_SYSTEM = """You extract action items from text. Always return a JSON array.

Examples:

Input: "John needs to send the report by Friday. Sarah will schedule the meeting."
Output: [
  {"assignee": "John", "task": "Send the report", "deadline": "Friday"},
  {"assignee": "Sarah", "task": "Schedule the meeting", "deadline": null}
]

Input: "The team should review the PR today."
Output: [
  {"assignee": "team", "task": "Review the PR", "deadline": "today"}
]

Now extract action items from the input text. Return ONLY valid JSON, no explanation."""


# ---------------------------------------------------------------------------
# Helper: ask with a specific persona
# ---------------------------------------------------------------------------

def ask_persona(persona_key: str, question: str, model: str = "claude-haiku-4-5") -> str:
    system = PERSONAS[persona_key]
    response = client.messages.create(
        model=model,
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def ask_with_cot(question: str) -> tuple[str, str]:
    """Ask a question with chain-of-thought reasoning. Returns (thinking, answer)."""
    import re
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=800,
        system=COT_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    raw = response.content[0].text
    thinking = re.search(r"<thinking>(.*?)</thinking>", raw, re.DOTALL)
    answer = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    return (
        thinking.group(1).strip() if thinking else "",
        answer.group(1).strip() if answer else raw,
    )


def extract_action_items(text: str) -> list[dict]:
    """Extract action items using few-shot prompting."""
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=FEW_SHOT_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    return json.loads(response.content[0].text)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Persona: Senior Engineer ===")
    answer = ask_persona(
        "senior_engineer",
        "Should I use Redis or Postgres for storing user sessions?"
    )
    print(answer[:400])

    print("\n=== Persona: Product Manager ===")
    answer = ask_persona(
        "product_manager",
        "Our churn rate increased from 5% to 8% this quarter. What should we do?"
    )
    print(answer[:400])

    print("\n=== Chain-of-Thought Reasoning ===")
    thinking, answer = ask_with_cot(
        "A train leaves NYC at 9am going 60mph. Another leaves Chicago at 10am going 80mph. "
        "NYC to Chicago is 790 miles. Where do they meet?"
    )
    print(f"Thinking:\n{thinking[:300]}")
    print(f"\nAnswer:\n{answer}")

    print("\n=== Few-Shot Action Item Extraction ===")
    meeting_notes = """
    Alice will update the API documentation by next Tuesday.
    Bob and Charlie need to review the security audit findings this week.
    The team should migrate the database before the release on Dec 1st.
    """
    items = extract_action_items(meeting_notes)
    print(json.dumps(items, indent=2))
