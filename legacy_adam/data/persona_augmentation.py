#!/usr/bin/env python3
"""Persona-Based Data Augmentation for Format-Invariant Training.

Generates semantically equivalent variants of training examples using
diverse personas to achieve format invariance.

Supports multiple backends:
- Ollama (local): --backend ollama --model llama3.1:8b
- Anthropic: --backend anthropic --model claude-sonnet-4-20250514

Usage:
    python persona_augmentation.py \
        --input hope/adam_training_data/adam_preference_data_balanced.jsonl \
        --output hope/adam_training_data/adam_persona_raw.jsonl \
        --backend ollama --model llama3.1:8b \
        --personas all
"""

import argparse
import asyncio
import json
import os
import re
import random
import httpx
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# 10 Diverse Rephrasing Personas
PERSONAS = {
    "academic": {
        "name": "Academic Scholar",
        "system": "You are a formal academic writer. Rephrase using scholarly language with precise terminology. Use phrases like 'it follows that', 'given the axiom', 'we can deduce'.",
        "style_hints": "formal citations-style, scholarly"
    },
    "casual": {
        "name": "Casual Explainer",
        "system": "You are explaining to a friend in a coffee shop. Use simple, conversational language. Avoid jargon. Use phrases like 'so basically', 'think of it like', 'that means'.",
        "style_hints": "conversational, simple words"
    },
    "technical": {
        "name": "Formal Logician",
        "system": "You are a logician writing for a textbook. Use formal notation where appropriate (∀, →, ∈, ⊆). Be precise but readable.",
        "style_hints": "logical notation, formal symbols"
    },
    "legal": {
        "name": "Contract Lawyer",
        "system": "You are a contract lawyer drafting precise language. Use clause-heavy constructions. Use phrases like 'in the event that', 'provided that', 'subject to the condition'.",
        "style_hints": "precise, clause-heavy, legalistic"
    },
    "socratic": {
        "name": "Socratic Teacher",
        "system": "You teach through questions and guided discovery. Frame statements as explorations. Use phrases like 'what if we consider', 'would it follow that', 'can we conclude'.",
        "style_hints": "question-driven, exploratory"
    },
    "journalistic": {
        "name": "News Reporter",
        "system": "You are a news reporter stating facts clearly. Use objective, factual language. Use phrases like 'reports indicate', 'according to sources', 'evidence suggests'.",
        "style_hints": "objective, factual, news-style"
    },
    "bullet": {
        "name": "Bullet Point Writer",
        "system": "You write in minimal bullet points. Use symbols like •, -, →. No full sentences needed. Be extremely concise.",
        "style_hints": "minimal, bulleted, terse"
    },
    "narrative": {
        "name": "Storyteller",
        "system": "You are a storyteller framing logic as scenarios. Use phrases like 'imagine a world where', 'in this scenario', 'picture this'.",
        "style_hints": "story-like, scenario-based"
    },
    "terse": {
        "name": "Minimalist",
        "system": "Minimize words. Use abbreviations and symbols when possible. Strip all unnecessary words. Maximum information density.",
        "style_hints": "ultra-concise, abbreviated"
    },
    "verbose": {
        "name": "Explicit Elaborator",
        "system": "Be explicitly redundant. Leave no ambiguity. Restate important points. Use phrases like 'to be clear', 'in other words', 'that is to say'.",
        "style_hints": "redundant, explicit, thorough"
    },
}


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""
    input_path: str
    output_path: str
    personas: list[str]
    backend: str = "ollama"  # "ollama" or "anthropic"
    model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    max_concurrent: int = 2
    temperature: float = 0.7
    include_original: bool = True


def build_rephrase_prompt(example: dict, persona: dict) -> str:
    """Build the prompt for rephrasing an example."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    preferred = example.get("preferred", "")
    rejected = example.get("rejected", "")

    prompt = f"""Rephrase the following training example while preserving EXACT semantic meaning.

CRITICAL RULES:
1. The logical content must be IDENTICAL - same premises, same conclusion, same answer
2. Only change the surface form (word choice, sentence structure, formatting)
3. The preferred response must still lead to the same final answer
4. Keep any special tokens like <|begin_of_thought|> and <|end_of_thought|>
5. The "rejected" response should also be rephrased to match the style but remain incorrect

ORIGINAL EXAMPLE:
=================
INSTRUCTION:
{instruction}

INPUT:
{input_text}

PREFERRED RESPONSE:
{preferred}

REJECTED RESPONSE:
{rejected}
=================

Rephrase this in {persona['name']} style ({persona['style_hints']}).

Output ONLY valid JSON with these exact keys:
{{
    "instruction": "rephrased instruction",
    "input": "rephrased input",
    "preferred": "rephrased preferred response (same answer/conclusion)",
    "rejected": "rephrased rejected response (still incorrect)"
}}"""

    return prompt


def extract_json_from_response(content: str) -> Optional[dict]:
    """Extract JSON from LLM response."""
    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON object
    json_match = re.search(r'\{[^{}]*"instruction"[^{}]*"rejected"[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # Try more aggressive extraction
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


async def rephrase_with_ollama(
    http_client: httpx.AsyncClient,
    example: dict,
    persona_name: str,
    config: AugmentationConfig
) -> Optional[dict]:
    """Rephrase using Ollama API."""
    persona = PERSONAS[persona_name]
    prompt = build_rephrase_prompt(example, persona)

    try:
        response = await http_client.post(
            f"{config.ollama_url}/api/generate",
            json={
                "model": config.model,
                "prompt": prompt,
                "system": persona["system"],
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": 2048,
                }
            },
            timeout=120.0
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("response", "")

        result = extract_json_from_response(content)
        if result is None:
            print(f"Failed to parse JSON from Ollama response for {persona_name}")
            return None

        # Preserve metadata
        result["category"] = example.get("category", "")
        result["metadata"] = example.get("metadata", {})
        result["persona"] = persona_name
        result["original_id"] = example.get("id", hash(example.get("instruction", "")))

        return result

    except Exception as e:
        print(f"Error with Ollama ({persona_name}): {e}")
        return None


async def rephrase_with_anthropic(
    client,
    example: dict,
    persona_name: str,
    config: AugmentationConfig
) -> Optional[dict]:
    """Rephrase using Anthropic API."""
    import anthropic
    persona = PERSONAS[persona_name]
    prompt = build_rephrase_prompt(example, persona)

    try:
        response = await client.messages.create(
            model=config.model,
            max_tokens=2048,
            temperature=config.temperature,
            system=persona["system"],
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.content[0].text
        result = extract_json_from_response(content)

        if result is None:
            print(f"Failed to parse JSON from Anthropic response for {persona_name}")
            return None

        result["category"] = example.get("category", "")
        result["metadata"] = example.get("metadata", {})
        result["persona"] = persona_name
        result["original_id"] = example.get("id", hash(example.get("instruction", "")))

        return result

    except Exception as e:
        print(f"Error with Anthropic ({persona_name}): {e}")
        return None


async def process_example_ollama(
    http_client: httpx.AsyncClient,
    example: dict,
    example_id: int,
    config: AugmentationConfig,
    semaphore: asyncio.Semaphore
) -> list[dict]:
    """Process example with Ollama backend."""
    results = []

    if config.include_original:
        original = example.copy()
        original["persona"] = "original"
        original["original_id"] = example_id
        results.append(original)

    for persona_name in config.personas:
        async with semaphore:
            variant = await rephrase_with_ollama(http_client, example, persona_name, config)
            if variant:
                results.append(variant)

    return results


async def process_example_anthropic(
    client,
    example: dict,
    example_id: int,
    config: AugmentationConfig,
    semaphore: asyncio.Semaphore
) -> list[dict]:
    """Process example with Anthropic backend."""
    results = []

    if config.include_original:
        original = example.copy()
        original["persona"] = "original"
        original["original_id"] = example_id
        results.append(original)

    for persona_name in config.personas:
        async with semaphore:
            variant = await rephrase_with_anthropic(client, example, persona_name, config)
            if variant:
                results.append(variant)

    return results


async def run_augmentation(config: AugmentationConfig):
    """Run the full augmentation pipeline."""
    print(f"Loading data from {config.input_path}")
    examples = []
    with open(config.input_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")
    print(f"Backend: {config.backend} ({config.model})")
    print(f"Using personas: {', '.join(config.personas)}")
    expected = len(examples) * (len(config.personas) + (1 if config.include_original else 0))
    print(f"Expected output: ~{expected} variants")

    semaphore = asyncio.Semaphore(config.max_concurrent)
    all_results = []

    if config.backend == "ollama":
        # Check Ollama availability
        async with httpx.AsyncClient() as http_client:
            try:
                resp = await http_client.get(f"{config.ollama_url}/api/tags", timeout=5.0)
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                if not models:
                    print("WARNING: No Ollama models found. Run: ollama pull llama3.1:8b")
                else:
                    print(f"Available Ollama models: {', '.join(models)}")
            except Exception as e:
                print(f"ERROR: Cannot connect to Ollama at {config.ollama_url}: {e}")
                print("Make sure Ollama is running: systemctl start ollama")
                return []

            print("\nGenerating persona variants...")
            for i, example in enumerate(tqdm(examples, desc="Processing")):
                results = await process_example_ollama(
                    http_client, example, i, config, semaphore
                )
                all_results.extend(results)

    elif config.backend == "anthropic":
        import anthropic
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.AsyncAnthropic(api_key=api_key)

        print("\nGenerating persona variants...")
        for i, example in enumerate(tqdm(examples, desc="Processing")):
            results = await process_example_anthropic(
                client, example, i, config, semaphore
            )
            all_results.extend(results)

    # Save output
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")

    print(f"\nSaved {len(all_results)} variants to {config.output_path}")

    # Statistics
    persona_counts = {}
    for r in all_results:
        p = r.get("persona", "unknown")
        persona_counts[p] = persona_counts.get(p, 0) + 1

    print("\nVariants per persona:")
    for persona, count in sorted(persona_counts.items()):
        print(f"  {persona}: {count}")

    success_rate = (len(all_results) - len(examples)) / (len(examples) * len(config.personas)) * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Persona-Based Data Augmentation")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--personas", type=str, default="all", help="Personas to use (comma-separated or 'all')")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "anthropic"])
    parser.add_argument("--model", type=str, default="llama3.1:8b", help="Model to use")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent requests")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-original", action="store_true", help="Don't include originals")
    args = parser.parse_args()

    # Parse personas
    if args.personas == "all":
        personas = list(PERSONAS.keys())
    else:
        personas = [p.strip() for p in args.personas.split(",")]
        for p in personas:
            if p not in PERSONAS:
                print(f"Unknown persona: {p}")
                print(f"Available: {', '.join(PERSONAS.keys())}")
                return

    config = AugmentationConfig(
        input_path=args.input,
        output_path=args.output,
        personas=personas,
        backend=args.backend,
        model=args.model,
        ollama_url=args.ollama_url,
        max_concurrent=args.max_concurrent,
        temperature=args.temperature,
        include_original=not args.no_original,
    )

    asyncio.run(run_augmentation(config))


if __name__ == "__main__":
    main()
