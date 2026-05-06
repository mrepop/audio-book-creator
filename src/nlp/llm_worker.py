#!/usr/bin/env python3
"""
LLM Worker Process

Standalone script that runs in the mlx-lm venv (venv-llm/).
Reads JSON requests from stdin, generates context annotations via
Qwen2.5-Instruct, and writes JSON responses to stdout.

Protocol:
  - Each line on stdin is a JSON object: {"sentences": [...], "context": "...", "characters": [...]}
  - Each line on stdout is a JSON object: {"annotations": [...]} or {"error": "..."}
  - Send {"command": "quit"} to exit gracefully.
"""

import sys
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LLM-WORKER] %(message)s",
    stream=sys.stderr,  # Logs go to stderr, data goes to stdout
)
logger = logging.getLogger(__name__)

# The model and tokenizer are loaded once at startup
_model = None
_tokenizer = None
_sampler = None


SYSTEM_PROMPT = """You are an expert audiobook director analyzing prose for text-to-speech production.
You will receive numbered sentences from a novel. For each sentence, provide vocal direction.

Rules:
- "speaker" is the character name who is speaking, or "Narrator" for narration.
- "emotion" is ONE of: neutral, happy, sad, angry, fearful, surprised, tender, excited, contemplative, desperate, solemn, bitter, anxious, relieved, mournful.
- "emphasis" is ONE of: normal, whispered, shouted, soft, strong, urgent.
- "pacing" is ONE of: normal, slow, fast, deliberate.
- "vocal_direction" is a SHORT natural-language instruction (10-20 words) describing HOW to speak this sentence aloud. Focus on tone, vocal quality, and emotional texture. Be specific.

EMOTIONAL CONTINUITY (critical for natural audio):
- Emotions should flow naturally between sentences. Avoid abrupt tonal shifts.
- If the mood changes, the TRANSITIONAL sentence should bridge the gap. For example, if moving from excitement to solemnity, the bridging sentence should "gradually settle" or "let the energy fade" -- not jump instantly.
- When a sentence follows intense emotion (shouting, crying, desperate), the next sentence should carry residual energy -- voices don't reset to neutral instantly.
- Narration between dialogue excerpts should reflect the emotional undertow of the scene, not revert to flat neutral.
- The vocal_direction should reference what came before when relevant: "Still carrying traces of anger, speak with forced composure" is better than just "Speak calmly."

OTHER RULES:
- Dialogue attribution tags (said, cried, whispered) tell you the STYLE of the dialogue, not the narration around it.
- Narration describing someone shouting should still be read in a normal narrative voice -- only the dialogue itself should be shouted.
- Vary your vocal directions. Avoid repeating the same instruction.

Respond with ONLY a JSON array. No other text."""


def build_prompt(sentences, context_text, character_names):
    """Build the user prompt for a context window of sentences."""
    parts = []

    if character_names:
        parts.append(f"Characters in this book: {', '.join(character_names)}")

    if context_text:
        parts.append(f"Preceding context (for continuity, do NOT annotate these):\n{context_text}")

    parts.append("Sentences to analyze:")
    for i, sent in enumerate(sentences):
        tag = "[DIALOGUE]" if sent.get("is_dialogue") else "[NARRATION]"
        parts.append(f'[{i+1}] {tag} "{sent["text"]}"')

    parts.append(
        "\nRespond with a JSON array of objects, one per sentence. "
        "Each object must have keys: index, speaker, emotion, emphasis, pacing, vocal_direction."
    )
    return "\n".join(parts)


def load_model(model_name):
    """Load the MLX model and tokenizer."""
    global _model, _tokenizer, _sampler
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    logger.info(f"Loading model: {model_name}")
    _model, _tokenizer = load(model_name)
    _sampler = make_sampler(temp=0.2, top_p=0.9)
    logger.info("Model loaded successfully")


def generate_annotations(sentences, context_text, character_names):
    """Generate annotations for a window of sentences."""
    from mlx_lm import generate

    user_prompt = build_prompt(sentences, context_text, character_names)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Estimate max tokens: ~80 tokens per sentence annotation
    max_tokens = min(len(sentences) * 120, 4096)

    response = generate(
        _model, _tokenizer, prompt=text,
        max_tokens=max_tokens, sampler=_sampler,
    )

    # Parse JSON from response
    return parse_json_response(response, len(sentences))


def parse_json_response(response, expected_count):
    """Extract and validate JSON array from LLM response."""
    # Try to find JSON array in the response
    text = response.strip()

    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try parsing as-is first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting array from within the text
    import re
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse JSON from LLM response ({len(text)} chars)")
    return None


def main():
    """Main event loop: read requests from stdin, write responses to stdout."""
    # Read model name from first line
    init_line = sys.stdin.readline().strip()
    try:
        init = json.loads(init_line)
        model_name = init.get("model", "mlx-community/Qwen2.5-7B-Instruct-4bit")
    except (json.JSONDecodeError, KeyError):
        model_name = "mlx-community/Qwen2.5-7B-Instruct-4bit"

    try:
        load_model(model_name)
        # Signal ready
        print(json.dumps({"status": "ready"}), flush=True)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {e}"}), flush=True)
        return

    # Process requests
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"error": "Invalid JSON"}), flush=True)
            continue

        if request.get("command") == "quit":
            logger.info("Quit command received, exiting")
            print(json.dumps({"status": "quit"}), flush=True)
            break

        try:
            sentences = request.get("sentences", [])
            context_text = request.get("context", "")
            character_names = request.get("characters", [])

            annotations = generate_annotations(sentences, context_text, character_names)

            if annotations is not None:
                print(json.dumps({"annotations": annotations}), flush=True)
            else:
                print(json.dumps({"error": "Failed to parse LLM output"}), flush=True)

        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
