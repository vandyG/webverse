from __future__ import annotations

import json
import os
import random
import secrets
from typing import Any, Dict, List, Optional

from agentuity import AgentContext, AgentRequest, AgentResponse
from google import genai

# TODO: Add your key via `agentuity env set --secret GOOGLE_API_KEY`
# Get your API key here: https://aistudio.google.com/apikey
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = (
    "You are the narrative director for a choose-your-own-adventure Spider-Man comic. "
    "Always answer with a single JSON object that matches this schema: {"  # noqa: Q000
    "'story': string describing the cinematic scene in 3-5 sentences, "
    "'dialogues': list of objects with keys 'character' and 'line', and "
    "'choices': list of exactly two objects with keys 'id' (kebab-case) and 'label'. "
    "Set the tone to upbeat heroism with quips and high stakes. Keep every dialogue "
    "line under 25 words. Never include markdown fencing or commentary outside the JSON object."  # noqa: E501,Q000
)

INTRO_INSTRUCTIONS = (
    "Start a brand-new Spider-Man adventure with a surprising inciting incident in New York City. "
    "Invent an original villain motivation or anomaly. End with a sharp cliffhanger that naturally "
    "leads into both choices."  # noqa: Q000
)

CONTINUATION_INSTRUCTIONS = (
    "Continue the serialized story using the provided history and the player's latest choice. "
    "Reference the most recent events, keep continuity tight, and escalate stakes. Close with "
    "a new cliffhanger that matches both next-step choices."  # noqa: Q000
)

FALLBACK_VILLAINS = [
    "the Lizard",
    "Doctor Octopus",
    "Electro",
    "the Green Goblin",
    "Mysterio",
    "the Vulture",
]

FALLBACK_LOCATIONS = [
    "Times Square",
    "the Brooklyn Bridge",
    "Queens rooftops",
    "a S.H.I.E.L.D. safehouse in Hell's Kitchen",
    "Grand Central Terminal",
    "the New York Public Library",
]

FALLBACK_COMPLICATIONS = [
    "a collapsing hovercraft",
    "an unstable quantum rift",
    "civilians caught in a gravity storm",
    "a swarm of rogue spider-bots",
    "an EMP pulse knocking out city power",
    "dimensional echoes tearing open the sky",
]


def welcome() -> Dict[str, Any]:
    return {
        "welcome": "Welcome to the Spider-Verse storyteller! Ask for a page to begin a branched adventure.",
        "prompts": [
            {
                "data": "Start a new Spider-Man comic adventure page",
                "contentType": "text/plain",
            },
            {
                "data": "Continue the story after choosing to swing toward the green lightning",
                "contentType": "text/plain",
            },
        ],
    }


def _safe_json_loads(raw: str, context: AgentContext) -> Dict[str, Any]:
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return loaded
        context.logger.warning("Received JSON payload but it was not an object: %s", type(loaded))
    except json.JSONDecodeError:
        context.logger.debug("Failed to decode JSON payload", exc_info=True)
    return {}


async def _extract_payload(request: AgentRequest, context: AgentContext) -> Dict[str, Any]:
    # Prefer JSON payload when available, but gracefully fallback to text.
    try:
        payload = await request.data.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        context.logger.debug("Request payload is not structured JSON", exc_info=True)

    text_payload = await request.data.text()
    if text_payload:
        return _safe_json_loads(text_payload, context)
    return {}


def _build_prompt(history: List[Dict[str, Any]], choice: Optional[str], seed: int) -> str:
    request_frame: Dict[str, Any] = {
        "random_seed": seed,
        "history": history[-5:],  # keep prompt brief but with enough context
        "latest_choice": choice,
        "request_type": "intro" if not history else "continuation",
    }

    contextual_instructions = INTRO_INSTRUCTIONS if not history else CONTINUATION_INSTRUCTIONS
    payload_json = json.dumps(request_frame, ensure_ascii=False, separators=(",", ":"))

    return (
        f"{SYSTEM_PROMPT}\n"  # noqa: Q000
        f"Guidance: {contextual_instructions}\n"
        "Use the JSON below as your context and craft the next page."
        f"\nContext:{payload_json}"
    )


def _normalize_dialogues(value: Any) -> List[Dict[str, str]]:
    dialogues: List[Dict[str, str]] = []

    if isinstance(value, dict):
        for character, line in value.items():
            line_text = str(line).strip()
            if line_text:
                dialogues.append({"character": str(character).strip() or "Narrator", "line": line_text})
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                character = str(item.get("character", "")) or str(item.get("speaker", ""))
                line = str(item.get("line") or item.get("dialogue") or item.get("text") or "").strip()
                if line:
                    dialogues.append({"character": character or "Narrator", "line": line})
            elif isinstance(item, str):
                line = item.strip()
                if line:
                    dialogues.append({"character": "Narrator", "line": line})

    if not dialogues:
        dialogues = [
            {"character": "Spider-Man", "line": "Guess it's another Tuesday in the Spider-Verse."},
            {"character": "Narrator", "line": "Our hero braces himself as chaos erupts around him."},
        ]

    return dialogues[:8]


def _normalize_choices(value: Any, seed: int) -> List[Dict[str, str]]:
    choices: List[Dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("text") or item.get("choice") or "").strip()
                choice_id = str(item.get("id") or item.get("slug") or "").strip()
                if label:
                    if not choice_id:
                        choice_id = label.lower().replace(" ", "-").replace("'", "")
                    choices.append({"id": choice_id[:64], "label": label})
            elif isinstance(item, str):
                label = item.strip()
                if label:
                    choice_id = label.lower().replace(" ", "-").replace("'", "")
                    choices.append({"id": choice_id[:64], "label": label})

    if len(choices) >= 2:
        return choices[:2]

    random.seed(seed)
    fallback_pairs = [
        ("swing-right-into-chaos", "Swing toward the source of the disturbance."),
        ("shadow-trail", "Stay hidden and trail the villain through the shadows."),
        ("shield-civilians", "Web up a barrier and protect the civilians first."),
        ("tech-diagnosis", "Scan the strange device with your suit's sensors."),
    ]
    random.shuffle(fallback_pairs)
    while len(choices) < 2 and fallback_pairs:
        cid, label = fallback_pairs.pop()
        choices.append({"id": cid, "label": label})

    return choices[:2]


def _fallback_story(seed: int) -> Dict[str, Any]:
    random.seed(seed)
    villain = random.choice(FALLBACK_VILLAINS)
    location = random.choice(FALLBACK_LOCATIONS)
    complication = random.choice(FALLBACK_COMPLICATIONS)

    story = (
        f"Spider-Man swings above {location} when he spots {villain} orchestrating {complication}. "
        "With sirens blaring below, Spidey cracks a joke to calm the nerves—even his own—before "
        "he dives into danger."  # noqa: Q000
    )

    dialogues = [
        {"character": "Spider-Man", "line": "Okay, bad guy roll call—who ordered the reality meltdown combo?"},
        {"character": villain.title(), "line": "Spider-Man, you're just in time to watch New York unravel!"},
    ]

    choices = [
        {"id": "dive-straight-in", "label": "Dive straight into the fray and confront the villain."},
        {"id": "secure-civilians", "label": "Secure the civilians before taking on the threat."},
    ]

    return {
        "story": story,
        "dialogues": dialogues,
        "choices": choices,
    }


def _coerce_model_payload(raw_text: str, seed: int, context: AgentContext) -> Dict[str, Any]:
    cleaned = raw_text.strip().removeprefix("```json").removeprefix("```{}").rstrip("`").strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Model response was not a JSON object")
    except Exception as exc:  # noqa: BLE001
        context.logger.warning("Falling back due to model parse failure: %s", exc)
        return _fallback_story(seed)

    story_text = str(parsed.get("story", "")).strip()
    if not story_text:
        context.logger.info("Model omitted story text; using fallback narrative.")
        parsed.update(_fallback_story(seed))
        story_text = parsed["story"]

    parsed["story"] = story_text
    parsed["dialogues"] = _normalize_dialogues(parsed.get("dialogues"))
    parsed["choices"] = _normalize_choices(parsed.get("choices"), seed)
    return parsed


async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    payload = await _extract_payload(request, context)

    history = payload.get("history")
    if not isinstance(history, list):
        history = []

    choice = payload.get("choice")
    if choice is not None:
        choice = str(choice)

    seed = secrets.randbits(32)
    prompt = _build_prompt(history, choice, seed)

    try:
        model_result = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        raw_text = getattr(model_result, "text", "") or ""
        if not raw_text and getattr(model_result, "candidates", None):
            fragments: List[str] = []
            for candidate in getattr(model_result, "candidates", []):
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None)
                if not parts:
                    continue
                for part in parts:
                    text_value = getattr(part, "text", None)
                    if text_value:
                        fragments.append(str(text_value))
            raw_text = "\n".join(fragments)
    except Exception as exc:  # noqa: BLE001
        context.logger.error("Gemini request failed: %s", exc, exc_info=True)
        page_payload = _fallback_story(seed)
    else:
        page_payload = _coerce_model_payload(raw_text, seed, context)

    page_number = len(history) + 1
    updated_history = list(history)
    updated_history.append(
        {
            "page": page_number,
            "choice": choice,
            "story": page_payload["story"],
        }
    )

    page_payload.update(
        {
            "page": page_number,
            "history": updated_history,
            "seed": seed,
        }
    )
    if choice:
        page_payload["previous_choice"] = choice

    response.json(page_payload)
    return response.handoff(
        params={"name": "illustrator"},
        args=page_payload,
    )