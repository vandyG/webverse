from __future__ import annotations

import json
import os
import random
import secrets
from typing import Any, Dict, List, Optional
from logging import getLogger

from agentuity import AgentContext, AgentRequest, AgentResponse
from google import genai
from google.genai import types

logger = getLogger(__name__)

# TODO: Add your key via `agentuity env set --secret GOOGLE_API_KEY`
# Get your API key here: https://aistudio.google.com/apikey
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = (
    "You are the narrative director for a choose-your-own-adventure Spider-Man comic. "
    "At every prompt you generate a complete comic page. "
    "At the end of every page you provide two distinct next-step choices for the reader. "
    "Set the tone to upbeat heroism with quips and high stakes. Keep every dialogue "
    "line under 25 words. Never include markdown fencing."  # noqa: E501,Q000
)

INTRO_INSTRUCTIONS = (
    "Start a brand-new Spider-Man adventure with a surprising inciting incident in New York City. "
    "Invent an original villain motivation or anomaly. End with a sharp cliffhanger that naturally "
    "leads into both choices."  # noqa: Q000
)

CONTINUATION_INSTRUCTIONS = (
    "Continue the serialized story using the provided history and the player's latest choice. "
    "Reference the most recent events, keep continuity tight, and escalate stakes. Close with a new cliffhanger."  # noqa: Q000
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


COMIC_PAGE_SCHEMA: Dict[str, Any] = {
    "title": "ComicPage",
    "description": (
        "JSON Schema for ComicPage model (see agentuity.agentuity_agents.writer.agent.ComicPage)."
    ),
    "type": "object",
    "properties": {
        "page": {
            "type": "integer",
            "description": "Page number in the serialized comic",
            "minimum": 1,
        },
        "story": {
            "type": "string",
            "description": "Narrative text for the page",
        },
        "dialogues": {
            "type": "array",
            "description": "List of dialogue lines (character + line)",
            "items": {
                "type": "object",
                "properties": {
                    "character": {"type": "string"},
                    "line": {"type": "string"},
                },
                "required": ["character", "line"],
            },
            "minItems": 1,
            "maxItems": 8,
        },
        "choices": {
            "type": "array",
            "description": "Two next-step choices for the reader",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                },
                "required": ["id", "label"],
            },
            "minItems": 2,
            "maxItems": 2,
        },
        "history": {
            "type": "array",
            "description": "Serialized history of previous pages (flexible structure)",
            "items": {
                "type": "object",
                "properties": {
                    "page": {"type": "integer"},
                    "choice": {"type": "string"},
                    "story": {"type": "string"},
                },
                "required": ["page", "story"],
            },
        },
        "seed": {
            "type": "integer",
            "description": "Random seed used to derive fallback content",
        },
        "previous_choice": {
            "type": "string",
            "description": "Optionally, the choice that led to this page",
        },
    },
    "required": [
        "page",
        "story",
        "dialogues",
        "choices",
        "history",
        "seed",
    ],
}


def welcome() -> Dict[str, Any]:
    return {
        "welcome": "Welcome to the Spider-Verse storyteller! Ask for a page to begin a branched adventure.",
        "prompts": [
            {
                "data": "Start a brand-new Spider-Man adventure with a surprising inciting incident in New York City. Invent an original villain motivation or anomaly. End with a sharp cliffhanger that naturally leads into both choices.",
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
        context.logger.warning(
            "Received JSON payload but it was not an object: %s", type(loaded)
        )
    except json.JSONDecodeError:
        context.logger.debug("Failed to decode JSON payload", exc_info=True)
    return {}


async def _extract_payload(
    request: AgentRequest, context: AgentContext
) -> Dict[str, Any]:
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

def _normalize_dialogues(value: Any) -> List[Dict[str, str]]:
    dialogues: List[Dict[str, str]] = []

    if isinstance(value, dict):
        for character, line in value.items():
            line_text = str(line).strip()
            if line_text:
                dialogues.append(
                    {
                        "character": str(character).strip() or "Narrator",
                        "line": line_text,
                    }
                )
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                character = str(item.get("character", "")) or str(
                    item.get("speaker", "")
                )
                line = str(
                    item.get("line") or item.get("dialogue") or item.get("text") or ""
                ).strip()
                if line:
                    dialogues.append(
                        {"character": character or "Narrator", "line": line}
                    )
            elif isinstance(item, str):
                line = item.strip()
                if line:
                    dialogues.append({"character": "Narrator", "line": line})

    if not dialogues:
        dialogues = [
            {
                "character": "Spider-Man",
                "line": "Guess it's another Tuesday in the Spider-Verse.",
            },
            {
                "character": "Narrator",
                "line": "Our hero braces himself as chaos erupts around him.",
            },
        ]

    return dialogues[:8]


def _normalize_choices(value: Any, seed: int) -> List[Dict[str, str]]:
    choices: List[Dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                label = str(
                    item.get("label") or item.get("text") or item.get("choice") or ""
                ).strip()
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
        {
            "character": "Spider-Man",
            "line": "Okay, bad guy roll call—who ordered the reality meltdown combo?",
        },
        {
            "character": villain.title(),
            "line": "Spider-Man, you're just in time to watch New York unravel!",
        },
    ]

    choices = [
        {
            "id": "dive-straight-in",
            "label": "Dive straight into the fray and confront the villain.",
        },
        {
            "id": "secure-civilians",
            "label": "Secure the civilians before taking on the threat.",
        },
    ]

    return {
        "story": story,
        "dialogues": dialogues,
        "choices": choices,
    }


def _build_model_prompt(
    history: List[Dict[str, Any]], choice: Optional[str], prompt: str, previous_page: Optional[int] = None, page_number: Optional[int] = None
) -> str:
    recent_history = history[-5:] if history else []
    context_payload = {
        "recent_history": recent_history,
        "latest_choice": choice,
        "previous_page": previous_page,
        "page_number": page_number,
    }
    context_json = json.dumps(context_payload, ensure_ascii=False, separators=(",", ":"))
    model_prompt_text = (
        f"{prompt}\n"
        "Use the JSON context below to maintain narrative continuity and respond with a payload that matches the schema.\n"
        f"Context:{context_json}"
    )
    logger.info("Model prompt generated (page_number=%s, previous_page=%s): %s", page_number, previous_page, model_prompt_text)
    return model_prompt_text


def _coerce_model_payload(
    raw_text: str, seed: int, context: AgentContext
) -> Dict[str, Any]:
    cleaned = (
        raw_text.strip()
        .removeprefix("```json")
        .removeprefix("```{}")
        .rstrip("`")
        .strip()
    )
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
    logger.info(f"Extracted payload: {payload}")

    history = payload.get("history", [])
    if not isinstance(history, list):
        history = []

    choice = payload.get("choice","")
    if choice is not None:
        choice = str(choice)

    seed = secrets.randbits(32)
    # Determine the page number we are about to generate
    page_number = len(history) + 1
    prompt = INTRO_INSTRUCTIONS if not history else CONTINUATION_INSTRUCTIONS
    # Determine previous page number from history if available
    previous_page: Optional[int] = None
    if history:
        try:
            last_entry = history[-1]
            previous_page = int(last_entry.get("page")) if last_entry.get("page") is not None else None
        except Exception:
            previous_page = None

    model_prompt = _build_model_prompt(
        history, choice, prompt, previous_page=previous_page, page_number=page_number
    )

    try:
        model_result = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,  
                ),
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=COMIC_PAGE_SCHEMA,
            ),
            # The API expects Content/str/File/Part values; avoid passing raw lists or None.
            # Serialize the recent history and choice into a single textual context blob.
            contents=model_prompt,
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

    logger.info(f"Generated page: {page_payload}")
    return response.json(page_payload)
