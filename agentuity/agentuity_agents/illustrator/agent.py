from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Optional
from logging import getLogger

from google import genai
from google.genai import types

from agentuity import AgentContext, AgentRequest, AgentResponse

logger = getLogger(__name__)

# TODO: Add your key via `agentuity env set --secret GOOGLE_API_KEY`
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)

ILLUSTRATOR_SYSTEM_PROMPT = (
    "You are the cinematic art director for a Spider-Man comic. "
    "Given story context, respond with the illustration of comic page. "
    "Generate a panel layout of up to 3 panels, each with a 'panel' number, description, and focus. "
    "Next give an art direction that guides composition and perspective. "
    "Also provide a 'color_palette' that defines the dominant colors and mood lighting. "
    "Generate an image prompt that is a concise description of the entire page for an image generator. "
    "Finally, suggest up to 4 'sound_effects' (stylized SFX strings). "
    "Keep everything faithful to Spider-Man's tone."  # noqa: E501,Q000
)

COMIC_PAGE_SCHEMA = {
  "title": "IllustrationPageSchema",
  "type": "object",
  "required": [
    "panel_layout",
    "art_direction",
    "color_palette",
    "lighting",
    "image_prompt",
    "sound_effects"
  ],
  "properties": {
    "panel_layout": {
      "type": "array",
      "minItems": 1,
      "maxItems": 3,
      "items": {
        "type": "object",
        "required": ["panel", "description", "focus"],
        "properties": {
          "panel": {
            "type": "integer",
            "minimum": 1,
            "description": "Panel index (1-based)."
          },
          "description": {
            "type": "string",
            "minLength": 1,
            "description": "Concise description of the panel action or scene."
          },
          "focus": {
            "type": "string",
            "minLength": 1,
            "description": "Primary subject or character focus (e.g., 'Spider-Man')."
          }
        }
      }
    },
    "art_direction": {
      "type": "string",
      "minLength": 1,
      "description": "Guidance for composition, perspective, and cinematic staging."
    },
    "color_palette": {
      "type": "string",
      "minLength": 1,
      "description": "Dominant colors and overall mood (e.g., 'vibrant reds, electric blues')."
    },
    "lighting": {
      "type": "string",
      "minLength": 1,
      "description": "Mood lighting description (e.g., 'nocturnal city glow with rim light')."
    },
    "image_prompt": {
      "type": "string",
      "minLength": 1,
      "description": "Succinct prompt for an image generator describing the entire page."
    },
    "sound_effects": {
      "type": "array",
      "minItems": 1,
      "maxItems": 4,
      "items": {
        "type": "string",
        "minLength": 1,
        "maxLength": 24,
        "description": "Stylized SFX string (e.g., 'THWIP!')."
      },
      "description": "Up to four stylized sound-effect strings."
    }
  }
}


async def _extract_page_payload(request: AgentRequest, context: AgentContext) -> Dict[str, Any]:
    try:
        payload = await request.data.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        context.logger.debug("Illustrator received non-JSON payload", exc_info=True)

    text_payload = await request.data.text()
    if text_payload:
        try:
            decoded = json.loads(text_payload)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            context.logger.warning("Illustrator could not parse text payload as JSON")
    return {}


def _build_prompt(page: Dict[str, Any]) -> str:
    condensed_context = {
        "page": page.get("page"),
        "story": page.get("story"),
        "dialogues": page.get("dialogues"),
        "choices": page.get("choices"),
        "previous_choice": page.get("previous_choice"),
        "recent_history": (page.get("history") or [])[-3:],
    }
    context_json = json.dumps(condensed_context, ensure_ascii=False, separators=(",", ":"))
    return (
        f"Context:{context_json}"
    )


def _normalize_panel_layout(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    panels: List[Dict[str, Any]] = []
    for entry in value:
        if isinstance(entry, dict):
            panel_number = entry.get("panel")
            try:
                panel_number = int(panel_number)
            except (TypeError, ValueError):
                panel_number = len(panels) + 1
            description = str(entry.get("description") or entry.get("scene") or "").strip()
            focus = str(entry.get("focus") or entry.get("characters") or "").strip()
            if description:
                panels.append(
                    {
                        "panel": panel_number,
                        "description": description,
                        "focus": focus or "Spider-Man",
                    }
                )
        elif isinstance(entry, str):
            panels.append(
                {
                    "panel": len(panels) + 1,
                    "description": entry.strip(),
                    "focus": "Spider-Man",
                }
            )

    return panels[:5]


def _fallback_illustration(page: Dict[str, Any]) -> Dict[str, Any]:
    story = str(page.get("story") or "Spider-Man springs into action above Manhattan.")
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", story) if segment.strip()]

    random.seed(page.get("seed", random.randrange(1, 10_000)))
    if len(sentences) < 3:
        sentences.extend(["Spider-Man surveys the chaos below.", "A looming threat crackles with energy."])
    panels = []
    for idx, line in enumerate(sentences[:3], start=1):
        panels.append(
            {
                "panel": idx,
                "description": line,
                "focus": "Spider-Man" if "Spider" in line else "Scene action",
            }
        )

    palette_choices = [
        "vibrant reds, electric blues, neon greens",
        "noir shadows with crimson highlights",
        "sunset oranges with stormy purples",
    ]
    lighting_choices = [
        "Dynamic rim lighting with sparks of energy",
        "Nocturnal city glow with reflective webs",
        "Backlit skyline with dramatic spotlight on Spider-Man",
    ]
    sound_effects = ["THWIP!", "KRAKOOM!", "VRRRMMM!"]

    return {
        "panel_layout": panels,
        "art_direction": "Lean into kinetic motion, tilted angles, and close-ups that heighten tension.",
        "color_palette": random.choice(palette_choices),
        "lighting": random.choice(lighting_choices),
        "image_prompt": f"Comic book illustration of Spider-Man in action: {story[:220]}",
        "sound_effects": sound_effects,
    }


def _coerce_model_payload(raw_text: str, page: Dict[str, Any], context: AgentContext) -> Dict[str, Any]:
    cleaned = raw_text.strip().removeprefix("```json").rstrip("`").strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Model response did not return an object")
    except Exception as exc:  # noqa: BLE001
        context.logger.warning("Illustrator model output parsing failed: %s", exc)
        return _fallback_illustration(page)

    fallback_cache: Optional[Dict[str, Any]] = None

    parsed_panel_layout = _normalize_panel_layout(parsed.get("panel_layout"))
    if not parsed_panel_layout:
        context.logger.info("Illustrator model missing panel layout; using fallback panels.")
        fallback_cache = fallback_cache or _fallback_illustration(page)
        parsed_panel_layout = fallback_cache["panel_layout"]

    parsed["panel_layout"] = parsed_panel_layout
    parsed["art_direction"] = str(parsed.get("art_direction") or "Cinematic motion with heroic staging.").strip()
    parsed["color_palette"] = str(parsed.get("color_palette") or "Rich reds and deep blues with energy highlights.").strip()
    parsed["lighting"] = str(parsed.get("lighting") or "High-contrast with streaked city lights.").strip()

    sound_effects = parsed.get("sound_effects")
    if not isinstance(sound_effects, list) or not sound_effects:
        sound_effects = ["THWIP!", "WHOOOSH!"]
    parsed["sound_effects"] = [str(s).upper()[:18] for s in sound_effects[:4]]

    image_prompt = parsed.get("image_prompt")
    if not image_prompt:
        fallback_cache = fallback_cache or _fallback_illustration(page)
        parsed["image_prompt"] = fallback_cache["image_prompt"]

    return parsed


async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    page_payload = await _extract_page_payload(request, context)
    if not page_payload:
        context.logger.error("Illustrator agent received empty page payload; responding with fallback.")
        illustration_payload = _fallback_illustration({})
        return response.json({"page": page_payload, "illustration": illustration_payload})

    prompt = _build_prompt(page_payload)

    try:
        model_result = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,  
                ),
                system_instruction=ILLUSTRATOR_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=COMIC_PAGE_SCHEMA,
            ),
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
        context.logger.error("Illustrator Gemini request failed: %s", exc, exc_info=True)
        illustration_payload = _fallback_illustration(page_payload)
    else:
        illustration_payload = _coerce_model_payload(raw_text, page_payload, context)

    enriched_response = {
        "page": page_payload,
        "illustration": illustration_payload,
    }

    logger.info(f"Enriched response: {enriched_response}")
    return response.json(enriched_response)
