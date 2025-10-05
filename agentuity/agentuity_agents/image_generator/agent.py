from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Optional, Tuple

from agentuity import AgentContext, AgentRequest, AgentResponse
from google import genai

# TODO: Add your key via `agentuity env set --secret GOOGLE_API_KEY`
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

client = genai.Client(api_key=api_key)

IMAGE_MODEL_NAME = "gemini-2.5-flash-image"

FALLBACK_PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
)


async def _extract_payload(request: AgentRequest, context: AgentContext) -> Dict[str, Any]:
    try:
        payload = await request.data.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        context.logger.debug("Image generator received non-JSON payload", exc_info=True)

    text_payload = await request.data.text()
    if text_payload:
        try:
            decoded = json.loads(text_payload)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            context.logger.warning("Image generator could not parse text payload as JSON")
    return {}


def _compose_prompt(page: Dict[str, Any], illustration: Dict[str, Any]) -> str:
    story = str(page.get("story") or "Spider-Man faces an unexpected threat in New York City.").strip()
    art_direction = str(illustration.get("art_direction") or "Dynamic comic book action from Spider-Man's perspective.").strip()
    color_palette = str(illustration.get("color_palette") or "Bold reds and blues with high-contrast highlights.").strip()
    lighting = str(illustration.get("lighting") or "City twilight glow with dramatic shadows.").strip()
    image_prompt = str(illustration.get("image_prompt") or "Spider-Man swings through Manhattan as energy crackles around him.").strip()

    panel_layout = illustration.get("panel_layout") or []
    if isinstance(panel_layout, list):
        panels = []
        for entry in panel_layout[:5]:
            if isinstance(entry, dict):
                panel_number = entry.get("panel") or len(panels) + 1
                description = str(entry.get("description") or "").strip()
                focus = str(entry.get("focus") or "").strip()
                if description:
                    panels.append(f"Panel {panel_number}: {description} (focus: {focus or 'Spider-Man'})")
            elif isinstance(entry, str) and entry.strip():
                panels.append(f"Panel {len(panels) + 1}: {entry.strip()}")
        panels_text = "\n".join(panels)
    else:
        panels_text = ""

    choice_summary = ""
    choices = page.get("choices") or illustration.get("choices")
    if isinstance(choices, list) and choices:
        formatted_choices = []
        for choice in choices[:2]:
            if isinstance(choice, dict):
                label = str(choice.get("label") or "").strip()
                if label:
                    formatted_choices.append(label)
            elif isinstance(choice, str) and choice.strip():
                formatted_choices.append(choice.strip())
        if formatted_choices:
            choice_summary = "Choices presented: " + " | ".join(formatted_choices)

    prompt_sections = [
        "Spider-Man comic page concept art.",
        f"Story beat: {story}",
        f"Art direction: {art_direction}",
        f"Color palette: {color_palette}",
        f"Lighting: {lighting}",
        f"Primary focus: {image_prompt}",
    ]
    if panels_text:
        prompt_sections.append("Panel breakdown:\n" + panels_text)
    if choice_summary:
        prompt_sections.append(choice_summary)

    prompt_sections.append("Style: dynamic Marvel comic illustration, crisp inks, expressive action, cinematic perspective.")

    return "\n".join(section for section in prompt_sections if section)


def _to_bytes(value: Any) -> Optional[bytes]:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        try:
            return base64.b64decode(value)
        except Exception:
            return value.encode("utf-8")
    return None


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    sanitized: Dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            try:
                value_str = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                value_str = str(value)
        else:
            value_str = str(value)
        value_str = value_str.replace("\r", " ").replace("\n", " ").strip()
        if value_str:
            sanitized[key] = value_str
    return sanitized


def _extract_image_from_response(result: Any, context: AgentContext) -> Tuple[bytes, str]:
    """Extract inline image bytes from a Gemini generate_content response."""

    def _from_inline_data(part: Any) -> Optional[Tuple[bytes, str]]:
        inline = getattr(part, "inline_data", None)
        if inline is None:
            return None
        data_attr = getattr(inline, "data", None)
        if data_attr is None:
            return None
        mime = getattr(inline, "mime_type", None) or "image/png"
        candidate = _to_bytes(data_attr)
        if candidate:
            return candidate, mime
        return None

    candidates = getattr(result, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            extracted = _from_inline_data(part)
            if extracted:
                return extracted
            if hasattr(part, "data") and hasattr(part.data, "data"):
                # Some SDK builds wrap inline data differently.
                extracted = _from_inline_data(part.data)
                if extracted:
                    return extracted
            if hasattr(part, "image"):
                extracted = _from_inline_data(part.image)
                if extracted:
                    return extracted

    top_content = getattr(result, "content", None)
    if top_content is not None:
        parts = getattr(top_content, "parts", None) or []
        for part in parts:
            extracted = _from_inline_data(part)
            if extracted:
                return extracted

    context.logger.warning("Gemini image response missing inline image data; using fallback pixel.")
    return FALLBACK_PIXEL, "image/png"


async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    payload = await _extract_payload(request, context)
    page = payload.get("page") if isinstance(payload.get("page"), dict) else payload
    illustration = payload.get("illustration") if isinstance(payload.get("illustration"), dict) else {}

    if not illustration:
        context.logger.error("Image generator received payload without illustration data.")
        prompt = _compose_prompt(page or {}, {})
    else:
        prompt = _compose_prompt(page or {}, illustration)

    try:
        result = client.models.generate_content(model=IMAGE_MODEL_NAME, contents=prompt)
    except Exception as exc:  # noqa: BLE001
        context.logger.error("Gemini image generation failed: %s", exc, exc_info=True)
        image_bytes, mime_type = FALLBACK_PIXEL, "image/png"
        metadata = {
            "error": str(exc),
            "prompt": prompt,
            "fallback": True,
            "page": page,
            "illustration": illustration,
        }
        return response.png(image_bytes, _sanitize_metadata(metadata))

    image_bytes, mime_type = _extract_image_from_response(result, context)

    metadata = {
        "prompt": prompt,
        "model": IMAGE_MODEL_NAME,
        "page": page,
        "illustration": illustration,
        "fallback": image_bytes == FALLBACK_PIXEL,
    }
    safe_metadata = _sanitize_metadata(metadata)

    if mime_type == "image/png":
        return response.png(image_bytes, safe_metadata)
    if mime_type == "image/jpeg":
        return response.jpeg(image_bytes, safe_metadata)
    if mime_type == "image/webp":
        return response.webp(image_bytes, safe_metadata)

    return response.png(image_bytes, safe_metadata)
