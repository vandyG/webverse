from __future__ import annotations

import base64
import json
from typing import Any, Dict

from agentuity import AgentRequest, AgentResponse, AgentContext


async def _extract_payload(
    request: AgentRequest, context: AgentContext
) -> Dict[str, Any]:
    """Return JSON payload from the inbound request when available."""
    try:
        payload = await request.data.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        context.logger.debug("Director received non-JSON request payload", exc_info=True)

    try:
        text_payload = await request.data.text()
    except Exception:
        context.logger.debug("Director request payload not readable as text", exc_info=True)
        return {}

    if text_payload:
        try:
            parsed = json.loads(text_payload)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            context.logger.debug("Director request text payload was not JSON decodable", exc_info=True)
    return {}


async def _invoke_agent(
    context: AgentContext, agent_name: str, payload: Dict[str, Any]
):
    """Call another agent by name with the supplied payload."""
    agent = context.get_agent(agent_name)
    context.logger.debug("Invoking agent '%s' with payload keys: %s", agent_name, list(payload.keys()))
    return await agent.run(payload or {})


async def _serialize_agent_response(
    agent_name: str, agent_response, context: AgentContext
) -> Dict[str, Any]:
    """Normalize a remote agent response into JSON-friendly structure."""
    data = agent_response.data
    metadata = agent_response.metadata or {}
    content_type = getattr(data, "content_type", "application/octet-stream")

    if content_type.startswith("application/json"):
        try:
            body: Any = await data.json()
        except ValueError:
            text = await data.text()
            try:
                body = json.loads(text)
            except Exception:
                context.logger.warning(
                    "%s agent returned JSON content type but payload was not valid JSON; forwarding raw text.",
                    agent_name,
                )
                body = text
    elif content_type.startswith("text/"):
        body = await data.text()
    else:
        binary = await data.binary()
        body = {
            "encoding": "base64",
            "data": base64.b64encode(binary).decode("utf-8"),
            "size": len(binary),
        }

    return {
        "agent": agent_name,
        "content_type": content_type,
        "metadata": metadata,
        "body": body,
    }


async def run(request: AgentRequest, response: AgentResponse, context: AgentContext):
    payload = await _extract_payload(request, context)
    context.logger.info("Director starting orchestration")

    # Stage 1: writer
    try:
        writer_raw = await _invoke_agent(context, "writer", payload)
        writer_result = await _serialize_agent_response("writer", writer_raw, context)
    except Exception as exc:  # noqa: BLE001
        context.logger.error("Writer agent invocation failed: %s", exc, exc_info=True)
        return response.json({"error": "writer_failed", "details": str(exc)})

    page_payload = writer_result.get("body")
    if not isinstance(page_payload, dict):
        context.logger.error("Writer agent returned unsupported payload type: %s", type(page_payload).__name__)
        return response.json(
            {
                "error": "writer_invalid_payload",
                "details": "Writer agent must return a JSON object payload.",
                "writer": writer_result,
            }
        )

    # Stage 2: illustrator
    try:
        illustrator_raw = await _invoke_agent(context, "illustrator", page_payload)
        illustrator_result = await _serialize_agent_response("illustrator", illustrator_raw, context)
    except Exception as exc:  # noqa: BLE001
        context.logger.error("Illustrator agent invocation failed: %s", exc, exc_info=True)
        return response.json(
            {
                "error": "illustrator_failed",
                "details": str(exc),
                "writer": writer_result,
            }
        )

    enriched_payload = illustrator_result.get("body")
    if not isinstance(enriched_payload, dict):
        context.logger.error(
            "Illustrator agent returned unsupported payload type: %s", type(enriched_payload).__name__
        )
        return response.json(
            {
                "error": "illustrator_invalid_payload",
                "details": "Illustrator agent must return a JSON object payload.",
                "writer": writer_result,
                "illustrator": illustrator_result,
            }
        )

    # Stage 3: image generator
    try:
        image_raw = await _invoke_agent(context, "image-generator", enriched_payload)
        image_result = await _serialize_agent_response("image-generator", image_raw, context)
        # raise NotImplementedError("Image generator agent not yet implemented")
    except Exception as exc:  # noqa: BLE001
        context.logger.error("Image generator agent invocation failed: %s", exc, exc_info=True)
        return response.json(
            {
                "error": "image_generator_failed",
                "details": str(exc),
                "writer": writer_result,
                "illustrator": illustrator_result,
            }
        )

    context.logger.info("Director orchestration completed successfully")
    return response.json(
        {
            "writer": writer_result,
            "illustrator": illustrator_result,
            "image": image_result,
        }
    )
