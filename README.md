# webverse

[![ci](https://github.com/vandyG/webverse/workflows/ci/badge.svg)](https://github.com/vandyG/webverse/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://vandyG.github.io/webverse/)
[![pypi version](https://img.shields.io/pypi/v/webverse.svg)](https://pypi.org/project/webverse/)

Spin your own Spider-Man style "Choose Your Own Adventure" digital comic with voice narration.

This repository contains the core Python libraries and agent framework used to generate episodic, branching comic-story experiences. It includes a small collection of autonomous agents grouped under `agentuity` which orchestrate story direction, illustration, and text/voice generation. The frontend UI that serves as a beautiful comic-book style platform for interacting with these agents is hosted in a separate repository: https://github.com/PratyakshMathur/webverse-userinterface

## Key components

- `webverse/` - Core Python package with CLI and utilities.
- `agentuity/` - A lightweight agent framework and a set of agents that together compose story generation workflows.
- `agentuity_agents/` - Pre-built agents (director, writer, illustrator, image_generator) demonstrating coordinated agents for story generation.
- `docs/` - Project documentation and API reference.
- `tests/` - Unit tests and CI configuration.

## Frontend (webverse-userinterface)

The web-based frontend is intentionally split into its own repo: https://github.com/PratyakshMathur/webverse-userinterface. It provides a polished comic-book UI for:

- Creating new episodes and branching choices.
- Previewing illustrated panels and generated narration audio.
- Playing narrated scenes with synchronized panel transitions.

If you want a full end-to-end experience, run the backend agents from this repo and pair them with the frontend repo which connects to the agents via REST or websocket endpoints (see `agentuity/server.py` and `agentuity/main.py` for examples).

## Installation

Prerequisites: Python 3.11+ (the project uses recent typing and async features). Recommended: use a virtualenv or `nix`/`poetry` as provided by `pyproject.toml` and `shell.nix`.

Basic install:

```bash
pip install webverse
```

With [`uv`](https://docs.astral.sh/uv/):
```bash
uv tool install webverse
```

## Quickstart (agentuity)

The `agentuity` folder contains an example lightweight agent orchestration system. You can run the agents locally to produce story content that can be consumed by the frontend UI.

Run a simple server that exposes agent endpoints (example):

```bash
python -m agentuity.server
```

Or run the example agent driver:

```bash
python agentuity/main.py
```

See `agentuity/AGENTS.md` for more usage details and configuration options.

## agentuity agents — detailed overview

The `agentuity_agents/` package in this repository demonstrates a small set of cooperating agents that together generate branching comic stories with illustrations and narration. They are intentionally simple to be easy to extend.

High-level flow:

1. Director: outlines the episode beats and high-level branching choices.
2. Writer: takes the director's beats and creates scene scripts, dialogue, and narration text.
3. Illustrator: converts scene descriptions into panel prompts and arranges comic panels.
4. Image Generator: (optional) takes illustration prompts and produces raster images (e.g., via a local or remote image generation model).

Agent details

- `director/agent.py`
	- Role: Creates an overall episode structure, major beats, and branching points.
	- Inputs: seeds, themes, or a simple prompt (e.g., "Make a Spider-Man adventure about responsibility and technology").
	- Outputs: JSON structure describing ordered beats and branching choices (labels, target beats).

- `writer/agent.py`
	- Role: Expands beats into scene-level scripts, panel captions, dialogue lines, and narration text. Also responsible for generating voice narration scripts and metadata such as timing.
	- Inputs: beats from the director and optional style/voice parameters.
	- Outputs: structured scene objects containing per-panel text, narrator copy, and suggested timings.

- `illustrator/agent.py`
	- Role: Converts scene and panel text into illustration prompts and page/panel layout metadata. May also produce caption placement and speech-bubble hints for the frontend.
	- Inputs: scene objects from the writer.
	- Outputs: prompt objects and layout JSON describing panels, focal points, and style notes.

- `image_generator/agent.py`
	- Role: (Optional) Interface to an image generation back-end or service to produce raster images from illustration prompts. This agent is kept modular so you can plug in API-based generators or local models.
	- Inputs: illustration prompts and layout metadata.
	- Outputs: image assets (URLs or local file paths) and metadata (dimensions, format).

Extending agents

The agents are intentionally small and easy to replace. Typical extensions include:

- Adding more sophisticated branching logic in `director`.
- Using language models or fine-tuned prompts in `writer` for richer, genre-specific prose.
- Integrating high-fidelity image models in `image_generator` for consistent character appearance.
- Adding a `sound` agent to mix background music and more advanced audio cues.

Examples

The repo contains minimal example code in `agentuity/main.py` and `agentuity/server.py` demonstrating how to wire these agents together and expose simple endpoints for the frontend to call. See `agentuity/AGENTS.md` for usage examples and configuration settings.

## Contributing

We welcome contributions. See `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` for guidelines. If you want to add a new agent, follow these steps:

1. Add your agent implementation under `agentuity_agents/<your_agent_name>/`.
2. Add unit tests in `tests/` covering inputs and outputs.
3. Update `agentuity/AGENTS.md` with usage instructions and configuration.

## Testing

Run tests with pytest:

```bash
pytest -q
```

## License

This project is released under the MIT license. See `LICENSE` for details.

## Thanks & credits

If you use or adapt the frontend UI, please give credit to the `webverse-userinterface` project by linking to https://github.com/PratyakshMathur/webverse-userinterface in your app and README.

---

If you'd like, I can also expand `agentuity/AGENTS.md` with an agent API spec or add examples that show the JSON shapes exchanged between agents and the frontend. Tell me which you'd prefer next.

## Demo

Watch a short demo of Webverse in action. If your Git hosting or viewer supports embedded HTML, the video will play inline. Otherwise you can download and view the file locally.

<video controls width="720">
	<source src="Webverse Demo.mp4" type="video/mp4">
	Your browser does not support the video tag. You can download the demo here:
	<a href="Webverse Demo.mp4">Download Webverse Demo.mp4</a>
</video>

