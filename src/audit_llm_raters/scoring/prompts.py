"""Versioned prompt rendering with cache-friendly stable prefixes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Template


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str
    version: str


class PromptLibrary:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def render(
        self,
        rubric_variant: str,
        prompt_variant: str,
        scale_min: int,
        scale_max: int,
        essay_text: str,
        anchors: str = "",
        prompt_id: str | None = None,
    ) -> RenderedPrompt:
        if prompt_variant not in {"zero_shot", "anchored"}:
            raise ValueError(f"Unknown prompt_variant: {prompt_variant}")
        if rubric_variant not in {"brief", "detailed"}:
            raise ValueError(f"Unknown rubric_variant: {rubric_variant}")
        system_path = self.directory / f"system_{rubric_variant}.txt"
        rubric_path = self.directory / f"rubric_{rubric_variant}.txt"
        if prompt_id is not None:
            specific = self.directory / "rubrics" / f"{prompt_id}_{rubric_variant}.txt"
            if specific.exists():
                rubric_path = specific
        system_template = Template(system_path.read_text(encoding="utf-8"))
        rubric = rubric_path.read_text(encoding="utf-8")
        anchor_block = ""
        if prompt_variant == "anchored":
            if not anchors:
                raise ValueError("Anchored prompts require a non-empty anchor block")
            anchor_block = Template(
                (self.directory / "anchors.txt").read_text(encoding="utf-8")
            ).substitute(anchors=anchors)
        system = system_template.substitute(
            scale_min=scale_min,
            scale_max=scale_max,
            rubric=rubric,
            anchor_block=anchor_block,
        )
        user = Template((self.directory / "essay.txt").read_text(encoding="utf-8")).substitute(
            essay_text=essay_text
        )
        digest = hashlib.sha256()
        for path in [system_path, rubric_path, self.directory / "essay.txt"]:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        if anchor_block:
            digest.update(anchor_block.encode())
        return RenderedPrompt(system=system, user=user, version=digest.hexdigest()[:16])


def format_anchors(anchors: list[dict]) -> str:
    blocks = []
    for index, anchor in enumerate(anchors, start=1):
        blocks.append(
            f"Example {index}\nHuman score: {anchor['score']}\nEssay:\n{anchor['essay_text']}"
        )
    return "\n\n".join(blocks)
