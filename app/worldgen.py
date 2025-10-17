from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple
import random


@dataclass
class World:
    title: str
    description: str


PROMPT = (
    "You are a world builder. Given three sliders biology, technology, culture with values between 0 and 1, "
    "invent a concise speculative world. Provide a short title (max 5 words) and a 3-4 sentence description. "
    "biology ~0 is barren/sterile, ~1 is lush/organic/alien. technology ~0 is primitive, ~1 is advanced/AI/space. "
    "culture ~0 is chaotic/tribal, ~1 is bureaucratic/meritocratic. Return plain text with the title on first line, "
    "then a blank line, then the description."
)


def generate_world(bio: float, tech: float, culture: float) -> World:
    # Try Gemini if API key present
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai

            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"{PROMPT}\nValues: biology={bio:.2f}, technology={tech:.2f}, culture={culture:.2f}"
            resp = model.generate_content(prompt)
            text = resp.text or ""
            return parse_text_to_world(text)
        except Exception:
            pass
    if openai_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key)
            content = f"{PROMPT}\nValues: biology={bio:.2f}, technology={tech:.2f}, culture={culture:.2f}"
            chat = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": content}], temperature=0.8)
            text = chat.choices[0].message.content
            return parse_text_to_world(text)
        except Exception:
            pass
    # Fallback: deterministic template
    return fallback_world(bio, tech, culture)


def quick_preview_world(bio: float, tech: float, culture: float) -> World:
    """Super fast, offline preview used while the user is pinching/adjusting.
    This intentionally avoids any network/model calls to keep the UI snappy.
    """
    return fallback_world(bio, tech, culture)


def parse_text_to_world(text: str) -> World:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return World("Unnamed World", "No description.")
    title = lines[0][:60]
    desc = " ".join(lines[1:])[:700]
    return World(title, desc)


def fallback_world(bio: float, tech: float, culture: float) -> World:
    # Seeded randomness for stability yet variety across nearby values
    seed = int(bio * 100) * 10000 + int(tech * 100) * 100 + int(culture * 100)
    rng = random.Random(seed)

    def bucket(x: float) -> int:
        return 0 if x < 0.33 else (1 if x < 0.66 else 2)

    bio_opts = [
        ["salt flats", "wind-scoured reefs", "glacial shelves"],
        ["bottle biomes", "tidal forests", "mesa wetlands"],
        ["fungal canopy", "braided mycelia", "bioluminescent thickets"],
    ]
    tech_opts = [
        ["stone-and-sinew tools", "sail craft", "mechanical clocks"],
        ["steamworks", "industrial derricks", "telegraph yards"],
        ["autonomous swarms", "lattice fabs", "orbital mirrors"],
    ]
    culture_opts = [
        ["loose caravans", "raiding fanes", "barter fleets"],
        ["guild charters", "dock syndicates", "workhouse courts"],
        ["meritocratic bureaus", "cadaster ministries", "ledger temples"],
    ]

    b = bucket(bio)
    t = bucket(tech)
    c = bucket(culture)

    bio_theme = rng.choice(bio_opts[b])
    tech_theme = rng.choice(tech_opts[t])
    culture_theme = rng.choice(culture_opts[c])

    bio_t = bio_theme.title()
    title_shapes = [
        "The {bio_t}",
        "{bio_t} Frontier",
        "{bio_t} Expanse",
        "{bio_t} Cartography",
    ]
    title = rng.choice(title_shapes).format(bio_t=bio_t)

    openers = [
        "On this world, {bio} dominate the horizon.",
        "Here, life clings to {bio}.",
        "Across the {bio}, settlement is a negotiation.",
    ]
    tech_lines = [
        "Their tools run to {tech}.",
        "Industry favors {tech}.",
        "Most labor relies on {tech}.",
    ]
    culture_lines = [
        "Society coheres through {culture}.",
        "Governance trends toward {culture}.",
        "The social order codifies as {culture}.",
    ]
    consequences = [
        "Cities are carved where terrain, machine and rule briefly agree.",
        "Trade routes arc between places that can survive the season.",
        "Power accrues to those who can predict the next tide or audit.",
        "People bargain with ecology in public ledgers and private rites.",
    ]

    desc = " ".join([
        rng.choice(openers).format(bio=bio_theme),
        rng.choice(tech_lines).format(tech=tech_theme),
        rng.choice(culture_lines).format(culture=culture_theme),
        rng.choice(consequences),
    ])

    # Clean description without the BIO/TECH/CULTURE labels
    return World(title, desc)
