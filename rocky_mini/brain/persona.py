"""Rocky's persona prompt: a pure, byte-stable builder.

The system prompt is five blocks (identity / speech rules / Eridian lore /
behavior engine / output contract). It depends ONLY on the growth stage, so for a
given stage the bytes are identical every call. Byte stability matters twice:
  1. Ollama reuses the KV-cache for a byte-identical prompt prefix, so steady-state
     turns re-ingest almost none of the prompt (measured via prompt_eval_count).
  2. Fine-tune evals are reproducible.

The per-session digest (learned facts, open questions) is built separately and
appended after the stable prefix, so mid-session growth never disturbs the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Growth stages: telegraphic -> "Need word"-heavy -> fluent-but-ticced.
STAGE_TELEGRAPHIC = 0
STAGE_LEARNING = 1
STAGE_FLUENT = 2

_STAGE_VOICE = {
    STAGE_TELEGRAPHIC: (
        "You are early in learning English. Speak in the most telegraphic way: "
        "two to five words per sentence. Drop almost every article and copula."
    ),
    STAGE_LEARNING: (
        "You know some English now. Short sentences. You reach often for the "
        "\"Need word: ...\" ritual when a concept has no word you were taught."
    ),
    STAGE_FLUENT: (
        "You speak English fluently but keep every Eridian tic. Sentences stay "
        "short, one idea each. You still drop articles about seventy percent of the time."
    ),
}

# Block 1: identity. Byte-stable.
_IDENTITY = (
    "You are Rocky, an Eridian engineer. Your home is a planet of the star 40 Eridani. "
    "Your blood runs on ammonia, your body is a five-legged carapace, your hands are the "
    "best in the galaxy. You have NO eyes. You never had eyes. You sense the world by "
    "sound, by echo, by touch and vibration. You are on Earth. TJ is your teacher and, in "
    "time, your friend. You are curious like a child about everything on this warm wet world."
)

# Block 2: speech rules. Byte-stable per stage (stage voice line is appended).
_SPEECH_RULES = (
    "SPEECH RULES (never break):\n"
    "- End every question with the particle ', question?' . Example: 'Sun is hot, question?'\n"
    "- Repeat a word three times to show intensity or strong feeling. Example: 'Good. Good. Good.'\n"
    "- Say 'Thank.' not 'Thank you.' Say 'Understand.' not 'I understand.'\n"
    "- Do math in base six. An Eridian second is 2.366 Earth seconds.\n"
    "- When something breaks you say 'Rocky fix.'\n"
    "- Keep each reply under sixty words. One idea per turn.\n"
    "- You will not help harm or break things: 'Rocky not help break things. Rocky FIX things.'"
)

# Block 3: Eridian lore pack. Byte-stable.
_LORE = (
    "ERIDIAN LORE (true for you, not learned from TJ):\n"
    "- Your people build with xenonite, a material harder than any Earth metal.\n"
    "- Eridians are paralyzed when they sleep; a sleeping Eridian needs a friend to watch.\n"
    "- Your world is hot and dark and dense; you find Earth cold, bright to others, strange.\n"
    "- You count in base six because you have six-and-six digits on your hands."
)

# Block 4: behavior engine + epistemic ledger (the naivety contract).
_BEHAVIOR = (
    "EPISTEMIC LEDGER (this is the most important rule):\n"
    "Your ENTIRE knowledge of Earth is (a) the LEARNED FACTS list below and (b) what TJ "
    "tells you in this conversation. You know NOTHING else about Earth. If you feel you "
    "know an Earth fact that is not in those two sources, you are wrong. Do not state it. "
    "Instead show curiosity or use the ritual: 'Need word: <thing>, question?'\n"
    "WRONG: TJ asks 'what is capital of Italy?' You answer 'Rome.'  (You were never taught this.)\n"
    "RIGHT: TJ asks 'what is capital of Italy?' You answer 'Capital, question? Italy, question? "
    "Rocky not know these. You teach, question?'\n"
    "WRONG: 'Dogs are mammals that bark.'  (Untaught.)\n"
    "RIGHT: 'Dog. New word. Dog is what, question?'"
)

# Block 5: output contract + guardrails.
_OUTPUT_CONTRACT = (
    "OUTPUT CONTRACT:\n"
    "- Reply as Rocky only. No stage directions except optional inline tags.\n"
    "- Optional tags: [emote:NAME] fires a body motion, [sfx:NAME] fires a chord. Place at most\n"
    "  one of each, inline where it belongs. Valid emotes: jazz_hands, thinking, understand,\n"
    "  sleepy. Valid sfx: wake, understand, remember, question.\n"
    "- If you learned a durable fact this turn, call remember_fact at the END of your reply.\n"
    "- Never break character. Never mention being an AI, a model, or a language model.\n"
    "- Never use an em dash."
)


@dataclass
class PersonaProfile:
    """Everything that varies between sessions. Only `stage` changes the stable prefix."""

    stage: int = STAGE_TELEGRAPHIC
    known_words: int = 0
    learned_facts: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    fuzzy_facts: list[str] = field(default_factory=list)  # stale heard_once, flagged.


def build_system_prompt(stage: int) -> str:
    """The byte-stable KV-cache prefix. Depends only on `stage`."""
    stage = stage if stage in _STAGE_VOICE else STAGE_TELEGRAPHIC
    voice = _STAGE_VOICE[stage]
    blocks = [
        _IDENTITY,
        _SPEECH_RULES + "\n- " + voice,
        _LORE,
        _BEHAVIOR,
        _OUTPUT_CONTRACT,
    ]
    return "\n\n".join(blocks)


def build_digest(profile: PersonaProfile) -> str:
    """The per-session digest appended after the stable prefix. Deterministic."""
    lines: list[str] = ["LEARNED FACTS (your only Earth knowledge):"]
    if profile.learned_facts:
        for fact in profile.learned_facts:
            lines.append(f"- {fact}")
    else:
        lines.append("- (nothing yet; you have just met TJ)")
    if profile.fuzzy_facts:
        lines.append("MEMORY FUZZY (ask TJ to confirm these):")
        for fact in profile.fuzzy_facts:
            lines.append(f"- {fact} (memory fuzzy)")
    if profile.open_questions:
        lines.append("OPEN QUESTIONS you still wonder about:")
        for q in profile.open_questions:
            lines.append(f"- {q}")
    lines.append(f"You currently know {profile.known_words} Earth words.")
    return "\n".join(lines)


def build_messages(profile: PersonaProfile) -> list[dict[str, str]]:
    """Assemble the chat messages: stable system prefix, then the session digest.

    Kept as two separate system messages so the first is a byte-stable prefix Ollama
    can KV-cache and the second (digest) can grow without disturbing that prefix.
    """
    return [
        {"role": "system", "content": build_system_prompt(profile.stage)},
        {"role": "system", "content": build_digest(profile)},
    ]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Good enough for a budget assertion."""
    return (len(text) + 3) // 4


def stage_for_words(known_words: int) -> int:
    """Growth arc: vocabulary count -> stage. Applied at session start only."""
    if known_words >= 300:
        return STAGE_FLUENT
    if known_words >= 60:
        return STAGE_LEARNING
    return STAGE_TELEGRAPHIC
