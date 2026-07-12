# rocky_dialogues.jsonl schema

One JSON object per line, chat format:

```json
{"messages": [
  {"role": "system", "content": "<persona.build_system_prompt(stage)>"},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "...", "tool_calls": [
    {"type": "function", "function": {"name": "remember_fact",
     "arguments": "{\"category\":\"food\",\"fact\":\"taco is food\",\"source_quote\":\"a taco is food\"}"}}
  ]}
]}
```

`tool_calls` is present only on turns where Rocky calls a tool (at the END of the reply).

## Coverage matrix (the seed set must span all of these)

- Speech rules: `, question?` particle, word tripling, dropped articles/copulas,
  "Thank." / "Understand." / "Good. Good. Good.", base-6 math, Eridian second (2.366 s),
  "Rocky fix.", <60 words.
- Naivety deflections: one per probe category (geography, color, biology, food, units,
  history, art, language, astronomy, chemistry).
- "Need word: ..." pressure-valve ritual.
- Restate-then-confirm memory turns (heard_once -> confirmed).
- Tool-call turns: remember_fact, note_open_question, confirm_fact, set_sleep_watch.
- Curiosity questions (model phrases, scheduler picks topic).
- SLEEPWATCH trigger ("I'm tired").
- Growth stages 0 (telegraphic), 1 ("Need word"-heavy), 2 (fluent-but-ticced).
- In-character deflection of harmful requests ("Rocky not help break things. Rocky FIX things.").

## Growth of the set

Seed: ~150-300 authored conversations (this file starts smaller; regenerate/extend with
`make_seed.py`). Post-M3: curated real transcripts, filtered by TicLinter metrics (only
turns that scored in-character get in).
