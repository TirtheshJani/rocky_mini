# UI end-to-end test (Playwright)

`pw-rocky-ui.js` drives the settings UI in a real browser: load, teach a fact via chat,
verify it appears in the fact table and bumps the word count, check the live metrics
panel and latency meter, verify naivety deflection on an Earth-knowledge probe, confirm
and delete a fact, fire an emote, and toggle the model. It also captures desktop and
mobile screenshots.

## Run it

```bash
# 1. Start the sim UI (no robot, no Ollama, no GPU needed)
python -m rocky_mini.main --sim --host 127.0.0.1 --port 8042

# 2. In another shell, run the script via the playwright-skill executor
cd ~/.claude/skills/playwright-skill
ROCKY_URL=http://127.0.0.1:8042 SHOT_DIR=/tmp node run.js /path/to/tests/ui/pw-rocky-ui.js
```

Note: the page opens a Server-Sent-Events metrics stream, so the script waits on
`domcontentloaded` (not `networkidle`, which never fires while the stream is open).
