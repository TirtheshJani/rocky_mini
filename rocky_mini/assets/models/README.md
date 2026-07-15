# Vendored models

## silero_vad.onnx

- Source: the `silero-vad` PyPI package, version 6.2.1 (`silero_vad/data/silero_vad.onnx`),
  originally from https://github.com/snakers4/silero-vad.
- License: MIT, Copyright (c) Silero Team.
- Why vendored: `rocky_mini/audio/vad.py` runs this model with onnxruntime directly.
  Depending on the `silero-vad` package instead would pull in torch and the CUDA stack
  (gigabytes), which the robot's CM4 can neither fit nor use. See decisions.md entry 11.
- To update: install a newer `silero-vad` in a scratch venv, copy the new
  `silero_vad.onnx` here, re-run `pytest tests/test_audio_input.py`, and update the
  version noted above.
