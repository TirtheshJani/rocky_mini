"""Recorded-emote tests: the Pollen RecordedMove adapter (additive deltas relative
to the first frame) and its composition through MovementManager stays inside the
clamp limits for every emote in ROCKY_EMOTES."""

import json
from pathlib import Path

import pytest

import rocky_mini
from rocky_mini.motion.emotes import PRIMARY_EMOTES, ROCKY_EMOTES
from rocky_mini.motion.manager import MovementManager
from rocky_mini.motion.pose import (
    ANTENNA_MAX,
    ANTENNA_MIN,
    FIELDS,
    PITCH_LIMIT,
    ROLL_LIMIT,
)
from rocky_mini.motion.recorded import load_recorded

EMOTES_DIR = Path(rocky_mini.__file__).parent / "assets" / "emotes"


# -- adapter unit test -----------------------------------------------------
def test_load_recorded_adapter_shape_and_zero_origin():
    path = EMOTES_DIR / "amazed1.json"
    raw = json.loads(path.read_text())
    expected_duration = raw["time"][-1] - raw["time"][0]

    traj = load_recorded(path)
    assert traj.duration == pytest.approx(expected_duration)
    assert traj.duration > 0

    # Deltas are relative to the first frame, so sample(0) is ~0 on every field.
    origin = traj.sample(0.0)
    assert set(origin.keys()) == set(FIELDS)
    assert max(abs(origin[f]) for f in FIELDS) < 1e-6

    # A mid-trajectory sample carries all seven delta fields.
    mid = traj.sample(traj.duration * 0.5)
    assert set(mid.keys()) == set(FIELDS)

    # Sampling past the end clamps just before the boundary and never raises.
    end = traj.sample(traj.duration + 1.0)
    assert set(end.keys()) == set(FIELDS)


# -- registry + manager integration ---------------------------------------
def test_recorded_emotes_registered():
    # Every Rocky vocabulary name resolved to a loaded trajectory.
    for name in ROCKY_EMOTES:
        traj = PRIMARY_EMOTES.get(name)
        assert traj is not None, f"recorded emote {name} not loaded"
        assert traj.duration > 0


def test_recorded_emotes_stay_within_clamp_limits():
    dt = 0.01
    for name in ROCKY_EMOTES:
        traj = PRIMARY_EMOTES[name]

        # The first tick's emote contribution is the sample at t=0: near zero.
        first = traj.sample(0.0)
        assert max(abs(first[f]) for f in FIELDS) < 1e-6

        holder = {"t": 0.0}
        mgr = MovementManager(clock=lambda h=holder: h["t"])
        mgr.start_emote(name, t=0.0)

        # Tick at 100 Hz from t=0 through duration + 0.5 s (past the end).
        n_steps = int((traj.duration + 0.5) / dt) + 1
        for i in range(n_steps):
            t = i * dt
            holder["t"] = t
            pose = mgr.tick(t=t, dt=dt)
            assert -PITCH_LIMIT <= pose.head_pitch <= PITCH_LIMIT, name
            assert -ROLL_LIMIT <= pose.head_roll <= ROLL_LIMIT, name
            assert ANTENNA_MIN <= pose.antenna_left <= ANTENNA_MAX, name
            assert ANTENNA_MIN <= pose.antenna_right <= ANTENNA_MAX, name

        # Running past its duration clears the primary emote slot.
        assert mgr._primary_emote is None, name
