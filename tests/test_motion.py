"""Motion tests: clamp safety, trajectory boundary de-risk, jazz-hands, wobble,
manager composition + no-snap blending."""

import math

import pytest

from rocky_mini.motion.emotes import Trajectory, jazz_hands_delta
from rocky_mini.motion.idle import breathing_delta, sleepwatch_delta
from rocky_mini.motion.manager import MovementManager
from rocky_mini.motion.pose import (
    ANTENNA_MIN,
    PITCH_LIMIT,
    Pose,
    YAW_DIFF_LIMIT,
    apply_deltas,
    clamp_pose,
)
from rocky_mini.motion.wobble import WobbleTracker


# -- clamp safety ----------------------------------------------------------
def test_clamp_limits_pitch_roll():
    p = clamp_pose(Pose(head_pitch=90, head_roll=-90))
    assert p.head_pitch == PITCH_LIMIT
    assert p.head_roll == -PITCH_LIMIT


def test_clamp_yaw_difference():
    p = clamp_pose(Pose(head_yaw=200, body_yaw=0))
    assert abs(p.head_yaw - p.body_yaw) <= YAW_DIFF_LIMIT + 1e-6


def test_antennas_never_zero():
    p = clamp_pose(Pose(antenna_left=0.0, antenna_right=-5.0))
    assert p.antenna_left >= ANTENNA_MIN
    assert p.antenna_right >= ANTENNA_MIN


# -- trajectory boundary de-risk (SDK RecordedMove.evaluate raises at t>=duration) --
def test_trajectory_evaluate_raises_at_boundary():
    traj = Trajectory([(0.0, {}), (1.0, {"head_pitch": 10.0})])
    with pytest.raises(ValueError):
        traj.evaluate(1.0)
    with pytest.raises(ValueError):
        traj.evaluate(1.5)


def test_trajectory_sample_clamps_before_boundary():
    traj = Trajectory([(0.0, {}), (1.0, {"head_pitch": 10.0})])
    # sample() must not raise at or past duration.
    v = traj.sample(1.0)
    assert v["head_pitch"] == pytest.approx(10.0, abs=1e-3)
    assert traj.sample(5.0)["head_pitch"] == pytest.approx(10.0, abs=1e-3)


def test_trajectory_interpolates_midpoint():
    traj = Trajectory([(0.0, {"head_pitch": 0.0}), (1.0, {"head_pitch": 10.0})])
    assert traj.evaluate(0.5)["head_pitch"] == pytest.approx(5.0)


# -- jazz hands ------------------------------------------------------------
def test_jazz_hands_shakes_and_finishes():
    mid = jazz_hands_delta(0.05)  # near a peak of the 6 Hz shake
    assert mid["antenna_left"] == pytest.approx(-mid["antenna_right"])
    assert jazz_hands_delta(2.0) == {}  # finished after 1.2 s


# -- wobble ----------------------------------------------------------------
def test_wobble_is_delayed_to_playout():
    w = WobbleTracker(gain=8.0, delay_s=0.2)
    w.feed(0.5, t=0.0)
    assert w.delta(0.1)["head_pitch"] == 0.0  # not yet audible
    assert w.delta(0.25)["head_pitch"] == pytest.approx(-4.0)  # 8 * 0.5, nod down


# -- idle ------------------------------------------------------------------
def test_breathing_is_bounded_and_moves():
    vals = [breathing_delta(t)["head_pitch"] for t in (0.0, 0.5, 1.0, 2.0)]
    assert max(abs(v) for v in vals) <= 3.0
    assert len(set(vals)) > 1  # it actually moves


def test_sleepwatch_lowers_head():
    d = sleepwatch_delta(0.0, doa_deg=30.0)
    assert d["head_pitch"] < 0  # head lowered
    assert d["head_yaw"] == pytest.approx(15.0)  # oriented toward DoA


# -- manager composition ---------------------------------------------------
def test_manager_tick_returns_clamped_pose():
    mgr = MovementManager()
    pose = mgr.tick(t=0.0, dt=0.01)
    assert abs(pose.head_pitch) <= PITCH_LIMIT
    assert pose.antenna_left >= ANTENNA_MIN


def test_manager_blends_without_snap():
    mgr = MovementManager(blend_tau=0.4)
    mgr.start_emote("understand", t=0.0)
    # One 10 ms tick should move only a small fraction toward the emote target, not snap.
    first = mgr.tick(t=0.01, dt=0.01)
    # blend_factor(0.01, 0.4) ~ 0.025, so the primary barely moves this tick.
    assert abs(first.head_pitch) < 5.0


def test_manager_jazz_hands_reflex_moves_antennas():
    mgr = MovementManager()
    baseline = mgr.tick(t=0.0, dt=0.01)
    mgr.start_emote("jazz_hands", t=0.0)
    shaken = mgr.tick(t=0.05, dt=0.01)
    assert shaken.antenna_left != baseline.antenna_left


def test_manager_doa_orients_head():
    mgr = MovementManager()
    mgr.set_doa(40.0)
    pose = mgr.tick(t=0.0, dt=0.01)
    assert pose.head_yaw > 0  # turned toward the sound
