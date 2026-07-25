"""Motion tests: clamp safety, trajectory boundary de-risk, jazz-hands, wobble,
manager composition + no-snap blending."""

import math
import random

import pytest

from rocky_mini.motion.emotes import Trajectory, jazz_hands_delta
from rocky_mini.motion.idle import IdlePolicy, breathing_delta, sleepwatch_delta
from rocky_mini.motion.manager import LISTEN_ANTENNA_RAISE, MovementManager
from rocky_mini.motion.pose import (
    ANTENNA_MAX,
    ANTENNA_MIN,
    PITCH_LIMIT,
    Pose,
    YAW_DIFF_LIMIT,
    apply_deltas,
    clamp_pose,
)
from rocky_mini.motion.wobble import WobbleTracker


class _Clock:
    """Controllable clock: tests advance .t and inject it as MovementManager's clock."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


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


# -- attentive antenna perk (listening) ------------------------------------
def test_set_listening_perks_and_blends_back():
    clock = _Clock()
    mgr = MovementManager(clock=clock)
    antennas: list[float] = []

    def _tick():
        pose = mgr.tick(clock.t, dt=0.01)
        antennas.extend([pose.antenna_left, pose.antenna_right])
        return pose

    # Baseline: listening off, antennas hover in the resting/breathing band (~10 deg).
    rest = None
    for i in range(20):
        clock.t = i * 0.01
        rest = _tick()
    assert 7.0 <= rest.antenna_left <= 13.0
    baseline_left = rest.antenna_left

    # Listening on: over ~0.6 s the antennas perk well above the breathing band.
    mgr.set_listening(True)
    up = None
    for _ in range(60):
        clock.t += 0.01
        up = _tick()
    assert up.antenna_left > baseline_left + 10.0
    assert up.antenna_left > 20.0  # settled above the resting/breathing band
    assert up.antenna_right > 20.0  # both antennas raised together
    assert up.antenna_left == pytest.approx(up.antenna_right, abs=1.0)

    # Listening off: over ~0.6 s the antennas blend back down toward rest.
    mgr.set_listening(False)
    down = None
    for _ in range(60):
        clock.t += 0.01
        down = _tick()
    assert down.antenna_left < up.antenna_left

    # Footgun 6: every commanded antenna stayed inside [3, 45] throughout.
    assert min(antennas) >= ANTENNA_MIN
    assert max(antennas) <= ANTENNA_MAX
    # The perk target sits well inside the clamp (rest ~10 + LISTEN_ANTENNA_RAISE).
    assert 3.0 <= 10.0 + LISTEN_ANTENNA_RAISE <= 45.0


# -- idle liveliness cadence -----------------------------------------------
def test_idle_policy_cadence_and_distribution():
    glance_s = 100.0
    policy = IdlePolicy(idle_glance_s=glance_s, rng=random.Random(12345))
    # Nothing before the first window: the first poll lazily arms the cadence clock.
    assert policy.maybe_act(0.0) is None
    assert policy.maybe_act(glance_s * 0.5) is None

    counts = {"none": 0, "sweep_look": 0, "calm": 0}
    calm = {"bored", "surprised", "welcome"}
    n = 20000
    for i in range(1, n + 1):
        action = policy.maybe_act(glance_s * i)
        if action is None:
            counts["none"] += 1
        elif action == "sweep_look":
            counts["sweep_look"] += 1
        else:
            assert action in calm  # every non-None action is a known idle action
            counts["calm"] += 1

    assert counts["none"] / n == pytest.approx(0.6, abs=0.03)
    assert counts["sweep_look"] / n == pytest.approx(0.2, abs=0.03)
    assert counts["calm"] / n == pytest.approx(0.2, abs=0.03)


def test_manager_idle_fires_during_silence():
    clock = _Clock()
    mgr = MovementManager(clock=clock, idle_glance_s=1.0, rng=random.Random(0))
    fired = False
    for _ in range(1000):
        clock.t += 1.0
        mgr.tick(clock.t, dt=1.0)
        if mgr._primary_emote is not None:
            fired = True
            break
    assert fired, "an idle primary emote should eventually start during long silence"


def test_manager_activity_resets_idle_timer():
    clock = _Clock()
    mgr = MovementManager(clock=clock, idle_glance_s=1.0, rng=random.Random(0))
    # Prime: the first tick lazily arms the next-eval to t + idle_glance_s (= 1.0).
    mgr.tick(clock.t, dt=0.01)  # t = 0.0
    # Just before that window closes, activity re-arms the cadence to now + 1.0.
    clock.t = 0.9
    mgr.set_doa(20.0)
    # Past the original window (1.0) but before the reset one (1.9): nothing fires.
    clock.t = 1.2
    mgr.tick(clock.t, dt=0.01)
    assert mgr._primary_emote is None


def test_manager_no_idle_action_while_sleep_watch():
    clock = _Clock()
    mgr = MovementManager(clock=clock, idle_glance_s=1.0, rng=random.Random(0))
    mgr.set_sleep_watch(True)
    for _ in range(200):
        clock.t += 1.0
        mgr.tick(clock.t, dt=1.0)
        assert mgr._primary_emote is None  # idle actions never fire during sleep-watch
