"""Motion domain: the single set_target owner (MovementManager) and its layers.

All pose math is pure and testable with no SDK. Only MovementManager.run() calls
mini.set_target(), exactly once per 100 Hz tick, and that call is import-guarded.
"""
