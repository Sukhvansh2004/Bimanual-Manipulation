"""Tests for the learned handover-feasibility filter."""

import numpy as np
import pytest

from bimanual_manipulation.planning.assignment.cost import HandoverResult
from bimanual_manipulation.planning.assignment.learned_filter import (
    ARCHITECTURES,
    FEATURE_GROUPS,
    N_FEATURES,
    HandoverFeasibilityNet,
    HandoverLogger,
    LearnedHandoverFilter,
    handover_features,
    rejection_category,
)
from bimanual_manipulation.planning.assignment.models import (
    ObjectTrajectory,
    RobotTrajectory,
    TrajectoryBundle,
)

N_WAYPOINTS = 6


def _robot_traj(rng: np.random.Generator, infeasible_at: int = -1) -> RobotTrajectory:
    """Build a random robot trajectory.

    Args:
        rng: Random generator.
        infeasible_at: Waypoint to mark infeasible (NaN joint config);
            ``-1`` keeps all waypoints feasible.

    Returns:
        A :class:`RobotTrajectory` with ``N_WAYPOINTS`` waypoints.
    """
    n = N_WAYPOINTS
    q = rng.normal(size=(n, 7))
    feasible = np.ones(n, dtype=bool)
    if infeasible_at >= 0:
        q[infeasible_at] = np.nan
        feasible[infeasible_at] = False
    return RobotTrajectory(
        eef_positions=rng.normal(size=(n, 3)),
        eef_quaternions=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        eef_frame_ids=np.array(["world"] * n),
        object_positions=rng.normal(size=(n, 3)),
        object_quaternions=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        object_frame_ids=np.array(["world"] * n),
        is_feasible=feasible,
        ik_success=feasible,
        collision_free=np.ones(n, dtype=bool),
        joint_configs=q,
    )


@pytest.fixture
def bundle() -> TrajectoryBundle:
    """Bundle with two robot-0 trajectories (the second infeasible at waypoint 2)
    and one robot-1 trajectory."""
    rng = np.random.default_rng(0)
    obj = ObjectTrajectory(
        positions=rng.normal(size=(N_WAYPOINTS, 3)),
        quaternions=np.tile([0.0, 0.0, 0.0, 1.0], (N_WAYPOINTS, 1)),
        frame_ids=np.array(["world"] * N_WAYPOINTS),
    )
    return TrajectoryBundle(
        object_trajectory=obj,
        robot0_trajectories=[_robot_traj(rng), _robot_traj(rng, infeasible_at=2)],
        robot1_trajectories=[_robot_traj(rng)],
    )


@pytest.fixture(params=ARCHITECTURES)
def model(request) -> HandoverFeasibilityNet:
    """Small untrained model (each architecture) with identity input normalisation."""
    model = HandoverFeasibilityNet(
        np.zeros(N_FEATURES), np.ones(N_FEATURES), hidden=16, arch=request.param,
    )
    model.net.eval()
    return model


class TestHandoverFeatures:
    def test_shape_and_dtype(self, bundle):
        f = handover_features(bundle, 3, 0, 0, 1, 0)
        assert f.shape == (N_FEATURES,)
        assert f.dtype == np.float32
        assert np.all(np.isfinite(f))

    def test_direction_flag(self, bundle):
        assert handover_features(bundle, 3, 0, 0, 1, 0)[0] == 0.0
        assert handover_features(bundle, 3, 1, 0, 0, 0)[0] == 1.0

    def test_missing_joint_config_returns_none(self, bundle):
        assert handover_features(bundle, 2, 0, 1, 1, 0) is None

    def test_out_of_range_segment_returns_none(self, bundle):
        assert handover_features(bundle, N_WAYPOINTS, 0, 0, 1, 0) is None


class TestRejectionCategory:
    @pytest.mark.parametrize("reason, feasible, expected", [
        ("", True, "feasible"),
        ("ik_fail_transfer(2_failed@3/10_seen)", False, "ik_fail_transfer"),
        ("ik_fail_receiver(1_failed@1/8_seen)", False, "ik_fail_receiver"),
        ("ik_fail_transfer(1/10) receiver(2/8)", False, "ik_fail_both"),
        ("collision(1_colliding@4/12_checked)", False, "collision"),
        ("empty_trajectory", False, "other"),
    ])
    def test_categories(self, reason, feasible, expected):
        assert rejection_category(reason, feasible) == expected


class TestHandoverLogger:
    def test_records_and_saves(self, bundle, tmp_path):
        chained = []
        log = HandoverLogger(bundle, chain=chained.append)
        log(HandoverResult(cost=1.0, feasible=True, transfer_robot_id=0, receiver_robot_id=1,
                           segment_idx=3, prev_traj_idx=0, curr_traj_idx=0))
        log(HandoverResult(cost=np.inf, feasible=False, transfer_robot_id=1, receiver_robot_id=0,
                           segment_idx=4, prev_traj_idx=0, curr_traj_idx=0,
                           rejection_reason="collision(1_colliding@2/5_checked)"))
        # No joint config at segment 2 for robot0 traj1 -> not logged, still chained
        log(HandoverResult(cost=np.inf, feasible=False, transfer_robot_id=0, receiver_robot_id=1,
                           segment_idx=2, prev_traj_idx=1, curr_traj_idx=0))
        assert len(chained) == 3

        path = tmp_path / "log.npz"
        log.save(str(path))
        d = np.load(path)
        assert d["features"].shape == (2, N_FEATURES)
        assert d["feasible"].tolist() == [True, False]
        assert d["category"].tolist() == ["feasible", "collision"]
        assert d["meta"].tolist()[0] == [3, 0, 0, 1, 0]

    def test_save_empty(self, bundle, tmp_path):
        path = tmp_path / "empty.npz"
        HandoverLogger(bundle).save(str(path))
        assert np.load(path)["features"].shape == (0, N_FEATURES)


class _ConstantModel:
    """Stand-in model returning fixed predictions."""

    def __init__(self, p: float, cost: float) -> None:
        """Args:
            p: Predicted ``P(feasible)`` for every input.
            cost: Predicted handover cost for every input.
        """
        self.p = p
        self.cost = cost

    def predict(self, feats: np.ndarray):
        """Return the fixed ``(p_feasible, cost)`` prediction for every row."""
        n = len(np.atleast_2d(feats))
        return np.full(n, self.p), np.full(n, self.cost)


class TestLearnedHandoverFilter:
    TRAJECTORY_INFO = {0: (0, 0), 1: (0, 1), 2: (1, 0)}

    def _skip(self, filt, bundle, prev_state=0, curr_state=2, segment=3,
              acc_cost=0.0, best_cost=np.inf):
        """Query *filt* for a robot0 -> robot1 handover; returns its skip decision."""
        return filt(bundle, self.TRAJECTORY_INFO, segment, prev_state, curr_state,
                    acc_cost, best_cost)

    def test_low_feasibility_skips(self, bundle):
        filt = LearnedHandoverFilter(_ConstantModel(p=0.05, cost=1.0), threshold=0.1)
        assert self._skip(filt, bundle)
        assert filt.stats()["filter_n_skipped_feasibility"] == 1

    def test_high_feasibility_keeps(self, bundle):
        filt = LearnedHandoverFilter(_ConstantModel(p=0.9, cost=1.0), threshold=0.1)
        assert not self._skip(filt, bundle)
        assert filt.stats()["filter_n_skipped"] == 0

    def test_cost_pruning(self, bundle):
        filt = LearnedHandoverFilter(_ConstantModel(p=0.9, cost=10.0), threshold=0.1,
                                     cost_slack=0.5)
        # 3 + 0.5 * 10 = 8 >= 7 -> pruned; 8 < 9 -> kept
        assert self._skip(filt, bundle, acc_cost=3.0, best_cost=7.0)
        assert not self._skip(filt, bundle, acc_cost=3.0, best_cost=9.0)
        assert not self._skip(filt, bundle, acc_cost=3.0, best_cost=np.inf)
        assert filt.stats()["filter_n_skipped_cost"] == 1

    def test_cost_pruning_disabled_by_default(self, bundle):
        filt = LearnedHandoverFilter(_ConstantModel(p=0.9, cost=1e6), threshold=0.1)
        assert not self._skip(filt, bundle, acc_cost=3.0, best_cost=4.0)

    def test_order_sorts_by_predicted_total_and_caches(self, bundle):
        filt = LearnedHandoverFilter(_ConstantModel(p=0.9, cost=5.0), threshold=0.1)
        # Both candidates are robot0 -> robot1; equal predicted cost, so the
        # order follows acc_cost; the infeasible-feature one (state 1 at
        # segment 2) goes last.
        ordered = filt.order(bundle, self.TRAJECTORY_INFO, 3, [(4.0, 1), (2.0, 0)], 2)
        assert [ps for _, ps in ordered] == [0, 1]
        assert (3, 0, 2) in filt._cache and (3, 1, 2) in filt._cache
        ordered = filt.order(bundle, self.TRAJECTORY_INFO, 2, [(1.0, 1), (2.0, 0)], 2)
        assert [ps for _, ps in ordered] == [0, 1]
        assert filt.n_queried == 0  # ordering alone does not count as a query

    def test_order_puts_predicted_infeasible_last(self, bundle):
        filt = LearnedHandoverFilter(_ConstantModel(p=0.05, cost=1.0), threshold=0.1)
        ordered = filt.order(bundle, self.TRAJECTORY_INFO, 3, [(2.0, 0), (1.0, 1)], 2)
        assert sorted(ps for _, ps in ordered) == [0, 1]
        assert self._skip(filt, bundle, prev_state=0)  # served from the cache
        assert filt.stats()["filter_n_skipped_feasibility"] == 1

    def test_missing_features_defers_to_exact(self, bundle):
        filt = LearnedHandoverFilter(_ConstantModel(p=0.0, cost=1.0), threshold=0.1)
        # robot0 traj1 has no joint config at segment 2
        assert not self._skip(filt, bundle, prev_state=1, segment=2)
        assert filt.n_queried == 0


class TestHandoverFeasibilityNet:
    def test_feature_groups_tile_feature_vector(self):
        bounds = [(start, stop) for _, start, stop in FEATURE_GROUPS]
        assert bounds[0][0] == 0 and bounds[-1][1] == N_FEATURES
        assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            HandoverFeasibilityNet(np.zeros(N_FEATURES), np.ones(N_FEATURES), arch="cnn")

    def test_predict_shapes_and_ranges(self, model):
        x = np.random.default_rng(1).normal(size=(4, N_FEATURES)).astype(np.float32)
        p, cost = model.predict(x)
        assert p.shape == cost.shape == (4,)
        assert np.all((p >= 0) & (p <= 1))
        assert np.all(cost > 0)

    def test_save_load_roundtrip(self, model, tmp_path):
        x = np.random.default_rng(1).normal(size=(4, N_FEATURES)).astype(np.float32)
        path = tmp_path / "model.pt"
        model.save(str(path))
        loaded = HandoverFeasibilityNet.load(str(path))
        for a, b in zip(loaded.predict(x), model.predict(x)):
            np.testing.assert_allclose(a, b, rtol=1e-6)
