"""Learned handover-feasibility filter for the assignment solver.

Exact handover evaluation (:func:`~cost.compute_switching_cost`) solves IK
for every waypoint of the handover primitive and collision-checks both arms,
which dominates DP planning time.  This module provides:

* :func:`handover_features` — a cheap feature vector for a candidate
  handover, computed only from data already stored in the trajectory bundle.
* :class:`HandoverLogger` — a ``handover_callback`` that records features and
  exact-evaluation outcomes, used both as classifier training data and for
  the planning-failure taxonomy.
* :class:`HandoverFeasibilityNet` / :class:`LearnedHandoverFilter` — a network
  (MLP or feature-group-token transformer)
  that predicts ``P(feasible)`` and the handover cost, and a solver
  ``handover_prefilter`` that skips exact evaluation of candidates predicted
  to be infeasible or not to improve the current best transition.  Handovers
  that pass the filter are still verified exactly, so every accepted
  handover remains IK- and collision-checked.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

from bimanual_manipulation.planning.assignment.cost import HandoverResult
from bimanual_manipulation.planning.assignment.models import TrajectoryBundle

logger = logging.getLogger(__name__)

N_FEATURES = 65


def _pose_matrix(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from a position and quaternion.

    Args:
        pos: ``(3,)`` translation.
        quat_xyzw: ``(4,)`` quaternion in ``[x, y, z, w]`` order.

    Returns:
        ``(4, 4)`` homogeneous transform.
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = pos
    return T


def handover_features(
    trajectories: TrajectoryBundle,
    segment_idx: int,
    prev_robot: int,
    prev_traj_idx: int,
    curr_robot: int,
    curr_traj_idx: int,
) -> Optional[np.ndarray]:
    """Compute the feature vector of a candidate handover.

    Features (all at waypoint *segment_idx*, world frame): transfer
    direction, normalised segment index, both arms' joint configs, object
    pose, both end-effector poses, both grasp transforms (object in EE
    frame), and the EE-to-EE offset and distance. Rotations use the
    continuous 6-D representation (first two matrix columns).

    Args:
        trajectories: Trajectory bundle for both arms.
        segment_idx: Waypoint index at which the handover happens.
        prev_robot: Robot id of the transferring arm.
        prev_traj_idx: Trajectory index of the transferring arm.
        curr_robot: Robot id of the receiving arm.
        curr_traj_idx: Trajectory index of the receiving arm.

    Returns:
        ``(N_FEATURES,)`` float32 vector, or ``None`` when *segment_idx* is
        out of range or either arm has no joint config there (the exact
        evaluator rejects those cases cheaply anyway).
    """
    obj = trajectories.object_trajectory
    prev = trajectories.get_trajectory(prev_robot, prev_traj_idx)
    curr = trajectories.get_trajectory(curr_robot, curr_traj_idx)
    i = segment_idx
    if i >= obj.n_waypoints or i >= prev.n_waypoints or i >= curr.n_waypoints:
        return None
    q_prev = prev.joint_configs[i]
    q_curr = curr.joint_configs[i]
    if not (np.all(np.isfinite(q_prev)) and np.all(np.isfinite(q_curr))):
        return None

    T_obj = _pose_matrix(obj.positions[i], obj.quaternions[i])
    T_prev = _pose_matrix(prev.eef_positions[i], prev.eef_quaternions[i])
    T_curr = _pose_matrix(curr.eef_positions[i], curr.eef_quaternions[i])
    # Grasp transforms (object in EE frame) for both arms
    G_prev = np.linalg.inv(T_prev) @ T_obj
    G_curr = np.linalg.inv(T_curr) @ T_obj
    d_eef = T_prev[:3, 3] - T_curr[:3, 3]

    feats = np.concatenate([
        [float(prev_robot), i / max(obj.n_waypoints - 1, 1)],   # 2
        q_prev, q_curr,                                          # 14
        T_obj[:3, 3], T_obj[:3, :2].T.ravel(),                   # 9
        T_prev[:3, 3], T_prev[:3, :2].T.ravel(),                 # 9
        T_curr[:3, 3], T_curr[:3, :2].T.ravel(),                 # 9
        G_prev[:3, 3], G_prev[:3, :2].T.ravel(),                 # 9
        G_curr[:3, 3], G_curr[:3, :2].T.ravel(),                 # 9
        d_eef, [np.linalg.norm(d_eef)],                          # 4
    ])
    return feats.astype(np.float32)


def rejection_category(reason: str, feasible: bool) -> str:
    """Collapse a handover rejection reason into a coarse category.

    Args:
        reason: ``HandoverResult.rejection_reason`` string.
        feasible: Whether the handover was feasible.

    Returns:
        One of ``"feasible"``, ``"ik_fail_transfer"``, ``"ik_fail_receiver"``,
        ``"ik_fail_both"``, ``"collision"`` or ``"other"``.
    """
    if feasible:
        return "feasible"
    if reason.startswith("ik_fail_transfer") and "receiver" in reason:
        return "ik_fail_both"
    if reason.startswith("ik_fail_transfer"):
        return "ik_fail_transfer"
    if reason.startswith("ik_fail_receiver"):
        return "ik_fail_receiver"
    if reason.startswith("collision"):
        return "collision"
    return "other"


class HandoverLogger:
    """``handover_callback`` that records every exact handover evaluation.

    Optionally chains to another callback (e.g. the Meshcat visualizer).
    """

    def __init__(
        self,
        trajectories: TrajectoryBundle,
        chain: Optional[Callable[[HandoverResult], None]] = None,
    ) -> None:
        """Create an empty logger.

        Args:
            trajectories: Trajectory bundle the solver is planning over.
            chain: Optional callback invoked after logging each result.
        """
        self.trajectories = trajectories
        self.chain = chain
        self.features: List[np.ndarray] = []
        self.feasible: List[bool] = []
        self.cost: List[float] = []
        self.category: List[str] = []
        self.meta: List[Tuple[int, int, int, int, int]] = []

    def __call__(self, result: HandoverResult) -> None:
        """Record one exact handover evaluation.

        Results whose features cannot be computed are not recorded but are
        still forwarded to the chained callback.

        Args:
            result: Outcome of an exact handover evaluation.
        """
        f = handover_features(
            self.trajectories, result.segment_idx,
            result.transfer_robot_id, result.prev_traj_idx,
            result.receiver_robot_id, result.curr_traj_idx,
        )
        if f is not None:
            self.features.append(f)
            self.feasible.append(bool(result.feasible))
            self.cost.append(float(result.cost))
            self.category.append(rejection_category(result.rejection_reason, result.feasible))
            self.meta.append((
                result.segment_idx, result.transfer_robot_id, result.prev_traj_idx,
                result.receiver_robot_id, result.curr_traj_idx,
            ))
        if self.chain is not None:
            self.chain(result)

    def save(self, path: str) -> None:
        """Write all recorded evaluations to a compressed NPZ.

        Arrays: ``features (M, N_FEATURES)``, ``feasible (M,)``,
        ``cost (M,)`` (``inf`` if infeasible), ``category (M,)`` and
        ``meta (M, 5)`` = ``(segment, r_from, t_from, r_to, t_to)``.

        Args:
            path: Output file path.
        """
        np.savez_compressed(
            path,
            features=np.asarray(self.features, dtype=np.float32).reshape(-1, N_FEATURES),
            feasible=np.asarray(self.feasible, dtype=bool),
            cost=np.asarray(self.cost, dtype=np.float64),
            category=np.asarray(self.category, dtype="U20"),
            meta=np.asarray(self.meta, dtype=np.int32).reshape(-1, 5),
        )
        logger.info("Saved %d handover evaluations to %s", len(self.feasible), path)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

FEATURE_GROUPS: Tuple[Tuple[str, int, int], ...] = (
    ("context", 0, 2),        # transfer direction, normalised segment index
    ("q_give", 2, 9),
    ("q_recv", 9, 16),
    ("object", 16, 25),
    ("ee_give", 25, 34),
    ("ee_recv", 34, 43),
    ("grasp_give", 43, 52),
    ("grasp_recv", 52, 61),
    ("gap", 61, 65),
)
"""``(name, start, stop)`` slices of :func:`handover_features` used as tokens."""

ARCHITECTURES = ("mlp", "transformer")


class _TwoHeadMLP(nn.Module):
    """Shared MLP trunk with a feasibility-logit head and a log-cost head."""

    def __init__(self, n_in: int, hidden: int) -> None:
        """Build the network.

        Args:
            n_in: Input feature dimension.
            hidden: Width of each hidden layer.
        """
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the network.

        Args:
            x: ``(B, n_in)`` standardised features.

        Returns:
            ``(feasibility_logit, log_cost)``, each of shape ``(B,)``.
        """
        out = self.head(self.trunk(x))
        return out[..., 0], out[..., 1]


class _GroupTokenTransformer(nn.Module):
    """Feature-token transformer with feasibility-logit and log-cost heads.

    Each feature group in :data:`FEATURE_GROUPS` is linearly embedded into
    one token (plus a learned group embedding); a CLS token attends over all
    groups and feeds both heads.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int, dropout: float = 0.1) -> None:
        """Build the network.

        Args:
            d_model: Token embedding width.
            n_layers: Number of transformer encoder layers.
            n_heads: Attention heads per layer.
            dropout: Dropout rate inside the encoder.
        """
        super().__init__()
        self.slices = [(start, stop) for _, start, stop in FEATURE_GROUPS]
        self.embed = nn.ModuleList(nn.Linear(stop - start, d_model) for start, stop in self.slices)
        self.group_embedding = nn.Parameter(torch.zeros(len(self.slices), d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)
        nn.init.normal_(self.group_embedding, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the network.

        Args:
            x: ``(B, N_FEATURES)`` standardised features.

        Returns:
            ``(feasibility_logit, log_cost)``, each of shape ``(B,)``.
        """
        tokens = torch.stack(
            [emb(x[:, start:stop]) for emb, (start, stop) in zip(self.embed, self.slices)], dim=1,
        ) + self.group_embedding
        tokens = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1)
        out = self.head(self.norm(self.encoder(tokens)[:, 0]))
        return out[..., 0], out[..., 1]


def _build_net(arch: str, n_in: int, hidden: int) -> nn.Module:
    """Construct the network for *arch*.

    Args:
        arch: One of :data:`ARCHITECTURES`.
        n_in: Input feature dimension.
        hidden: MLP hidden width, or transformer ``d_model``.

    Returns:
        The (untrained) network.

    Raises:
        ValueError: If *arch* is unknown.
    """
    if arch == "mlp":
        return _TwoHeadMLP(n_in, hidden)
    if arch == "transformer":
        return _GroupTokenTransformer(d_model=hidden, n_layers=2, n_heads=4)
    raise ValueError(f"Unknown architecture {arch!r}; expected one of {ARCHITECTURES}")


class HandoverFeasibilityNet:
    """Predicts ``P(feasible)`` and the handover cost of a candidate handover.

    Inputs are standardised with training-set statistics.  The cost head is
    trained on feasible handovers only and predicts ``log(cost)``.
    """

    def __init__(
        self, mean: np.ndarray, std: np.ndarray, hidden: int = 256, arch: str = "mlp",
    ) -> None:
        """Create an untrained model.

        Args:
            mean: ``(N_FEATURES,)`` training-set feature mean.
            std: ``(N_FEATURES,)`` training-set feature standard deviation.
            hidden: MLP hidden width, or transformer ``d_model``.
            arch: One of :data:`ARCHITECTURES`.
        """
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.hidden = hidden
        self.arch = arch
        self.net = _build_net(arch, len(mean), hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standardise *x* and run the network.

        Args:
            x: ``(B, N_FEATURES)`` raw features.

        Returns:
            ``(feasibility_logit, log_cost)``, each of shape ``(B,)``.
        """
        mean = torch.as_tensor(self.mean, device=x.device)
        std = torch.as_tensor(self.std, device=x.device)
        return self.net((x - mean) / std)

    def predict(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict feasibility and cost without tracking gradients.

        Args:
            feats: ``(N_FEATURES,)`` or ``(B, N_FEATURES)`` raw features.

        Returns:
            ``(p_feasible, cost)``, each a ``(B,)`` numpy array.
        """
        with torch.no_grad():
            x = torch.as_tensor(np.atleast_2d(feats), dtype=torch.float32)
            logit, log_cost = self.forward(x)
            return torch.sigmoid(logit).numpy(), torch.exp(log_cost).numpy()

    def n_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def save(self, path: str) -> None:
        """Save weights, architecture and normalisation statistics.

        Args:
            path: Output checkpoint path.
        """
        torch.save({
            "state_dict": self.net.state_dict(),
            "mean": self.mean, "std": self.std, "hidden": self.hidden, "arch": self.arch,
        }, path)

    @classmethod
    def load(cls, path: str) -> "HandoverFeasibilityNet":
        """Load a checkpoint written by :meth:`save` (in eval mode, on CPU).

        Args:
            path: Checkpoint path.

        Returns:
            The loaded model.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ckpt["mean"], ckpt["std"], ckpt["hidden"], ckpt.get("arch", "mlp"))
        model.net.load_state_dict(ckpt["state_dict"])
        model.net.eval()
        return model


class LearnedHandoverFilter:
    """Learned ``handover_prefilter`` for the DP solver.

    A candidate handover is rejected without exact evaluation when either

    * ``P(feasible) < threshold`` (predicted infeasible), or
    * ``acc_cost + cost_slack * predicted_cost >= best_cost`` (predicted not
      to improve the best predecessor found so far for this state).

    ``cost_slack`` in ``(0, 1]`` discounts the predicted cost so that only
    candidates that are clearly worse are pruned; ``0`` disables cost
    pruning.  Accepted candidates are still evaluated exactly.

    :meth:`order` can additionally be passed to the solver as
    ``handover_order``: it scores all candidates of a state in one batch and
    visits them best-predicted-first, which tightens the incumbent early.
    """

    def __init__(
        self,
        model: HandoverFeasibilityNet,
        threshold: float,
        cost_slack: float = 0.0,
    ) -> None:
        """Create the filter.

        Args:
            model: Trained handover model.
            threshold: Minimum ``P(feasible)`` for exact evaluation.
            cost_slack: Discount on the predicted cost in the dominance test
                (``0`` disables cost pruning).
        """
        self.model = model
        self.threshold = threshold
        self.cost_slack = cost_slack
        self.n_queried = 0
        self.n_skipped_feasibility = 0
        self.n_skipped_cost = 0
        self.t_model = 0.0
        self._cache: Dict[Tuple[int, int, int], Tuple[float, float]] = {}

    def order(
        self,
        trajectories: TrajectoryBundle,
        trajectory_info: Dict[int, Tuple[int, int]],
        segment_idx: int,
        candidates: List[Tuple[float, int]],
        curr_state: int,
    ) -> List[Tuple[float, int]]:
        """Sort a state's handover candidates by predicted total cost.

        Predictions are computed in one batch and cached for the subsequent
        per-candidate :meth:`__call__` checks. Candidates without features
        keep their place at the end (in cumulative-cost order).

        Args:
            trajectories: Trajectory bundle for both arms.
            trajectory_info: Maps state index to ``(robot_id, traj_idx)``.
            segment_idx: Segment at which the handovers would happen.
            candidates: Cross-robot ``(acc_cost, prev_state)`` pairs.
            curr_state: Current (receiving) state index.

        Returns:
            The candidates ordered by ``acc_cost + predicted_cost``, with
            predicted-infeasible ones last.
        """
        self._cache = {}
        cr, ct = trajectory_info[curr_state]
        scored, feats, unscored = [], [], []
        for acc_cost, ps in candidates:
            pr, pt = trajectory_info[ps]
            f = handover_features(trajectories, segment_idx, pr, pt, cr, ct)
            if f is None:
                unscored.append((acc_cost, ps))
            else:
                scored.append((acc_cost, ps))
                feats.append(f)
        if not scored:
            return unscored

        t0 = time.perf_counter()
        p, cost = self.model.predict(np.stack(feats))
        self.t_model += time.perf_counter() - t0
        keys = []
        for (acc_cost, ps), pi, ci in zip(scored, p, cost):
            self._cache[(segment_idx, ps, curr_state)] = (float(pi), float(ci))
            infeasible = pi < self.threshold
            keys.append((infeasible, acc_cost + float(ci)))
        ranked = [c for _, c in sorted(zip(keys, scored), key=lambda kc: kc[0])]
        return ranked + unscored

    def __call__(
        self,
        trajectories: TrajectoryBundle,
        trajectory_info: Dict[int, Tuple[int, int]],
        segment_idx: int,
        prev_state: int,
        curr_state: int,
        acc_cost: float,
        best_cost: float,
    ) -> bool:
        """Decide whether to skip the exact evaluation of a handover.

        Args:
            trajectories: Trajectory bundle for both arms.
            trajectory_info: Maps state index to ``(robot_id, traj_idx)``.
            segment_idx: Segment at which the handover would happen.
            prev_state: Predecessor (transferring) state index.
            curr_state: Current (receiving) state index.
            acc_cost: Cumulative cost of the predecessor state.
            best_cost: Incumbent best cost for the current state.

        Returns:
            ``True`` to skip (treat as infeasible), ``False`` to evaluate
            exactly.
        """
        cached = self._cache.get((segment_idx, prev_state, curr_state))
        if cached is not None:
            p_feasible, cost = cached
        else:
            pr, pt = trajectory_info[prev_state]
            cr, ct = trajectory_info[curr_state]
            f = handover_features(trajectories, segment_idx, pr, pt, cr, ct)
            if f is None:
                return False  # let the exact evaluator reject it
            t0 = time.perf_counter()
            p, c = self.model.predict(f)
            self.t_model += time.perf_counter() - t0
            p_feasible, cost = float(p[0]), float(c[0])
        self.n_queried += 1

        if p_feasible < self.threshold:
            self.n_skipped_feasibility += 1
            return True
        if self.cost_slack > 0 and acc_cost + self.cost_slack * cost >= best_cost:
            self.n_skipped_cost += 1
            return True
        return False

    def stats(self) -> Dict[str, float]:
        """Return filter settings and counters for metrics logging.

        Returns:
            Dict with the threshold, cost slack, number of queried and
            skipped candidates (total, by feasibility, by cost) and total
            model inference time in seconds.
        """
        return {
            "filter_threshold": self.threshold,
            "filter_cost_slack": self.cost_slack,
            "filter_n_queried": self.n_queried,
            "filter_n_skipped": self.n_skipped_feasibility + self.n_skipped_cost,
            "filter_n_skipped_feasibility": self.n_skipped_feasibility,
            "filter_n_skipped_cost": self.n_skipped_cost,
            "filter_model_time_s": self.t_model,
        }
