import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


def safe_env_step(env: Any, action: Any, retries: int = 3, delay: float = 0.01):
    """Safely call ``env.step(action)`` and retry on intermittent
    RuntimeError: "dictionary changed size during iteration" which can
    occasionally arise inside robosuite's internal robot control logic.

    This keeps the retry/mitigation inside our codebase (no external
    patching of robosuite). The function makes a defensive copy of the
    action before passing it to the environment.
    """
    action_copy = None
    try:
        # Defensive copy for numpy arrays and array-like objects
        import numpy as _np  # local import to avoid top-level dependency surprises
        if isinstance(action, _np.ndarray):
            action_copy = _np.array(action, copy=True)
        else:
            # Fallback: try to create a numpy copy, otherwise use the original
            try:
                action_copy = _np.array(action, copy=True)
            except Exception:
                action_copy = action
    except Exception:
        action_copy = action

    last_exception = None
    for attempt in range(retries):
        try:
            return env.step(action_copy)
        except RuntimeError as e:
            last_exception = e
            msg = str(e)
            if "dictionary changed size during iteration" in msg:
                logger.warning(
                    "Detected intermittent RuntimeError in env.step(): '%s' (attempt %d/%d). Retrying...",
                    msg,
                    attempt + 1,
                    retries,
                )
                time.sleep(delay * (attempt + 1))
                continue
            # Re-raise non-matching runtime errors
            raise
    # Final attempt (let it raise if it still fails)
    if last_exception is not None:
        logger.error("Final attempt to call env.step still failed: %s", last_exception)
    return env.step(action_copy)
