"""Runnable example for the SedTRAILS logging helpers."""

from pathlib import Path

from sedtrails.logger.logger import log_exception, log_simulation_state, setup_logging


def run_simulation(output_dir: str | Path = "logs") -> None:
    """Run a small logging demonstration.

    Parameters
    ----------
    output_dir : str or pathlib.Path, optional
        Directory where ``setup_logging`` writes ``log.txt``.
    """
    logger = setup_logging(str(output_dir))
    try:
        state = {
            "status": "simulation_progress",
            "step": 1,
            "progress_pct": 50,
            "position": "(10.0, 20.0)",
        }
        log_simulation_state(logger, state)

        # Simulate a failure
        state["step"] = 2
        state["progress_pct"] = 75
        state["position"] = "(15.0, 25.0)"
        log_simulation_state(logger, state)

        raise ValueError("Unexpected simulation failure!")  # Intentional error

    except Exception as e:
        log_exception(logger, e, context="logger example")

# Run the simulation
if __name__ == "__main__":
    run_simulation()
