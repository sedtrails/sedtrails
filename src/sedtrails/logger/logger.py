"""
SedTRAILS Logger
================
A global logger for the SedTRAILS Particle Tracer System.
"""

import logging
import os
import sys


def setup_logging(output_dir: str, level: str = 'INFO') -> logging.Logger:
    """
    Configure global logging.
    - Attaches handlers to the top-level 'sedtrails' logger
    - Detects prior setup by checking handler markers
    - Captures warnings and installs a global exception hook

    Parameters
    ----------
    output_dir : str
        Directory where output files are written.
    level : str
        The level value.

    Returns
    -------
    logging.Logger
        Computed value returned by the function.
    """
    os.makedirs(output_dir, exist_ok=True)
    logfile = os.path.join(output_dir, 'log.txt')

    logger = logging.getLogger('sedtrails')

    # Remove handlers, so we do not append but overwrite the log file
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    # If our handlers are already present, skip reconfiguration
    if any(getattr(h, 'is_sedtrails_handler', False) for h in logger.handlers):
        return logger

    # Configure the 'sedtrails' logger (parents: sedtrails.* module loggers)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')

    # File handler
    file_handler = logging.FileHandler(logfile, mode='w')  # Overwrite logs instead of appending
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    file_handler.is_sedtrails_handler = True  # marker to prevent duplicates
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    console_handler.is_sedtrails_handler = True  # marker to prevent duplicates
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    # Install a global exception hook
    install_global_excepthook(logger)

    logger.info('=== LOGGING INITIALIZED ===')
    logger.info(f'Log file: {os.path.abspath(logfile)}')
    return logger


def install_global_excepthook(logger: logging.Logger) -> None:
    """
    Install an excepthook that logs unhandled exceptions.
    Avoids globals by tagging the hook function itself.

    Parameters
    ----------
    logger : logging.Logger
        The logger value.
    """
    # If our hook is already installed, skip
    if getattr(sys.excepthook, '_sedtrails_excepthook', False):
        return

    def _hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.exception('Unhandled exception', exc_info=(exc_type, exc_value, exc_tb))
        # Call original hook (prints trace if configured elsewhere)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    _hook._sedtrails_excepthook = True  # tag for idempotency
    sys.excepthook = _hook


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger. Prefer logging.getLogger(__name__) in modules so logs
    propagate to the 'sedtrails' logger configured above.

    Parameters
    ----------
    name : str | None
        Name of the requested object.

    Returns
    -------
    logging.Logger
        Computed value returned by the function.
    """
    return logging.getLogger(name or 'sedtrails')


def log_simulation_state(logger: logging.Logger, state: dict, level=logging.INFO) -> None:
    """
    Logs the current state of the simulation with human-readable sentences.
    Long messages are split into multiple lines for readability.

    Parameters
    ----------
    logger : logging.Logger
        The logger value.
    state : dict
        The state value.
    level : object
        The level value.
    """
    status = state.get('status', state.get('state', 'unknown'))

    # Create messages, split when necessary
    if status == 'simulation_started':
        command = state.get('command', 'unknown')
        config = state.get('config_file', 'unknown')
        python_ver = state.get('python_version', 'unknown')
        working_dir = state.get('working_directory', 'unknown')

        # Split into multiple log lines
        logger.log(level, '=== SIMULATION START ===')
        logger.log(level, f'Command: {command}')
        logger.log(level, f'Python: {python_ver}')
        logger.log(level, f'Config: {config}')
        logger.log(level, f'Working directory: {working_dir}')

    elif status == 'config_loading':
        config_path = state.get('config_file_path', 'unknown')
        logger.log(level, f'Loading configuration: {config_path}')

    elif status == 'simulation_parameters':
        duration = state.get('duration', 'unknown')
        timestep = state.get('timestep', 'unknown')
        start_time = state.get('start_time', 'unknown')
        strategy = state.get('seeding_strategy', 'unknown')
        output_dir = state.get('output_dir', 'unknown')

        # Format seeding strategy for readability
        formatted_strategy = _format_seeding_strategy(strategy)

        # Split parameters into multiple lines
        logger.log(level, '--- Simulation Parameters ---')
        logger.log(level, f'Duration: {duration}')
        logger.log(level, f'Time step: {timestep}')
        logger.log(level, f'Start time: {start_time}')
        logger.log(level, f'Seeding: {formatted_strategy}')
        logger.log(level, f'Output directory: {output_dir}')

    elif status == 'particles_created' or status == 'particles_initialized':
        count = state.get('count', 1)
        position = state.get('start_position', state.get('position', 'unknown'))
        seeding_strategy = state.get('seeding_strategy', {})

        logger.log(level, f'Created {count} particle(s)')
        logger.log(level, f'Starting position: {position}')

        # Format seeding strategy if provided
        if seeding_strategy and seeding_strategy != {}:
            formatted_strategy = _format_seeding_strategy(seeding_strategy)
            logger.log(level, f'Seeding method: {formatted_strategy}')

    elif status == 'data_conversion_completed':
        timesteps = state.get('num_timesteps', 'unknown')
        field_name = state.get('flow_field_name', 'unknown')
        duration = state.get('simulation_duration_seconds', 'unknown')
        timestep = state.get('simulation_timestep_seconds', 'unknown')

        logger.log(level, '--- Data Conversion Complete ---')
        logger.log(level, f'Timesteps: {timesteps}')
        logger.log(level, f'Flow field: {field_name}')
        logger.log(level, f'Duration: {duration}s')
        logger.log(level, f'Time step: {timestep}s')

    elif status == 'numba_compilation_started':
        grid_x = state.get('grid_size_x', 'unknown')
        grid_y = state.get('grid_size_y', 'unknown')
        logger.log(level, f'Compiling calculator for {grid_x}x{grid_y} grid')

    elif status == 'compilation_complete':
        time_sec = state.get('time_sec', 'unknown')
        logger.log(level, f'Calculator compiled in {time_sec} seconds')

    elif status == 'warmup_complete':
        time_sec = state.get('time_sec', 'unknown')
        logger.log(level, f'JIT warmed up in {time_sec} seconds')

    elif status == 'simulation_progress':
        progress = state.get('progress_pct', 0)
        step = state.get('step', 0)
        position = state.get('position', 'unknown')
        logger.log(level, f'Progress: {progress}% (step {step}) at {position}')

    elif status == 'simulation_complete' or status == 'simulation_completed':
        steps = state.get('total_steps', 'unknown')
        time_sec = state.get('total_time_sec', 'unknown')
        final_pos = state.get('final_position', 'unknown')

        logger.log(level, '=== SIMULATION COMPLETE ===')
        logger.log(level, f'Total steps: {steps}')
        logger.log(level, f'Runtime: {time_sec}s')
        logger.log(level, f'Final position: {final_pos}')

    elif status == 'visualization_saved':
        filename = state.get('file', 'trajectory plot')
        output_path = state.get('output_plot_path', 'unknown')

        logger.log(level, f'Visualization saved: {filename}')
        if output_path != 'unknown':
            logger.log(level, f'Location: {output_path}')

    elif status == 'creating_visualization':
        points = state.get('trajectory_points', 'unknown')
        logger.log(level, f'Creating visualization with {points} trajectory points')

    elif status == 'simulation_failed':
        error_type = state.get('error_type', 'unknown')
        error_message = state.get('error_message', 'unknown')

        logger.log(level, '=== SIMULATION FAILED ===')
        logger.log(level, f'Error type: {error_type}')
        logger.log(level, f'Error message: {error_message}')
        logger.log(level, 'Check log above for full stack trace')

    else:
        # Fallback for unmapped states
        clean_status = status.replace('_', ' ').title()
        logger.log(level, f'Status: {clean_status}')

        # Log each parameter on separate line if many parameters
        params = {k: v for k, v in state.items() if k not in ['status', 'state']}
        if len(params) > 3:  # If more than 3 params, split them
            for k, v in params.items():
                logger.log(level, f'  {k}: {v}')
        elif params:  # If 3 or fewer, keep on one line
            param_str = ', '.join(f'{k}: {v}' for k, v in params.items())
            logger.log(level, f'  {param_str}')


def log_exception(logger: logging.Logger, e: Exception, context: str = None) -> None:
    """
    Logs exceptions with stack trace and context information.

    Parameters
    ----------
    logger : logging.Logger
        Logger used to write the exception details.
    e : Exception
        The exception that occurred
    context : str, optional
        Additional context about where/when the exception occurred
    """
    # Log exception with context
    if context:
        logger.error(f'=== ERROR: {context} ===')
    else:
        logger.error('=== SIMULATION ERROR ===')

    logger.error(f'Exception type: {type(e).__name__}')
    logger.error(f'Exception message: {str(e)}')
    logger.error('Stack trace:', exc_info=(type(e), e, e.__traceback__))
    logger.error('=' * 50)


def _format_seeding_strategy(strategy):
    """Helper function to format seeding strategy for readable logging."""
    if not strategy or strategy == 'unknown':
        return 'none'

    if isinstance(strategy, str):
        return strategy

    if isinstance(strategy, dict):
        formatted_parts = []

        for strategy_type, config in strategy.items():
            if strategy_type == 'point':
                points = config.get('points', [])
                if isinstance(points, list):
                    point_count = len(points)
                    if point_count == 1:
                        formatted_parts.append(f'single point at {points[0]}')
                    else:
                        formatted_parts.append(f'{point_count} points')
                else:
                    formatted_parts.append('point seeding')

            elif strategy_type == 'line':
                formatted_parts.append('line seeding')

            elif strategy_type == 'grid':
                formatted_parts.append('grid seeding')

            elif strategy_type == 'random':
                count = config.get('count', 'unknown')
                formatted_parts.append(f'random ({count} particles)')

            else:
                formatted_parts.append(f'{strategy_type} seeding')

        return ', '.join(formatted_parts)

    return str(strategy)
