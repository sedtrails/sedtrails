"""
SedTRAILS flow-field reading example.

This script demonstrates the lower-level workflow for inspecting input forcing:

1. Convert a hydrodynamic model file to SedTRAILS' internal data object.
2. Create a FieldDataRetriever for temporal interpolation.
3. Retrieve a named flow field at several times.
4. Plot each retrieved flow field.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sedtrails.particle_tracer.data_retriever import FieldDataRetriever
from sedtrails.transport_converter import FormatConverter

from visualization_utils import plot_flow_field


# ===== CONFIGURATION =====

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent

FILE_PATH = REPO_ROOT / 'sample-data' / 'inlet_sedtrails.nc'
INPUT_FORMAT = 'fm_netcdf'
REFERENCE_DATE = '1970-01-01'
FLOW_FIELD_NAME = 'depth_avg_flow_velocity'
N_PLOTS = 5
OUTPUT_DIR = EXAMPLES_DIR / 'results' / 'flowfields'


def main() -> int:
    """Run the example."""
    if not FILE_PATH.exists():
        raise FileNotFoundError(
            f'Input file not found: {FILE_PATH}. '
            'Download/create the forcing file, or edit FILE_PATH in this example.'
        )

    # ===== STEP 1: Convert the input file =====
    print('\n=== STEP 1: Loading and converting input data ===')
    print(f'Processing file: {FILE_PATH}')

    format_config = {
        'input_file': str(FILE_PATH),
        'input_format': INPUT_FORMAT,
        'reference_date': REFERENCE_DATE,
        'morfac': 1.0,
    }
    converter = FormatConverter(format_config)
    sedtrails_data = converter.convert_to_sedtrails()

    times = np.asarray(sedtrails_data.times, dtype=float)
    print(f'Time range: {times[0]:.0f}s to {times[-1]:.0f}s')
    print(f'Total timestamps: {times.size}')
    print(f'Spatial points: {np.asarray(sedtrails_data.x).size}')

    # ===== STEP 2: Create a flow-field retriever =====
    print('\n=== STEP 2: Creating FieldDataRetriever ===')
    retriever = FieldDataRetriever(sedtrails_data)
    print(f'Retriever is ready for flow field {FLOW_FIELD_NAME!r}')

    # ===== STEP 3: Retrieve and plot selected times =====
    print('\n=== STEP 3: Retrieving and plotting flow fields ===')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_count = max(1, min(N_PLOTS, times.size))
    selected_indices = np.linspace(0, times.size - 1, plot_count, dtype=int)

    for plot_number, time_index in enumerate(selected_indices, start=1):
        current_time = float(times[time_index])
        print(f'Plot {plot_number}/{plot_count}: t={current_time:.0f}s')

        # Current signature: get_flow_field(time, flow_field_name)
        flow_data = retriever.get_flow_field(current_time, FLOW_FIELD_NAME)

        fig, _ = plot_flow_field(
            flow_data,
            title=f'{FLOW_FIELD_NAME} at t={current_time:.0f}s',
            downsample=10,
        )
        output_path = OUTPUT_DIR / f'{FLOW_FIELD_NAME}_{plot_number:03d}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'  saved {output_path}')

    print('\nFlow-field reading example completed successfully.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
