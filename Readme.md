### README

```markdown
# SUMO-RL Traffic Signal Control

This project benchmarks traffic signal control agents using the SUMO (Simulation of Urban MObility) traffic simulator. It supports various reinforcement learning agents and baseline methods for single-intersection traffic control.

## Requirements

- Python 3.8+
- SUMO (Simulation of Urban MObility)

## Installation

### 1. Install SUMO

#### Windows
1. Download the latest SUMO release from the [SUMO website](https://sumo.dlr.de/docs/Downloads.php).
2. Extract the downloaded archive to a directory of your choice.
3. Add the `bin` directory of SUMO to your system's PATH:
   - Open the Start Menu and search for "Environment Variables".
   - Under "System Variables", find the `Path` variable and click "Edit".
   - Add the path to the `bin` directory of your SUMO installation (e.g., `C:\path\to\sumo\bin`).
4. Set the `SUMO_HOME` environment variable:
   - Create a new environment variable named `SUMO_HOME` and set its value to the root directory of your SUMO installation (e.g., `C:\path\to\sumo`).

#### Linux
1. Install SUMO using your package manager or build it from source:
   ```bash
   sudo apt update
   sudo apt install sumo sumo-tools
   ```
   Alternatively, download the latest release from the [SUMO website](https://sumo.dlr.de/docs/Downloads.php) and follow the build instructions.
2. Add the following to your shell configuration file (e.g., `.bashrc` or `.zshrc`):
   ```bash
   export SUMO_HOME=/path/to/sumo
   export PATH=$SUMO_HOME/bin:$PATH
   ```
3. Reload your shell configuration:
   ```bash
   source ~/.bashrc
   ```

### 2. Install Python Dependencies

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/sumo-rl.git
   cd sumo-rl
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Enable `libsumo`

To use `libsumo` (a faster alternative to `traci`), set the following environment variable:
```bash
export LIBSUMO_AS_TRACI=1
```
On Windows, you can set this variable in the command prompt before running the script:
```cmd
set LIBSUMO_AS_TRACI=1
```

## Usage

Run the main script to train and evaluate agents:
```bash
python main.py
```

For more options, use:
```bash
python main.py --help
```

## Dependencies

- `matplotlib`: For plotting training and evaluation results.
- `torch`: For tensor operations and logging with TensorBoard.
- `tqdm`: For progress bars.
- `sumo-rl`: Custom SUMO-based RL environment.
- `libsumo`: Faster alternative to `traci` for SUMO communication.

## Notes

- Ensure that the `SUMO_HOME` environment variable is correctly set.
- Use `libsumo` for better performance during simulations.
