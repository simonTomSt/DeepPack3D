# DeepPack3D

**DeepPack3D** is a Python-based 3D bin-packing software optimized for robotic palletization systems. It supports various methods, including reinforcement learning (RL) and heuristic baselines, and provides flexible options for data input and visualization.

<img src="./outputs/0_0_0.jpg" width="200"><img src="./outputs/0_1_0.jpg" width="200"><img src="./outputs/0_44_0.jpg" width="200">

## Custom Configuration

This version has been configured for specific bin dimensions and package types:

**Bin Size:** 100 × 140 × 120 units (Width × Height × Depth)

**Package Types (16 types):**

- 24 × 8 × 17 (scaled from 240×80×170 mm)
- 24 × 16 × 17 (scaled from 240×160×170 mm)
- 34 × 8 × 24 (scaled from 340×80×240 mm)
- 34 × 16 × 24 (scaled from 340×160×240 mm)
- 40 × 10 × 40 (scaled from 400×100×400 mm)
- 40 × 12 × 40 (scaled from 400×120×400 mm)
- 40 × 14 × 40 (scaled from 400×140×400 mm)
- 40 × 16 × 40 (scaled from 400×160×400 mm)
- 40 × 18 × 40 (scaled from 400×180×400 mm)
- 40 × 20 × 40 (scaled from 400×200×400 mm)
- 60 × 10 × 40 (scaled from 600×100×400 mm)
- 60 × 12 × 40 (scaled from 600×120×400 mm)
- 60 × 14 × 40 (scaled from 600×140×400 mm)
- 60 × 16 × 40 (scaled from 600×160×400 mm)
- 60 × 18 × 40 (scaled from 600×180×400 mm)
- 60 × 20 × 40 (scaled from 600×200×400 mm)

## Features

- Supports multiple methods: Reinforcement Learning (**RL**), Best Lookahead (**BL**), Best Area Fit (**BAF**), Best Shortest Side Fit (**BSSF**), and Best Longest Side Fit (**BLSF**).
- **Custom package types:** Predefined set of 16 package dimensions optimized for specific use cases.
- **Flexible bin dimensions:** Configurable bin size (currently 100×140×120).
- Provides options for data generation, user input, or loading from a file.
- Offers training and testing modes for RL.
- Includes visualization to monitor the packing process.
- GPU-enabled for accelerated RL training and inference.

## Quick Start

### Test Custom Configuration

```bash
# Test all heuristic methods with custom packages
python test_custom.py

# Use Best Area Fit with visualization
python deeppack3d.py baf 1 --visualize --verbose=1

# Use custom packages (default mode)
python deeppack3d.py bl 1 --data=custom --visualize --verbose=1

# Use file input with custom packages
python deeppack3d.py bssf 1 --data=file --path=./custom_packages.txt --visualize --verbose=1
```

### Available Data Sources

- `--data=custom` (default): Uses predefined custom package types
- `--data=file`: Load packages from file (e.g., custom_packages.txt)
- `--data=input`: Interactive input mode
- `--data=generated`: Auto-generated random packages

## Installation

The software runs in **Python 3.10** and **Tensorflow 2.10.0**.

You can refer to Tensorflow official website for the installation guideline.

> https://www.tensorflow.org/install

### From repository

1. Clone the repository:

```bash
git clone https://github.com/zgtcktom/DeepPack3D.git
cd DeepPack3D
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure you have a compatible GPU environment if using RL methods.

### From wheel

Alternatively, you can create a distributable package.

1. Creating a wheel

```bash
python setup.py sdist bdist_wheel
```

2. Install wheel

```bash
pip install ./dist/DeepPack3D-0.1.0-py3-none-any.whl
```

3. Run python module

```bash
python -m deeppack3d rl 5 --n_iterations=-1 --data=file --path=./input.txt --verbose=1
```

## Usage

### Command-Line Interface

You can run DeepPack3D directly from the command line:

```bash
python deeppack3d.py <method> <lookahead> [options]
```

#### Example Command:

```bash
python deeppack3d.py bl 5 --n_iterations=-1 --data=file --path=./input.txt --verbose=1
```

#### Arguments:

- `<method>`: Choose the method from `{"rl", "bl", "baf", "bssf", "blsf"}`.
- `<lookahead>`: Set the lookahead value.

#### Options:

- `--data`: Input source (`generated`, `input`, or `file`).
  - Default: `generated`.
- `--path`: File path (used only if `--data=file`).
  - Default: `None`.
- `--n_iterations`: Number of iterations (used only if `--data=generated`).
  - Default: `100`.
- `--seed`: Random seed for reproducibility (used only if `--data=generated`).
  - Default: `None`.
- `--verbose`: Verbose level (`0` for silent, `1` for standard, `2` for detailed).
  - Default: `1`.
- `--train`: Enable training mode (used only with `method=rl`).
  - Default: `False`.
- `--batch_size`: Batch size (used only with `--train`).
  - Default: `32`.
- `--visualize`: Enable visualization mode.
  - Default: `False`.

### Library

You can also import DeepPack3D as a Python library to integrate with other systems or workflows.

Example:

```python
from deeppack3d import deeppack3d

for result in deeppack3d('rl', 5, n_iterations=-1, data='file', path='./input.txt', verbose=0):
	if result is None:
	 	print('new bin')
		continue
	_, (x, y, z), (w, h, d), _ = result
	print(f'placing item ({w}, {h}, {d}) at ({x}, {y}, {z})')
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the software.

## Citation/Reference

If you find this package useful, please feel free to cite our following work.

Tsang, Y. P., Mo, D. Y., Chung, K. T., & Lee, C. K. M. (2025). A deep reinforcement learning approach for online and concurrent 3D bin packing optimisation with bin replacement strategies. _Computers in Industry_, _164_, 104202. https://doi.org/10.1016/j.compind.2024.104202

## License

This project is licensed under the MIT License. See the LICENSE file for details.
