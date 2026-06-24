# E²-LoRA for Class-Incremental Learning

CIL implementation based on the [PyCIL](https://github.com/LAMDA-CL/PyCIL) framework.

## Requirements

```bash
conda env create -f ../environment.yml
conda activate e2lora
```

Or install dependencies manually:

```bash
pip install -r ../requirements.txt
```

## Dataset Preparation

We follow the same dataset structure as PyCIL. Set `CIL_DATA_ROOT` to your data directory:

```bash
export CIL_DATA_ROOT=/path/to/your/data
```

Supported datasets: CIFAR-10, CIFAR-100, ImageNet-R, ImageNet-A, CUB-200, Cars-196, ObjectNet, OmniBenchmark, VTAB.

For ImageNet-R/A, CUB, Cars, ObjectNet, OmniBenchmark, and VTAB, organize data as:
```
$CIL_DATA_ROOT/
├── imagenet-r/train/   # class subfolders
├── imagenet-r/test/
├── cub/train/
├── cub/test/
├── cars196/train/
├── cars196/val/
└── ...
```

CIFAR-10/100 will be automatically downloaded by torchvision.

Refer to [PyCIL](https://github.com/LAMDA-CL/PyCIL) for detailed dataset download instructions.

## Training

```bash
# ImageNet-R, 20 initial classes, 20 incremental classes
python main.py --config=./exps/e2lora_inr_lora.json --seed 1993

# CIFAR-100, 20 initial classes, 20 incremental classes
python main.py --config=./exps/e2lora_cifar_lora_20.json --seed 1993

# ImageNet-A
python main.py --config=./exps/e2lora_ina_lora.json --seed 1993

# CUB
python main.py --config=./exps/e2lora_cub_lora.json --seed 1993

# Cars
python main.py --config=./exps/e2lora_car_lora.json --seed 1993
```

Config files are in `exps/`. Key parameters:
- `init_cls`: number of classes in the first task
- `increment`: number of classes added per task
- `epochs`: training epochs per task
- `ca_epochs`: classifier alignment epochs
- `bcb_lrscale`: backbone learning rate scale

## Results

![CIL Results](figs/cil_results.png)

## Code Structure

```
class_incremental_learning/
├── models/e2lora.py    # E²-LoRA Learner (stage1 training, PCA-LoRA, classifier alignment)
├── models/base.py      # Base learner class
├── utils/inc_net.py    # Network definition (LoRAMlp, LoRAAttention, E2LoRANet)
├── utils/data.py       # Dataset classes
├── utils/data_manager.py
├── utils/factory.py    # Model factory
├── utils/toolkit.py    # Utility functions
├── backbone/linears.py # Classifier head
├── trainer.py          # Training loop
├── main.py             # Entry point
└── exps/               # Experiment configs (*.json)
```

## Acknowledgments

Built upon [PyCIL](https://github.com/LAMDA-CL/PyCIL). Thanks to the PyCIL authors for their well-organized codebase.
