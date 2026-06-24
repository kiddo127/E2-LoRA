# E²-LoRA for Domain-Incremental Learning

DIL implementation based on the [DUCT](https://github.com/Estrella-fugaz/CVPR25-Duct) framework.

## Requirements

```bash
conda env create -f ../../environment.yml
conda activate e2lora
```

Or install dependencies manually:

```bash
pip install -r ../../requirements.txt
```

## Dataset Preparation

We follow the same dataset structure as DUCT. Update `data_path` in config files to point to your data directory.

Supported datasets: DomainNet, Office-Home.

### DomainNet

Organize as:
```
/path/to/DomainNet/
├── clipart_train.txt
├── clipart_test.txt
├── infograph_train.txt
├── infograph_test.txt
├── painting_train.txt
├── painting_test.txt
├── quickdraw_train.txt
├── quickdraw_test.txt
├── real_train.txt
├── real_test.txt
├── sketch_train.txt
└── sketch_test.txt
```

Each txt file contains lines of: `<relative_image_path> <label>`

### Office-Home

```
/path/to/OfficeHome/
├── Art/
├── Clipart/
├── Product/
└── Real_World/
```

The train/test split will be automatically generated.

Refer to [DUCT](https://github.com/Estrella-fugaz/CVPR25-Duct) for detailed dataset download instructions.

## Training

Update `data_path` in the config file before running:

```bash
# DomainNet
python main.py --config configs/Template_domainnet_e2lora.json

# Office-Home
python main.py --config configs/Template_officehome_e2lora.json
```

Config files are in `configs/`. Key parameters:
- `init_cls`: number of classes per domain
- `increment`: number of classes added per domain
- `total_sessions`: number of domains
- `epochs`: training epochs per task
- `ca_epochs`: classifier alignment epochs
- `bcb_lrscale`: backbone learning rate scale

## Results

![DIL Results](figs/dil_results.png)

## Code Structure

```
domain_incremental_learning/
├── methods/e2lora.py  # E²-LoRA Learner (stage1 training, PCA-LoRA, classifier alignment)
├── methods/base.py    # Base learner class
├── models/vit_inc.py  # Network definition (LoRAMlp, LoRAAttention, E2LoRANet)
├── models/linears.py  # Classifier head
├── utils/data.py      # Dataset classes
├── utils/data_manager.py
├── utils/factory.py   # Model factory
├── utils/toolkit.py   # Utility functions
├── trainer.py         # Training loop
├── main.py            # Entry point
└── configs/           # Experiment configs (*.json)
```

## Acknowledgments

Built upon [DUCT](https://github.com/Estrella-fugaz/CVPR25-Duct). Thanks to the DUCT authors for their well-structured DIL implementation.
