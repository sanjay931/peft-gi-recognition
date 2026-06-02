# Gastrointestinal Image Classification with ConvNeXt-V2 and LoRA

Parameter-efficient fine-tuning of ConvNeXt-V2-Base for multi-class gastrointestinal endoscopy image classification using Low-Rank Adaptation (LoRA). The project covers two datasets — Kvasir-Capsule and GastroVision — trained and evaluated in Google Colab.



## Datasets

**Kvasir-Capsule** — A capsule endoscopy dataset containing images of the gastrointestinal tract across multiple pathological and normal findings.

**GastroVision** — An endoscopy benchmark with 20 anatomical and pathological categories spanning the upper and lower GI tract.

Both datasets are expected to reside in Google Drive under the paths configured in each notebook.

---

## Architecture

**Backbone:** `convnextv2_base` via [timm](https://github.com/huggingface/pytorch-image-models) with ImageNet-22K pretrained weights. LoRA adapters are injected via [PEFT](https://github.com/huggingface/peft); only the adapter matrices and classification head are updated during training.

| Parameter | Value |
|---|---|
| Rank (`r`) | 16 |
| Alpha (`lora_alpha`) | 32 |
| Target modules | `qkv`, `fc1`, `fc2` |
| Dropout | 0.1 |
| Bias | none |

---

## Requirements

```
torch
torchvision
timm
peft
scikit-learn
matplotlib
tqdm
numpy
grad-cam
captum
opencv-python
```

All dependencies are installed within each notebook via `pip`.

---

## Setup

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Enable a GPU runtime: **Runtime > Change runtime type > T4 GPU** (or higher).
3. Run the first cell to mount Google Drive and create checkpoint directories.
4. Place your dataset in Google Drive following the structure below.

---

## Data Preparation

Datasets must follow the `ImageFolder` convention with `train`, `val`, and `test` splits:

```
dataset_root/
    train/
        class_name_1/
        class_name_2/
    val/
        class_name_1/
        class_name_2/
    test/
        class_name_1/
        class_name_2/
```

| Split | Transforms |
|---|---|
| Train | Resize 256x256, RandomCrop 224, RandomHorizontalFlip, RandomRotation(15), ColorJitter, Normalize |
| Val / Test | Resize 224x224, Normalize |

---


## Evaluation

The best checkpoint is evaluated on the test split. Reported metrics: Accuracy, Weighted F1, Matthews Correlation Coefficient (MCC), Precision, Recall, and a per-class classification report.

---

## Visualization

**Kvasir-Capsule** — Grad-CAM activations from the last block of the final ConvNeXt stage, rendered as an overlay with the predicted class and confidence.

**GastroVision** — SmoothGrad saliency maps (via [Captum](https://captum.ai/)) compared across three architectures: InceptionV3, DenseNet121, and ConvNeXt-V2 + LoRA. Results are displayed as a grid with the original image and per-model heatmaps.
