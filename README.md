# Mixed Dynamic Spiking Estimation (MDSE) attack 

Code for the paper, *"Attacking the spike: On the security of spiking neural networks to adversarial examples"*: [Paper here](https://www.sciencedirect.com/science/article/pii/S0925231225021782)

## Abstract

We provide code for Mixed Dynamic Spiking Estimation (MDSE) attack proposed in the paper published in Neurocomputing journal (2025).
In the paper we evaluate the attack among different SNNs, CNNs, and Vision transformers.
MDSE utilizes a dynamic gradient estimation scheme to fully exploit multiple surrogate gradient estimator functions. In addition, our novel attack generates adversarial examples capable of fooling both SNN and non-SNN models simultaneously. 

## Citation

```bibtex
@article{xu2025attacking,
  title={Attacking the spike: On the security of spiking neural networks to adversarial examples},
  author={Xu, Nuo and Mahmood, Kaleel and Fang, Haowen and Rathbun, Ethan and Ding, Caiwen and Wen, Wujie},
  journal={Neurocomputing},
  pages={131506},
  year={2025},
  publisher={Elsevier}
}
```

## Features

- **MDSE Attack**: Novel dynamic gradient estimation attack for SNNs
- **Multi-Model Support**: Attack both CNN and SNN models simultaneously
- **Multiple Surrogate Gradients**: Support for Arctan, Linear, STDB, Erfc, Logistic, etc.
- **Comprehensive Attack Methods**: MDSE, SAGA, MIM, PGD, AutoPGD, Greedy attacks
- **Multiple Datasets**: CIFAR-10, CIFAR-100, ImageNet support
- **Multiple Model Architectures**: VGG, ResNet, ViT, BiT, DietSNN, SpikingJelly

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/MDSE.git
cd MDSE

# Install dependencies
conda env create -f SNN_environment.yml
conda activate pytorch

# Download models
# Download from Google Drive link below and extract to ./ann and ./snn directories

# Run MDSE attack
python MDSE_twomodel.py
```

## Step by Step Guide

1. **Install the packages** listed in the Software Installation Section (see below)
2. **Download the models** from the Google Drive link listed in the Models Section
3. **Move the Models folder** into the directory ".\ann" and ".\snn"
4. **Open the [MDSE_twomodel.py](MDSE_twomodel.py)** file in the Python IDE of your choice. Fill the downloaded model directory. Run the main.

### Usage Examples

**Basic MDSE Attack:**
```python
# Default: CNN + SNN attack on CIFAR-10
AttackMethods.MDSE_two(modelDir1, SNNmodelDir2, dataset='CIFAR10')
```

**Different Datasets:**
```python
# CIFAR-100
AttackMethods.MDSE_two(modelDir1, SNNmodelDir2, dataset='CIFAR100')

# ImageNet (requires ImageNet models)
AttackMethods.SNN_AutoSAGA_imagenet_two(modelDir1, modelDir2, dataset='IMAGENET')
```

**Different Attack Methods:**
```python
# SAGA attack
AttackMethods.SNN_SAGA(modelDir, syntheticmodelDir, dataset, secondcoeff=0.5)

# AutoSAGA attack
AttackMethods.SNN_AutoSAGA_two(modelDir1, modelDir2, dataset)

# SNN vs SNN attack
AttackMethods.SNN_AutoSAGA_two_snnsnn(modelDir1, modelDir2, dataset)
```

## Attack Parameters

### MDSE Attack Parameters
- **epsMax**: 0.031 (maximum perturbation)
- **numSteps**: 40 (number of attack iterations)
- **epsStep**: 0.01 (step size)
- **batchSize**: 100 (batch size for attack)
- **alphaLearningRate**: 100000 (learning rate for gradient estimation)
- **fittingFactor**: 50.0 (fitting factor for optimization)

### Supported Surrogate Gradient Functions
- **Arctan**: Arctangent surrogate gradient
- **Linear**: Linear surrogate gradient
- **STDB**: Spike-timing-dependent plasticity
- **Erfc**: Error function
- **Logistic**: Logistic function
- **Sigmoid**: Sigmoid function
- **PiecewiseQuadratic**: Piecewise quadratic
- **STBPActFun**: STBP activation function
- **FastSigmoid**: Fast sigmoid approximation

### Model Configurations
- **CNN Models**: VGG-16, ResNet-56, ResNet-152
- **SNN Models**: DietSNN, SpikingJelly ResNet, VGG-SNN
- **Transformer Models**: ViT-L-16, ViT-B-16, BiT-M-R50x1, BiT-M-R152x4

## Software Installation

We use the following software packages:
- pytorch==1.12.1
- torchvision==0.13.0
- numpy
- opencv-python
- spikingjelly

**Installation via conda (recommended):**
```bash
# Option 1: Use the simplified environment file
conda env create -f environment.yml
conda activate mdse

# Option 2: Use the full environment file (includes all dependencies)
conda env create -f SNN_environment.yml
conda activate pytorch
```

**Installation via pip:**
```bash
pip install -r requirements.txt
```

**Manual installation:**
```bash
pip install torch==1.12.1 torchvision==0.13.0
pip install numpy opencv-python spikingjelly
pip install timm scikit-learn matplotlib tqdm
```

There are more packages needed to run certain models, you may install if needed. We upload one environment yml file as reference, but there are some unnecessary libs if you only need to run the demo attack.

## Models

We provide the following models:
- **VGG-16** (CNN and SNN versions)
- **Trans-SNN-VGG16-T5** (Transformer-based SNN)
- **RESNET** (ResNet-56, ResNet-152)
- **BP trained SNNs** (Backpropagation-trained SNNs)
- **ViT-L-16** (Vision Transformer Large)
- **BiT-M-R101x3** (BigTransfer ResNet-101)
- **DietSNN** (Diet Spiking Neural Networks)
- **SpikingJelly Models** (Various SNN architectures)

**Download models:** https://drive.google.com/drive/folders/1EyQFF7KSQci4N-DehKyMIp3-WELz7ko9?usp=sharing

For now we provide models for CIFAR10, more pretrained models for CIFAR10, CIFAR100 and ImageNet will be uploaded later.

## System Requirements

- **OS**: Ubuntu 20.04.5 (tested)
- **GPU**: RTX 3090 Ti (recommended)
- **RAM**: 16GB+ (recommended)
- **CUDA**: 11.7+ (required for GPU acceleration)

**Performance Notes:**
- The Adaptive attack has additional hardware requirements
- Attacks on ImageNet's ViT or BiT models will take a long time due to the very small batch size
- CIFAR-10 attacks typically complete in 10-30 minutes
- ImageNet attacks may take several hours

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory:**
```python
# Reduce batch size
batchSize = 32  # instead of 100
```

**2. Model Loading Errors:**
- Ensure model files are in correct directories (./ann and ./snn)
- Check model file paths in MDSE_twomodel.py

**3. Import Errors:**
```bash
# Install missing packages
pip install spikingjelly
pip install timm  # for transformer models
```

**4. Performance Issues:**
- Use smaller batch sizes for limited GPU memory
- Reduce numSteps for faster attacks
- Use fewer surrogate gradient functions

### Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed error information
4. Contact the authors

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or concerns please contact the author at: nux219@lehigh.edu

## Acknowledgments

- SpikingJelly framework for SNN implementations
- PyTorch team for the deep learning framework
- Vision Transformer and BigTransfer model implementations
