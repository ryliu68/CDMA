## About
This repository contains the official PyTorch implementation of our paper: [**"CDMA: Query-Efficient Black-box Attack to Deep Neural Networks with Conditional Diffusion Models"**](https://ieeexplore.ieee.org/abstract/document/10504304/). 

## Usage

### Prerequisites
To run this code, ensure you have the following packages installed:

- `torch>=1.7.0`
- `torchvision>=0.8.1`
- `tqdm>=4.31.1`
- `pillow>=7.0.0`
- `matplotlib>=3.2.2`
- `numpy>=1.18.1`

Installation of these packages can be achieved through pip:
```bash pip install torch torchvision tqdm pillow matplotlib numpy```


### Model Weights

Download the pre-trained weights for the victim models and CDMA from [GoogleDrive](https://drive.google.com/file/d/1IsgrXW4LBrGwbZgzOuhK9BWA5NXB5SDc/view?usp=sharing). After downloading, unzip the file into the root directory of this project.


### Train
```bash scripts/sb_train_cifar10_e16.sh.sh``` 

**Tip:** Ensure the training dataset contains pre-collected adversarial image pairs matched with their corresponding clean images.


### Evaluation
#### - Untargeted Attack:
```bash scripts/attack.sh```

#### - Targeted Attack:
 ```bash scripts/attack_T.sh```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments
The implementation extends the functionality of the [Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models) GitHub repository. Thanks for their excellent works!


## Citation

If you think this work or our codes are useful for your research, please cite our paper via:

```bibtex
@article{liu2024boosting,
  title={Boosting Black-box Attack to Deep Neural Networks with Conditional Diffusion Models},
  author={Liu, Renyang and Zhou, Wei and Zhang, Tianwei and Chen, Kangjie and Zhao, Jun and Lam, Kwok-Yan},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024},
  volume={19},
  pages={5207--5219},
  publisher={IEEE}
}
```