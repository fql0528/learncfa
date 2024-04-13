# Joint Learning of RGBW Color Filter Arrays and Demosaicking
#### Chenyan Bai, Faqi Liu, Jia Li
>RGBW color filter arrays (CFAs) have gained widespread attention for their superior performance in low-light conditions. Most existing demosaicking methods are tailored for specific RGBW CFAs or involve manual joint design with CFAs. The close relationship between sampling and reconstruction means that restricting the search space through predefined CFAs and demosaicking methods severely limits the ability to achieve optimal performance. In this paper, we propose a new approach for joint learning of RGBW CFA and demosaicking. This approach can simultaneously learn optimal CFAs and demosaicking methods of any size, while also being capable of reconstructing mosaicked images of any size. We use a surrogate function and arbitrary CFA interpolation to ensure end-to-end learning of the RGBW CFA. We also propose a dual-branch fusion reconstruction network that utilizes the W channel to guide the reconstruction of R, G, and B channels, reducing color errors while preserving more image details. Extensive experiments demonstrate the superiority of our proposed method.

## Environment installation
We use PyTorch for training, and use custom convolutional layers to simulate RGBW CFA design and demosaicing.Assuming you have [Anaconda](https://www.anaconda.com/products/individual#Downloads):
```bash
conda create -n learncfa python=3.7
conda activate learncfa
pip install requirement.txt
```
## Prepare Data
We trained, validated, and tested using the [Gehler-Shi dataset](https://www2.cs.sfu.ca/~colour/data/shi_gehler/) dataset. As the original Gehler-Shi dataset consists of RAW format images, which were unavailable during the development of this code, we converted them into normalized 8-bit PNG files. These files can be directly inputted into the autoencoder for training, validation, and testing purposes.
The preprocessed Gehler-Shi dataset:
```bash
cd canon_data/datasets
```
During training, we cropped the images into 128×128 blocks for training, totaling 107472 image blocks. However, validation and testing were conducted on full-size images.
Image cropping：
```bash
python canon_data/crop_batch.py
```
## Training
* Autoencoder model (encoder network + decoder network)
  * To jointly train the RGBW CFA and demosaicking network, please execute:
  ```bash
python train/train_learn_rgbw_to_rgb.py  --split ‘train’
  ```
