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
  ```
  python train/train_learn_rgbw_to_rgb.py  --split 'train'
  ```
* Decoder network
  * Fix the learned RGBW CFA or representative RGBW CFA and train the decoder network. Please run:
   ```
   python train/train_rgbw_to_rgb.py  --split 'train'
   ``` 
## Validating
During validation, please ensure that the parameter settings remain exactly the same as during training.
* Autoencoder model (encoder network + decoder network)
  * To verify the performance of the pre-trained autoencoder network and adjust the network hyperparameters, please run:
  ```
  python Test/ Test_learn_rgbw_to_rgbw.py  --test_epoch lrgbw_20 --split 'test'
  ```
 * Decoder network
   * To fix the learned RGBW CFA or representative RGBW CFA and verify the performance of the pre-trained decoder network, adjusting network hyperparameters, please run:
   ```
   python Test/ Test_rgbw_to_rgbw.py  --test_epoch ourrgbw_20 --split 'val'
   ```
## Testing
During testing, please ensure that the parameter settings remain exactly the same as during training.
  * Autoencoder model (encoder network + decoder network)
    * To test the performance of the pre-trained autoencoder network, please run:
    ```
    python Test/ Test_learn_rgbw_to_rgbw.py  --test_epoch lrgbw_20 --split 'test'
    ```
* Decoder network
  * To fix the learned RGBW CFA or representative RGBW CFA and test the performance of the pre-trained decoder network, please run:
  ```
  python Test/ Test_rgbw_to_rgbw.py --test_epoch ourrgbw_20 --split 'test'
  ```

## Citations
If our autoencoder network has been helpful for your research or work, please consider citing our paper on the autoencoder network.
```
@article{bai4753575joint,
  title={Joint Learning of Rgbw Color Filter Arrays and Demosaicking},
  author={Bai, Chenyan and Liu, Faqi and Li, Jia},
  journal={Available at SSRN 4753575}
}
```
## Contact
If you have any questions, please contact jiali.gm@gmail.com  or fql_2021@126.com

