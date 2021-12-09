# CS1470 Final Project - Fake Finders
Jovan Kemp (jkemp1) & Kei Yoshida (kyoshid1)

## Check-in
Click these links for:
- [Check-in #1](https://github.com/kyoshida14/CS1470_final_project/blob/main/checkin1.md)
- [Check-in #2](https://github.com/kyoshida14/CS1470_final_project/blob/main/checkin2.md)

## Main files
- train.py : train our model with various categories of images using resnet50model.py (Keras model) or our coded resnet.py (ResNet50 with error).
- eval.py : evaluate the model (produce final loss & accuracy)
- demo.py : demo to test with 1) vanilla ResNet50 (pretrained with ImageNet), 2) our model, and 3) the model by Wang et al. (2020) (a from PyTorch to Keras). For Method 3, the datasets of weights from the original paper (Wang et al., 2020) are too big to upload, but they can be downloaded [HERE](https://github.com/PeterWang512/CNNDetection/tree/master/weights).
- Our model can be found HERE.
- Images used to train can be found HERE; images used to test can be found HERE.
