# Banknote Fraud Detection - Decision Trees from Scratch

[![License: MIT]](https://github.com/Unique-Divine/Langevin-Dynamics-for-NN-Optimization/blob/main/LICENSE)

[License: MIT]: https://img.shields.io/badge/License-MIT-yellow.svg 

## Summary: ##

The goal of this project was to develop accurate predictive models to solve a binary classification problem: detecting  fraudulent banknotes. I figured that learning to write decision tree algorithms from scratch could serve as an effective technical exercise. Unexpectedly, this [from-scratch implementation][cart.py] ended up outperforming the default trees from Scikit-learn.  
 
- [![View](https://img.shields.io/badge/Jupyter%20nbviewer-View%20notebook-brightgreen?&logo=Jupyter)][nbviewer]: To view the notebook without downloading anything
- Source code for the custom 

[nbviewer]: https://nbviewer.jupyter.org/github/Unique-Divine/Banknote-Fraud-Detection/blob/master/Banknote%20Fraud%20-%20Decision%20Tree.ipynb
[cart.py]: https://github.com/Unique-Divine/Banknote-Fraud-Detection/blob/master/cart.py

<!-- Add graph of results. -->


## Data Description:

The data was downloaded from [this source](https://archive.ics.uci.edu/ml/datasets/banknote+authentication#).
> Extracted from images were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object, gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tools were used to extract features from images.

#### Dataset Attributes:

1. Variance of Wavelet Transformed image (continuous)
2. Skewn of Wavelet Transformed image (continuous)
3. Curtosis of Wavelet Transformed image (continuous)
4. Entropy of image (continuous)
5. Class (integer): Whether or not the banknote is real or fake. 


 