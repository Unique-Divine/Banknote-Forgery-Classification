# Banknote Fraud Classification - Decision Trees from Scratch

To view the notebook without downloading anything, [go here](https://nbviewer.jupyter.org/github/Unique-Divine/Banknote-Fraud-Detection/blob/master/Banknote%20Fraud%20-%20Decision%20Tree.ipynb).

## Summary: ##
The goal of this project was to develop accurate predictive models to solve a binary classification problem, detecting  fraudulent banknotes. I figured that classifying the banknotes with decision trees written from scratch could serve as an effective technical exercise.
 
## Data Description:
The data was downloaded from [this source](https://archive.ics.uci.edu/ml/datasets/banknote+authentication#).
> Extracted from images were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object, gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tools were used to extract features from images.

#### Dataset Attributes:
1. Variance of Wavelet Transformed image (continuous)
2. Skewn of Wavelet Transformed image (continuous)
3. Curtosis of Wavelet Transformed image (continuous)
4. Entropy of image (continuous)
5. Class (integer): Whether or not the banknote is real or fake. 