## PyTorch-1D-ACGAN-Data-Augmentation

This code is based on the reference codes linked below.

#### [reference ](https://towardsdatascience.com/understanding-acgans-with-code-pytorch-2de35e05d3e4)

## How to run the code

please excute the **ACGAN_augmentator_and_classifier_1d.py**


## About the file

This code is for **1-D array data augmentation** 

The given data in 'data' directory is simple data for training and testing this code.

The 'image' floder receives the results of augmented data.

The data class consists of 0 to 6.

Therfore, if you want to generate the specific class of data, 

You would do it like below code.

Suppose wanting to generate class 6.

```python
eval_label = 6*np.ones(batch_size).astype('int64')
```

(above code is on line 32 of **ACGAN_classifier_1d.py**)

## Data Augmentation Results with Plotting

You can check the results in **image** folder.

![20210504_161953](https://user-images.githubusercontent.com/71545160/116971504-a24ec100-acf4-11eb-8321-cde0ebc28b26.png)
