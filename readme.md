#### MDCNN-I

This project is a idea demonstration of the Multi-scale Deep Convolutional Neural Networks.
For the first stage of training, MDCNN guarantees quick convergence, then degenerates into a U-Net at the second stage.

Example of Usage:

```python
# download some images to folder `images` for the training set, size should no less than `512X512`, then train model with
python3 ./train.py
```
