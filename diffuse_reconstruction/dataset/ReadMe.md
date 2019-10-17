### Experimental data for the MCNN application to image objects from diffusive reflections

---

The recorded images from screen and camera are stored in 4 folders. Each folder contains 256 image-pairs.
The first 256 image-pairs recorded were stored folder `0`, then folder `1` and `2` and `3` sequentially.
In our MCNN application, the dataset in folder `1`, `2` and `3` are used for training. And the performance is validated with the datset in folder `0`.

These 4 dataset for traning and testing, which are compressed, cropped and left-right flipped, are also available at <http://144.91.70.240/d/25822e291351488780e5/>.

The screenshot and camera capture script is presented in file `video_gen.py` in this folder for readers who want to generate their own dataset.

