# Real-time Visual Saliency by Division of Gaussians - Reference Implementation

![Python - PEP8](https://github.com/tobybreckon/DoG-saliency/workflows/Python%20-%20PEP8/badge.svg)
![Saliency Test](https://github.com/tobybreckon/DoG-saliency/workflows/Saliency%20Test/badge.svg)

Tested using Python 3.8.2 and [OpenCV 4.3.0](http://www.opencv.org)

![DOG-Saliency](https://github.com/tobybreckon/DoG-saliency/blob/master/test/true_saliency_maps/fig_2_saliency.png)|![DOG-Saliency](https://github.com/tobybreckon/DoG-saliency/blob/master/test/samples/fig_2.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Exemplar Real-time Salient Object Detection Using DOG Saliency

## Abstract:

_"This paper introduces a novel method for deriving visual saliency maps in real-time without compromising the quality of
the output. This is achieved by replacing the computationally
expensive centre-surround filters with a simpler mathematical
model named Division of Gaussians (DIVoG). The results are
compared to five other approaches, demonstrating at least six
times faster execution than the current state-of-the-art whilst
maintaining high detection accuracy. Given the multitude of
computer vision applications that make use of visual saliency
algorithms such a reduction in computational complexity is
essential for improving their real-time performance."_

[[Katramados, Breckon, In Proc. International Conference on Image Processing, IEEE, 2011](https://breckon.org/toby/publications/papers/katramados11salient.pdf)]

---

## Reference implementation:

This Saliency Map generator uses the Division of Gaussians (DIVoG / DoG) approach to produce real-time saliency maps. Put simply this algorithm performs the following three steps (as set out in the original DIVoG research paper):
- Bottom-up construction of Gaussian pyramid
- Top-down construction of Gaussian pyramid based on the output of Step 1
- Element-by element division of the input image with the output of Step 2

This repository contains `saliencyDoG.py` which corresponds to the Division of Gaussians algortihm as defined in [Katramados / Breckon, 2011]. `demo.py` is simply an example of usage of the SaliencyDoG library (supported by `camera_stream.py`, providing an unbuffered video feed from a live camera input), which demonstrates saliencyDoG using either live or input video, and a live result. Each frame is processed sequentially, producing the real-time saliency map. `test.py` should be used to verify correct versions of libraries are installed, before using the library.

`saliencyDoG.py` contains class `SaliencyDoG`. An object for a salience mapper can be created (with specific options), and used on various images, e.g.
```python
from saliencyDoG import SaliencyDoG
import cv2

img = cv2.imread('dog.png')
saliency_mapper = SaliencyDoG(pyramid_height=5, shift=5, ch_3=False,
                              low_pass_filter=False, multi_layer_map=False)
img_saliency_map = saliency_mapper.generate_saliency(img)
```
where parameters:
- `pyramid_height` - n as defined in [Katramados / Breckon 2011] - default = 5
- `shift` - k as defined in [Katramados / Breckon 2011] - default = 5
- `ch_3` - process colour image on every channel (approximetly 3x slower) - default = False
- `low_pass_filter` - toggle low pass filter - default = False
- `multi_layer_map` - the second version of the algortihm as defined in [Katramados / Breckon 2011] (significantly slower, with simmilar results) - default = False

The SaliencyDoG class makes use of the [Transparent API (T-API)](https://www.learnopencv.com/opencv-transparent-api/), to make use of any possible hardware acceleration

---

## Instructions to use:

To download and test the supplied code do:

```
$ git clone https://github.com/tobybreckon/DoG-saliency.git
$ cd DoG-saliency
$ python3.x -m pip install -r requirements.txt
$ pytest test.py
```
Ensure that all tests are passed before proceeding. If any tests fail, ensure you have installed the modules from `requirements.txt` and are using at least python 3.7.5 and OpenCv 4.2.0. 

Subsequently run the following command to obtain real-time saliency output from a connected camera or video file specified on the command line:

```
$ python3.x demo.py [-h] [-c CAMERA_TO_USE] [-r RESCALE] [-fs] [-g] [-l] [-m] [video_file]
```

positional arguments:
-   `video_file`&nbsp;&nbsp;specify optional video file

optional arguments:
-   `-h`&nbsp;&nbsp;show help message and exit
-   `-c CAMERA_TO_USE`&nbsp;&nbsp;specify camera to use (int) - default = 0
-   `-r RESCALE`&nbsp;&nbsp;rescale image by this factor (float) - default = 1.0
-   `-fs`&nbsp;&nbsp; optionally run in full screen mode
-   `-g`&nbsp;&nbsp; optionally process frames as grayscale
-   `-l`&nbsp;&nbsp; optionally apply a low_pass_filter to saliency map
-   `-m`&nbsp;&nbsp; optionally use every pyramid layer in the production of the saliency map

During run-time, keyboard commands `x` will quit the program, `f` will toggle fullscreen, and `s` will toggle between saliency mapping and the original input image frames.

---

## Example video:

[![Examples](https://img.youtube.com/vi/3oeuWO7SlvQ/0.jpg)](https://www.youtube.com/watch?v=3oeuWO7SlvQ)


Video Example - click image above to [play](https://www.youtube.com/watch?v=3oeuWO7SlvQ).

---

## References:

If you are making use of this work in any way please reference the following articles in any report, publication, presentation, software release or any other associated materials:

[Real-time Visual Saliency by Division of Gaussians](https://breckon.org/toby/publications/papers/katramados11salient.pdf)
(Katramados, Breckon), In Proc. International Conference on Image Processing, IEEE, 2011.
```
@InProceedings{katramados11salient,
  author    =    {Katramados, I. and Breckon, T.P.},
  title     = 	 {Real-time Visual Saliency by Division of Gaussians},
  booktitle = 	 {Proc. Int. Conf. on Image Processing},
  pages     = 	 {1741-1744},
  year      = 	 {2011},
  month     = 	 {September},
  publisher =    {IEEE},
  url       = 	 {https://breckon.org/toby/publications/papers/katramados11salient.pdf},
  doi       = 	 {10.1109/ICIP.2011.6115785},
}
```

For non-commercial use (i.e. academic, non-for-profit and research) the (very permissive) terms of the MIT free software [LICENSE](LICENSE) must be adhered to.

For commercial use, the Division of Gaussians (DIVoG / DoG) saliency detection algorithm is patented (WIPO reference: [WO2013034878A3](https://patents.google.com/patent/WO2013034878A3/)) and available for licensing via [Cranfield University](https://www.cranfield.ac.uk/).

### Acknowledgements:

Ryan Lail, this reference implementation of [Katramados / Breckon, 2011], Durham University, July 2020.

---
