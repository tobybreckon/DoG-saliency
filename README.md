# Real-time Visual Saliency by Division of Gaussians - Reference Implementation

![Python - PEP8](https://github.com/tobybreckon/DoG-saliency/workflows/Python%20-%20PEP8/badge.svg)

Tested using Python 3.7.5 and [OpenCV 4.2.0](http://www.opencv.org)

![DOG-Saliency](https://github.com/tobybreckon/DoG-saliency/blob/development/test/true_saliency_maps/fig_2_saliency.png)
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

This Saliency Map generator uses the Division of Gaussians approach. Simply this algorithm performs the following three steps: 
- Bottom-up construction of Gaussian pyramid
- Top-down construction of Gaussian pyramid based on the output of Step 1
- Element-by element division of the input image with the output of Step 2

This repository contains `saliencyDoG.py` which corresponds to the Division of Gaussians algortihm as defined in the paper. `demo.py` is simply an example of usage of the SaliencyDoG library (supported by `camera_stream.py`, providing an unbuffered video feed from a live camera input). `test.py` should be used to verify correct versions of libraries are installed, before using the library.

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


---

## Instructions to use:

To download and test the supplied code do:

```
$ git clone https://github.com/tobybreckon/DoG-saliency.git
$ cd DoG-saliency
$ python3.x -m pip install -r requirements.txt
$ pytest test.py
$ python demo.py
```

where ....

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

For non-commercial use (i.e. academic, non-for-profit and research) the (very permissive) terms of the MIT free software [LICENSE](LICENSE) must be adhered to. For commercial use, the DOG saliency algorithm itself is patented (WIPO reference: [WO2013034878A3](https://patents.google.com/patent/WO2013034878A3/)) and available for licensing via [Cranfield University](https://www.cranfield.ac.uk/).

### Acknowledgements:

Ryan Lail, this reference implementation of [Katramados / Breckon, 2011], Durham University, July 2020.

---
