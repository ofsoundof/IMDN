## [NTIRE 2022 Workshop and Challenge](https://data.vision.ee.ethz.ch/cvl/ntire22/) @ CVPR 2022
## Efficient Super-Resolution Challenge


Jointly with NTIRE workshop we have a challenge on Efficient Super-Resolution, that is, the task of super-resolving (increasing the resolution) an input image with a magnification factor x4 based on a set of prior examples of low and corresponding high resolution images. The challenge has three tracks.

**[Track 1: Parameters](https://competitions.codalab.org/competitions/20167)**, the aim is to obtain a network design / solution with the lowest amount of parameters while being constrained to maintain or improve the PSNR result and the inference time (runtime) of IMDN ([Hui et al, 2017](https://arxiv.org/abs/1909.11856)).

**[Track 2: Inference](https://competitions.codalab.org/competitions/20168)**, the aim is to obtain a network design / solution with the lowest inference time (runtime) on a common GPU (ie. Titan Xp) while being constrained to maintain or improve over IMDN ([Hui et al, 2017](https://arxiv.org/abs/1909.11856)) in terms of number of parameters and the PSNR result.

**[Track 3: Fidelity](https://competitions.codalab.org/competitions/20169)**, the aim is to obtain a network design / solution with the best fidelity (PSNR) while being constrained to maintain or improve over IMDN ([Hui et al, 2017](https://arxiv.org/abs/1909.11856)) in terms of number of parameters and inference time on a common GPU (ie. Titan Xp).

## Baseline model (IMDN)

* Number of parameters: 893,936 (0.89M)

    ```python
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    ```

* Average PSNR on validation data: 29.13 dB

* Average inference time (Titan Xp) on validation data: 0.10 second 

    Note: The best average inference time among three trials is selected.

Run [test_demo.py](test_demo.py) to test the model

## How to use the code during test phase.

1. `git clone https://github.com/ofsoundof/IMDN`
2. Put your model script under the `models` folder.
3. Put your pretrained model under the `model_zoo` folder.
4. Modify `model_path` in `test_demo.py`. Modify
the imported models.
5. `python test_demo.py`
