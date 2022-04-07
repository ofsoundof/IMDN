## [NTIRE 2022 Workshop and Challenge](https://data.vision.ee.ethz.ch/cvl/ntire22/) @ CVPR 2022
## Efficient Super-Resolution Challenge


Jointly with NTIRE workshop we have a challenge on Efficient Super-Resolution, that is, the task of super-resolving (increasing the resolution) an input image with a magnification factor x4 based on a set of prior examples of low and corresponding high resolution images. The challenge has three tracks.

**Track 1: Inference Runtime**, the aim is to obtain a network design / solution with the lowest inference time (runtime) on a common GPU (ie. Titan Xp) while being constrained to maintain or improve over IMDN ([Hui et al, 2017](https://arxiv.org/abs/1909.11856)) in terms of number of parameters and the PSNR result.

**Track 2: Model Complexity (Parameters and FLOPs)**, the aim is to obtain a network design / solution with the lowest amount of parameters and FLOPs while being constrained to maintain or improve the PSNR result and the inference time (runtime) of IMDN ([Hui et al, 2017](https://arxiv.org/abs/1909.11856)).

**Track 3: Overall Performance (Runtime, Parameters, FLOPs, Activation, Memory)**, the aim is to obtain a network design / solution with the best overall performance in terms of number of parameters, FLOPS, activations, and inference time and GPU memory on a common GPU (ie. Titan Xp).

## Baseline model (IMDN)

* Number of parameters: 893,936 (0.89M)

    ```python
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    ```

* Average PSNR on validation data: 29.13 dB

* Average inference time (Titan Xp) on validation data: 0.049 second 


Run [test_demo.py](test_demo.py) to test the model

## How to use the code during test phase.

1. `git clone https://github.com/ofsoundof/IMDN`
2. Put your model script under the `models` folder.
3. Put your pretrained model under the `model_zoo` folder.
4. Modify `model_path` in `test_demo.py`. Modify
the imported models.
5. `python test_demo.py`

## How to calculate the number of parameters, FLOPs, and activations

```
    from utils.model_summary import get_model_flops, get_model_activation

    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    logger.info("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    logger.info("{:>16s} : {:<d}".format("#Conv2d", num_conv))

    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    logger.info("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
```
