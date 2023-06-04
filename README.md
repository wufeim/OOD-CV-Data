## OOD-CV Data

Enhancing the robustness of vision algorithms in real-world scenarios is challenging. One reason is that existing robustness benchmarks are limited, as they either rely on synthetic data or ignore the effects of individual nuisance factors. We present OOD-CV, a benchmark dataset that includes out-of-distribution examples of 10 object categories in terms of pose, shape, texture, context and the weather conditions, and enables benchmarking of models for image classification, object detection, and 3D pose estimation.

[[`Paper`](https://arxiv.org/abs/2111.14341)] [[`Webpage`](http://www.ood-cv.org)] [[`BibTeX`](#citing-ood-cv)]

### Setup Environment

```sh
conda create -n oodcv python=3.9
conda activate oodcv
conda install numpy scipy pillow
pip install wget gdown pyyaml BboxTools opencv-python scikit-image
```

### Preprocess PASCAL3D+ and OOD-CV Data

```sh
python prepare_ood_cv.py \
    --config config.yaml \
    --workers 6
```

### Citing OOD-CV

If you find our work useful, please consider giving a star and citation:

```
@inproceedings{zhao2022ood,
  title={OOD-CV: A Benchmark for Robustness to Out-of-Distribution Shifts of Individual Nuisances in Natural Images},
  author={Zhao, Bingchen and Yu, Shaozuo and Ma, Wufei and Yu, Mingxin and Mei, Shenxiao and Wang, Angtian and He, Ju and Yuille, Alan and Kortylewski, Adam},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part VIII},
  pages={163--180},
  year={2022},
  organization={Springer}
}
```
