# Multi-modal fusion for sensing-aided beam tracking in mmWave communications [[Paper]](https://www.sciencedirect.com/science/article/pii/S1874490724002325#sec3)

Millimeter wave (mmWave) communication has attracted extensive attention and research due to its wide
bandwidth and abundant spectrum resources. Existing studies demonstrate the potential of sensingaided beam tracking. However, most studies are single-modal data assistance without considering
multi-modal calibration or the impact of inference latency of different sub-modules. Thus, in this study,
we design a decision-level multi-modal (mmWave received signal power vector, RGB image and GPS data)
fusion for sensing-aided beam tracking (DMBT) method. The proposed DMBT method includes three designed
mechanisms, namely normal prediction process (NPP), beam misalignment alert (BMA) and beam tracking correction (BTC). The normal prediction process conducts partial beam training instead of exhaustive beam training, which largely reduces large beam training overhead. It also comprehensively selects prediction results from multi-modal data to enhance the DMBT method robustness to noise. The beam misalignment alert based on RGB image and GPS
data detects whether there exists beam misalignment and also predict the optimal beam. The beam tracking correction is designed to capture the optimal beam if misalignment happens by reusing certain blocks in normal prediction process and possibly outdated prediction results. Finally, we evaluate the proposed DMBT method in the vehicle-to-infrastructure scenario based on a real-world dataset. Codes related to the article were already updated on [[Codes Repository]](https://github.com/Trans-cending/DMBT_V2I_mmwave).

## Preparation

These instructions will get you a copy of the project up and running on your local machine for development
and testing purposes.

The packages required are

* joblib==1.2.0
* numpy==1.23.5
* pandas==1.5.2
* random_fourier_features_pytorch==1.0.1
* scikit_learn==1.0.2
* torch==1.12.0+cu113
* torchvision==0.13.0+cu113

you can install it by

```
pip install -r requirements.txt
```

## Description

The project tests two different beam numbers, namely $M=32$ and $M=64$. For easier testing, the project is divided into two code packages and results can be saved separately, namely 'Model_beam_32' and 'Model_beam_64'.

$N_\mathrm{r}\in\{1,2,\dotsm,6\}$ includes nearly every possible condition, which means that ORS, COS, and CHS have different combination types in the 'Union' process in BTC. Note that $N_\mathrm{r}=0$ has no actual meaning, just representing the Baseline 1 results.

To enhance the speed of testing process, object detection part of RGUS is not uploaded since it can possibly cost considerable time. Instead, the locations of detected targets were saved in files in advance provided in the downloaded dataset.

## Usage

If you want to test $M=32$, please run

```
cd path\to\your\Model_beam_32
python test_main.py
```

If you want to test $M=64$, please run

```
cd path\to\your\Model_beam_64
python test_main.py
```

The results will be recorded in 'Error_correction_comparison_mm' and 'Error_correction_comparison_loc'.

## License

The code is Mozilla-licensed. The license applies to the pre-trained models and datasets as well.

## Citation

😎If you find the repository is helpful to your project or modify your algorithm based on this code, please cite as follows

```bibtex
@article{BIAN2024102514,
title = {Multi-modal fusion for sensing-aided beam tracking in mmWave communications},
journal = {Physical Communication},
volume = {67},
pages = {102514},
year = {2024},
issn = {1874-4907},
doi = {https://doi.org/10.1016/j.phycom.2024.102514},
url = {https://www.sciencedirect.com/science/article/pii/S1874490724002325},
author = {Yijie Bian and Jie Yang and Lingyun Dai and Xi Lin and Xinyao Cheng and Hang Que and Le Liang and Shi Jin},
keywords = {mmWave communications, Deep learning, Beam training and tracking, Multi-modal data, Decision-level fusion},
abstract = {Millimeter wave (mmWave) communication has attracted extensive attention and research due to its wide bandwidth and abundant spectrum resources. Effective and fast beam tracking is a critical challenge for the practical deployment of mmWave communications. Existing studies demonstrate the potential of sensing-aided beam tracking. However, most studies are focus on single-modal data assistance without considering multi-modal calibration or the impact of inference latency of different sub-modules. Thus, in this study, we design a decision-level multi-modal (mmWave received signal power vector, RGB image and GPS data) fusion for sensing-aided beam tracking (DMBT) method. The proposed DMBT method includes three designed mechanisms, namely normal prediction process, beam misalignment alert and beam tracking correction. The normal prediction process conducts partial beam training instead of exhaustive beam training, which largely reduces large beam training overhead. It also comprehensively selects prediction results from multi-modal data to enhance the DMBT method robustness to noise. The beam misalignment alert based on RGB image and GPS data detects whether there exists beam misalignment and also predict the optimal beam. The beam tracking correction is designed to capture the optimal beam if misalignment happens by reusing certain blocks in normal prediction process and possibly outdated prediction results. Finally, we evaluate the proposed DMBT method in the vehicle-to-infrastructure scenario based on a real-world dataset. The results show that the method is capable of self-correction and mitigating the negative effect of the relative inference latency. Moreover, 75%–93% beam training overhead can be saved to maintain reliable communication even when faced with considerable noise in measurement data.}
}
```

## Acknowledgement

Thanks to the Deepsense 6G public dataset [[Public Dataset]](https://www.deepsense6g.net/).
