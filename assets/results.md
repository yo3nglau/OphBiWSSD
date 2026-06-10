# Results

The performance of OphBiWSSD consistently outperforms five top-tier TAL benchmarks under four video features ([CSN](https://arxiv.org/abs/1904.02811), [SlowFast](https://github.com/facebookresearch/SlowFast), [SwinViViT](https://github.com/SwinTransformer/Video-Swin-Transformer), [VideoMAE](https://github.com/OpenGVLab/VideoMAEv2)): [ActionFormer](https://github.com/happyharrycn/actionformer_release) (ECCV 2022), [TriDet](https://github.com/dingfengshi/tridet) (CVPR 2023), [DyFADet](https://github.com/yangle15/DyFADet-pytorch) (ECCV 2024), [CLTDR-GMG](https://github.com/LiQiang0307/CLTDR-GMG) (AAAI 2025), [ActionMamba](https://github.com/OpenGVLab/video-mamba-suite) (IJCV 2026). mAP (%) is reported at tIoU thresholds α ∈ {0.3, 0.4, 0.5, 0.6, 0.7}. † uses the same detection head and loss as CLTDR-GMG. **Bold**: best result. <ins>Underline</ins>: second best result.

## Temporal Phase Localization (52 classes)

### **CSN**

| Model | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Avg. |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ActionFormer | 31.62 | 28.89 | 24.88 | 20.56 | 14.99 | 24.19 |
| TriDet | 32.58 | 29.92 | 25.82 | 21.50 | 14.13 | 24.79 |
| DyFADet | 34.14 | 30.50 | 27.08 | 21.53 | 16.16 | 25.88 |
| CLTDR-GMG | 35.48 | 32.34 | 28.50 | 23.51 | 18.63 | 27.69 |
| ActionMamba | 36.14 | 33.60 | 29.97 | 22.77 | 16.25 | 27.75 |
| **OphBiWSSD (Ours)** | **38.21** | **35.25** | **31.62** | <ins>24.99</ins> | <ins>19.69</ins> | **29.95** |
| **OphBiWSSD† (Ours)** | <ins>37.87</ins> | <ins>35.14</ins> | <ins>30.28</ins> | **25.39** | **20.61** | <ins>29.86</ins> |

### **SlowFast**

| Model | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Avg. |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ActionFormer | 33.06 | 30.44 | 26.11 | 20.51 | 14.25 | 24.88 |
| TriDet | 34.50 | 32.26 | 28.40 | 23.25 | 17.60 | 27.20 |
| DyFADet | 34.99 | 31.86 | 28.59 | 21.52 | 16.80 | 26.75 |
| CLTDR-GMG | 36.93 | 33.95 | 29.41 | 24.86 | 19.22 | 28.87 |
| ActionMamba | 39.05 | 36.00 | 31.30 | 25.13 | 19.24 | 30.14 |
| **OphBiWSSD (Ours)** | <ins>40.99</ins> | <ins>36.86</ins> | <ins>32.19</ins> | <ins>26.67</ins> | <ins>21.42</ins> | <ins>31.63</ins> |
| **OphBiWSSD† (Ours)** | **41.74** | **39.07** | **34.55** | **29.28** | **23.20** | **33.57** |

### **SwinViViT**

| Model | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Avg. |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ActionFormer | 37.28 | 34.32 | 29.61 | 24.08 | 18.67 | 28.79 |
| TriDet | 38.62 | 35.13 | 30.85 | 25.16 | 20.51 | 30.06 |
| DyFADet | 38.52 | 35.51 | 31.37 | 26.17 | 19.76 | 30.27 |
| CLTDR-GMG | 37.61 | 35.06 | 30.73 | 25.39 | 19.57 | 29.67 |
| ActionMamba | 41.03 | 36.82 | 31.48 | 24.02 | 19.21 | 30.51 |
| **OphBiWSSD (Ours)** | <ins>41.13</ins> | <ins>38.16</ins> | <ins>33.19</ins> | <ins>29.31</ins> | <ins>22.25</ins> | <ins>32.81</ins> |
| **OphBiWSSD† (Ours)** | **42.31** | **39.12** | **35.15** | **29.33** | **23.22** | **33.83** |

### **VideoMAE**

| Model | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Avg. |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ActionFormer | 46.49 | 43.61 | 38.62 | 33.33 | 26.04 | 37.62 |
| TriDet | 46.70 | 44.23 | 41.33 | 35.22 | 28.61 | 39.22 |
| DyFADet | 47.82 | 45.30 | 39.74 | 34.24 | 28.58 | 39.14 |
| CLTDR-GMG | 50.24 | 47.39 | 43.18 | 37.82 | 31.41 | 42.01 |
| ActionMamba | 49.53 | 47.37 | 42.65 | 35.77 | 27.32 | 40.53 |
| **OphBiWSSD (Ours)** | <ins>52.98</ins> | <ins>50.19</ins> | <ins>45.76</ins> | <ins>39.97</ins> | <ins>33.18</ins> | <ins>44.42</ins> |
| **OphBiWSSD† (Ours)** | **54.76** | **51.55** | **46.41** | **41.32** | **34.96** | **45.80** |

## Temporal Operation Localization (107 classes)

### **CSN**

| Model | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Avg. |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ActionFormer | 29.75 | 26.95 | 23.21 | 17.77 | 13.42 | 22.22 |
| TriDet | 33.64 | 30.30 | 27.33 | 22.47 | 16.73 | 26.09 |
| DyFADet | 30.92 | 27.70 | 23.79 | 19.14 | 14.15 | 23.14 |
| CLTDR-GMG | 32.24 | 29.23 | 26.76 | 22.50 | 16.51 | 25.45 |
| ActionMamba | 35.38 | 31.83 | 27.74 | 22.20 | 17.03 | 26.84 |
| **OphBiWSSD (Ours)** | **39.08** | <ins>34.83</ins> | <ins>31.04</ins> | <ins>25.56</ins> | <ins>20.33</ins> | <ins>30.17</ins> |
| **OphBiWSSD† (Ours)** | <ins>38.94</ins> | **35.86** | **31.75** | **26.81** | **21.56** | **30.98** |

### **SlowFast**

| Model | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Avg. |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ActionFormer | 30.75 | 28.66 | 24.91 | 18.93 | 14.26 | 23.50 |
| TriDet | 35.54 | 32.87 | 28.88 | 24.10 | 17.47 | 27.77 |
| DyFADet | 30.37 | 27.35 | 24.42 | 19.60 | 15.28 | 23.41 |
| CLTDR-GMG | 34.13 | 31.71 | 28.01 | 23.35 | 17.99 | 27.04 |
| ActionMamba | 39.21 | 36.06 | 31.29 | 25.36 | 19.85 | 30.36 |
| **OphBiWSSD (Ours)** | <ins>41.66</ins> | **38.55** | <ins>34.01</ins> | <ins>28.60</ins> | <ins>21.85</ins> | <ins>32.93</ins> |
| **OphBiWSSD† (Ours)** | **41.85** | <ins>38.49</ins> | **34.54** | **29.01** | **23.00** | **33.38** |

### **SwinViViT**

| Model | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Avg. |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ActionFormer | 33.87 | 30.11 | 26.14 | 20.26 | 13.97 | 24.87 |
| TriDet | 35.00 | 32.03 | 27.86 | 23.37 | 16.86 | 27.02 |
| DyFADet | 35.04 | 31.65 | 28.51 | 23.19 | 17.84 | 27.25 |
| CLTDR-GMG | 36.47 | 33.80 | 29.84 | 25.69 | 19.37 | 29.04 |
| ActionMamba | 37.05 | 33.84 | 29.02 | 22.69 | 15.94 | 27.71 |
| **OphBiWSSD (Ours)** | **41.59** | **38.69** | **33.71** | **27.76** | **22.42** | **32.83** |
| **OphBiWSSD† (Ours)** | <ins>39.05</ins> | <ins>36.55</ins> | <ins>31.95</ins> | <ins>27.26</ins> | <ins>21.23</ins> | <ins>31.21</ins> |

### **VideoMAE**

| Model | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | Avg. |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| ActionFormer | 45.44 | 42.68 | 37.92 | 31.46 | 24.61 | 36.42 |
| TriDet | 48.45 | 46.02 | 41.36 | 35.82 | 29.69 | 40.27 |
| DyFADet | 46.20 | 43.98 | 40.12 | 33.86 | 27.48 | 38.33 |
| CLTDR-GMG | 48.95 | 45.51 | 41.19 | 36.67 | 29.68 | 40.40 |
| ActionMamba | 49.18 | 46.34 | 42.05 | 34.71 | 26.87 | 39.83 |
| **OphBiWSSD (Ours)** | <ins>52.59</ins> | <ins>49.22</ins> | <ins>45.13</ins> | <ins>37.83</ins> | <ins>30.62</ins> | <ins>43.08</ins> |
| **OphBiWSSD† (Ours)** | **52.60** | **49.83** | **46.35** | **39.80** | **33.46** | **44.41** |
