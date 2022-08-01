# Video Joint Modelling Based on Hierarchical Transformer for Co-summarization (VJMHT)
 [Paper](https://ieeexplore.ieee.org/abstract/document/9808180/)

Haopeng Li, Qiuhong Ke, Mingming Gong, Rui Zhang

IEEE Transactions on Pattern Analysis and Machine Intelligence




## Introduction
We propose **V**ideo **J**oint **M**odelling based on **H**ierarchical **T**ransformer (**VJMHT**) for co-summarization, which takes into consideration the semantic dependencies across videos. 

VJMHT consists of two layers of Transformer: the first layer extracts semantic representation from individual shots of similar videos, while the second layer performs shot-level video joint modelling to aggregate cross-video semantic information. By this means, complete cross-video high-level patterns are explicitly modelled and learned for the summarization of individual videos.


Moreover, Transformer-based video representation reconstruction is introduced to maximize the high-level similarity between the summary and the original video.


## Requirements and Dependencies

- Python=3.8.5
- PyTorch=1.9, ortools=8.1.8487

## Data Preparation

Download the [datasets](https://unimelbcloud-my.sharepoint.com/:f:/g/personal/haopengl1_student_unimelb_edu_au/El305SWgUh5GtyFeq3sMpsEBijWY9CkQ3hOhRElRMm2dMg?e=155YfL) to ``datasets/``.

## Testing Model

Download [our models](https://unimelbcloud-my.sharepoint.com/:f:/g/personal/haopengl1_student_unimelb_edu_au/Eu1HIbsLHYZBuxVAQLk8cnYBr2pTL7KVj0LURYWNY-RwZw?e=gXgVzw) to ``results/``.

Run the following command to test our models.

```
$ python main.py -c configs/cfg_file.py --eval
```

where ``cfg_file.py`` is the configeration file that can be found in ``configs/``. The results are saved in ``results/CFG_FILE/``.

Example:

```
$ python main.py -c configs/tvsum_can.py --eval
```

## Training Model
Run the following command to train the model.

```
$ python main.py -c configs/CFG_file
```

Example:

```
$ python main.py -c configs/tvsum_can.py
```

### Contact
[Haopeng Li](mailto:haopeng.li@student.unimelb.edu.au)



### License and Citation

The use of this code is RESTRICTED to **non-commercial research and educational purposes**.

```
@article{li2022video,
  title={Video Joint Modelling Based on Hierarchical Transformer for Co-summarization},
  author={Li, Haopeng and Ke, Qiuhong and Gong, Mingming and Zhang, Rui},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```


