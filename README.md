# Temporal Feature Enhancement Dilated Convolution Network for Weakly-supervised Temporal Action Localization (WACV 2023)

[Paper](https://openaccess.thecvf.com/content/WACV2023/html/Zhou_Temporal_Feature_Enhancement_Dilated_Convolution_Network_for_Weakly-Supervised_Temporal_Action_WACV_2023_paper.html)

This repository holds the official implementation of TFE-DCN method presented in WACV 2023.

Jianxiong Zhou, and Ying Wu. In the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023.

# Abstract
Weakly-supervised Temporal Action Localization (WTAL) aims to classify and localize action instances in untrimmed videos with only video-level labels. Existing methods typically use snippet-level RGB and optical flow features extracted from pre-trained extractors directly. Because of two limitations: the short temporal span of snippets and the inappropriate initial features, these WTAL methods suffer from the lack of effective use of temporal information and have limited performance. In this paper, we propose the Temporal Feature Enhancement Dilated Convolution Network (TFE-DCN) to address these two limitations. The proposed TFE-DCN has an enlarged receptive field that covers a long temporal span to observe the full dynamics of action instances, which makes it powerful to capture temporal dependencies between snippets. Furthermore, we propose the Modality Enhancement Module that can enhance RGB features with the help of enhanced optical flow features, making the overall features appropriate for the WTAL task. Experiments conducted on THUMOS’14 and ActivityNet v1.3 datasets show that our proposed approach far outperforms state-of-the-art WTAL methods.

![fig2](https://user-images.githubusercontent.com/122836421/212775057-a082fe70-14fb-4767-af15-27fc3516f065.png)

# Results
![Results](https://user-images.githubusercontent.com/122836421/212782229-3bb8ba64-3cd7-4d1d-9f8e-849810b98e6d.png)




# Dependencies
* Create the conda environment as what I used.

``` 
conda create -n TFEDCN python=3.6

conda activate TFEDCN

pip install -r requirements.txt

pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

pip install tqdm==4.41.1

conda install matplotlib
``` 

# THUMOS'14 Dataset
The feature for THUMOS'14 Dataset can be downloaded here. The annotations are included with this package.

# Training
* Run the train scripts:

# References
We referenced the following repos for the code:
* [ActivityNet](https://github.com/activitynet/ActivityNet)
* [MM2021-CO2-Net](https://github.com/harlanhong/MM2021-CO2-Net)

# Citation
Please [★star] this repo and [cite] the following paper if you feel our TFE-DCN useful to your research:
```
@InProceedings{jianxiong_2023_wacv,
  title={Temporal Feature Enhancement Dilated Convolution Network for Weakly-Supervised Temporal Action Localization},
  author={Zhou, Jianxiong and Wu, Ying},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of 
             Computer Vision (WACV)},
  year={2023}
}
```
