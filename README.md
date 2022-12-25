# SimSwap++: Towards Faster and High-Quality Identity Swapping
### [Xuanhong Chen](https://scholar.google.com/citations?user=UuCqlfEAAAAJ&hl=en), [Bingbing Ni](https://scholar.google.com/citations?user=V9W87PYAAAAJ&hl=en) $\dagger$, Yutian Liu, Naiyuan Liu, Linzhi Zeng, Hang Wang
$\dagger$ Corresponding author

## Project page of SimSwap++


[![logo](/docs/img/logo.png)](https://github.com/neuralchen/SimSwapPlus)

# VGGFace2-HQ Dataset
VGGFace2-HQ contains more than $1.36M$ $512 \times 512$ aligned
face images and up to $9, 630$ distinct identities. In addition, this dataset
consists of two parts:
- (1) a natural image sub-collection, which collects up to $200, 000$ images covering $1, 000$ different identities;
- (2) a synthetic image sub-collection, containing $8, 630$ cleaned and re-annotated identities (i.e., clean up the images with mismatching identities and low-quality faces in the cropped [VGGFace2](https://github.com/ox-vgg/vgg_face2)).

## Download the dataset:
<!-- ***Limited by the capacity of the cloud disk, we divided the dataset into two parts*** -->

### Via Google Drive:

[[Google Drive]  VGGFace2-HQ](https://drive.google.com/drive/folders/1ZHy7jrd6cGb2lUa4qYugXe41G_Ef9Ibw?usp=sharing)

<!-- [[Google Drive]  VGGFace2-HQ Part2 (89GB)](https://drive.google.com/drive/folders/1ZHy7jrd6cGb2lUa4qYugXe41G_Ef9Ibw?usp=sharing) -->

***We are especially grateful to [Kairui Feng](https://scholar.google.com.hk/citations?user=4N5hE8YAAAAJ&hl=zh-CN) PhD student from Princeton University.***

### Via Baidu Drive:

[[Baidu Drive] VGGFace2-HQ](https://pan.baidu.com/s/1LwPFhgbdBj5AeoPTXgoqDw) Password: ```sjtu```

<!-- [[Baidu Drive] VGGFace2-HQ Part2 (89GB)](https://pan.baidu.com/s/1LwPFhgbdBj5AeoPTXgoqDw) Password: ```sjtu``` -->

## Samples from VGGFace2-HQ

### Natural Part:

[![naturalpart.png](/docs/img/naturalpart.png)](https://github.com/neuralchen/SimSwapPlus)

### Synthetic Part:

[![naturalpart.png](/docs/img/syntheticpart.png)](https://github.com/neuralchen/SimSwapPlus)

# Methodology of SimSwap++
## Additional Results:

Will be avaliable 2022/12/25


## Video in-the-wild Results (under construction):

### Group1 [SimSwap++ (S)]:
[Source ID: Kelly Clarkson Target ID: Taylor Swift (1080p on YouTube)](https://www.youtube.com/watch?v=U9WFnMHs6Nw)

<img src="./docs/video/1.webp"/>

[Source ID: Geoffrey Hinton Target ID: Taylor Swift (1080p on YouTube)](https://youtu.be/QLrneMYKki0)

[![Geoffrey Hinton](/docs/video/2.webp)](https://youtu.be/QLrneMYKki0)

[Source ID: Gal Gadot Target ID: Taylor Swift (1080p on YouTube)](https://youtu.be/I00NuaICEQE)

[![Gal Gadot](/docs/video/3.webp)](https://youtu.be/I00NuaICEQE)

[Source ID: Leonardo DiCaprio Target ID: Taylor Swift (1080p on YouTube)](https://www.youtube.com/watch?v=75W6j-0ux4k)

<img src="./docs/video/4.webp"/>

[Source ID: Elon Musk Target ID: Taylor Swift (1080p on YouTube)](https://youtu.be/YRhql8WGSIE)

[![Elon Musk](/docs/video/5.webp)](https://youtu.be/YRhql8WGSIE)


[Source ID: Robert Downey Target ID: Taylor Swift (1080p on YouTube)](https://www.youtube.com/watch?v=qbmtj4z0RmE)

[![Robert Downey](/docs/video/6.webp)](https://www.youtube.com/watch?v=qbmtj4z0RmE)

[Source ID: Aamir Khan Target ID: Taylor Swift (1080p on YouTube)](https://youtu.be/BY-sMBTbtBU)

[![Aamir Khan](/docs/video/7.webp)](https://youtu.be/BY-sMBTbtBU)

### Group2:



Will be avaliable 2022/12/25

# Acknowledgements

<!--ts-->
* [GFPGAN](https://github.com/TencentARC/GFPGAN)
* [Insightface](https://github.com/deepinsight/insightface)
* [VGGFace2 Dataset for Face Recognition](https://github.com/ox-vgg/vgg_face2)
<!--te-->