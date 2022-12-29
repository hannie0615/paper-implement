## AlexNet(2012) 리뷰 및 구현

- ```AlexNet Model(Paper)``` (원문: [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf))  
- ```ImageNet classification```
- ```pytorch```로 구현  


## Abstract

2010년 ILSVRC 대회에서 SOTA(state-of-the-art 달성)  
CNN이 큰 주목을 받게 된 계기가 됨  
오버피팅을 줄이기 위해 드롭아웃 등 기법 적용

- 병렬 GPUs사용 (현재는 1개의 GPU로 사용하도록 수정)
- ReLu, MaxPoolinng, Dropout 사용
- 빠른 학습
- Data Augmentation



## Architecture

 구조는 5개의 합성곱 층(convolutional layer)과 3개의 완전 연결층(fully connected layer)으로 이루어짐.

![c1](https://user-images.githubusercontent.com/50253860/204150748-2e5828ea-2393-465d-96ac-943eae5a5bb4.png)

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 96, 55, 55]          34,944
              ReLU-2           [-1, 96, 55, 55]               0
 LocalResponseNorm-3           [-1, 96, 55, 55]               0
         MaxPool2d-4           [-1, 96, 27, 27]               0
            Conv2d-5          [-1, 256, 27, 27]         614,656
              ReLU-6          [-1, 256, 27, 27]               0
 LocalResponseNorm-7          [-1, 256, 27, 27]               0
         MaxPool2d-8          [-1, 256, 13, 13]               0
            Conv2d-9          [-1, 384, 13, 13]         885,120
             ReLU-10          [-1, 384, 13, 13]               0
           Conv2d-11          [-1, 384, 13, 13]       1,327,488
             ReLU-12          [-1, 384, 13, 13]               0
           Conv2d-13          [-1, 256, 13, 13]         884,992
             ReLU-14          [-1, 256, 13, 13]               0
        MaxPool2d-15            [-1, 256, 6, 6]               0
AdaptiveAvgPool2d-16            [-1, 256, 6, 6]               0
           Linear-17                 [-1, 4096]      37,752,832
             ReLU-18                 [-1, 4096]               0
          Dropout-19                 [-1, 4096]               0
           Linear-20                 [-1, 4096]      16,781,312
             ReLU-21                 [-1, 4096]               0
          Dropout-22                 [-1, 4096]               0
           Linear-23                   [-1, 10]          40,970
================================================================
Total params: 58,322,314
Trainable params: 58,322,314
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.59
Forward/backward pass size (MB): 14.75
Params size (MB): 222.48
Estimated Total Size (MB): 237.82
----------------------------------------------------------------



## 실행 결과

```C:\Users\KETI\AppData\Local\Programs\Python\Python37\python.exe C:/Users/KETI/hannie/main.py   

Files already downloaded and verified   
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).   


Epoch  0

Train:
[12500/12500] | Loss: 1.42579

Validation
[2500/2500] | Loss: 1.07362

Epoch  1

Train:
[12500/12500] | Loss: 0.93875

Validation
[2500/2500] | Loss: 0.84799

Epoch  2

Train:
[12500/12500] | Loss: 0.73101

Validation
[2500/2500] | Loss: 0.74433

Epoch  3

.
.
((생략))
.
.

Epoch  48

Train:
[12500/12500] | Loss: 0.67797

Validation
[2500/2500] | Loss: 1.72786

Epoch  49

Train:
[12500/12500] | Loss: 0.69543

Validation
[2500/2500] | Loss: 0.83729

Process finished with exit code 0
```

(보유한 gpu 한계로는 epochs=50 까지만 학습 가능)


## Discussion 

결과적으로 크고 깊은 CNN이 지도학습을 통해 매우 어려운 데이터셋에서 기록적인 결과를 달성할 수 있다는 것을 보여준다.  
네트워크의 깊이는 결과를 얻기 위해 매우 중요하다.  



