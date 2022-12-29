## ResNet
```ResNet Model(Paper)``` (원문: [ResNet](https://arxiv.org/pdf/1512.03385.pdf))  



## Abstract
- 레이어를 residual function으로 재구성한다. 
- ILSVRC 2015 classification 과제에서 1위를 차지했다. 

## Introduction

- 최근에 깊은 네트워크(DNN) 학습이 인기가 많다. 하지만 네트워크의 깊이가 깊어질수록 gradient descent, overfitting 등의 문제가 발생했다. 이 문제들은 대부분 정규화 계층해서 해결되었다. 
그렇지만 이번에는 더 깊은 네트워크가 수렴할수록 성능 저하(degradation) 문제가 발생했다.  
본 논문에서는 ```Residual Lerning Framework``` 를 도입하여 성능 저하 문제를 해결하였다.  

- residual mapping을 통해 하나 이상의 레이어를 스킵하는 shortcut connection을 구현한다. 식별자 shortcut connection은 추가 매개변수가 필요하지 않고 계산이 복잡하지 않다.


## Architecture

두가지 모델
- 일반 네트워크 
  - 3*3 필터 사용
  - 피처맵의 크기가 절반으로 줄면 필터 수 2배로 늘려서 시간 복잡성 보존
  - 레이어는 34개, 마지막은 1000개의 클래스로 softmax
  
- 잔차 네트워크
  - 일반 네트워크에서 shortcut 연결 추가
  - 식별자 shortcut 연결은 입력과 출력의 차원이 같아야 함.
  

![resnet](https://user-images.githubusercontent.com/50253860/204150928-a0b6f560-3448-47da-b89e-e48b2fac4472.png)


## Discussion

- 극단적으로 깊은 Residual Net은 최적화하기 쉽다.  
단순히 층을 쌓은 일반 네트워크는 깊이가 증가하면 훈련 오류가 커진다.

- 심층 Residual 네트워크는 매우 깊어도 쉽게 정확성을 얻을 수 있다.  
이전 네트워크보다 훨씬 나은 결과를 얻을 수 있다. 

