C:\Users\KETI\AppData\Local\Programs\Python\Python37\python.exe C:/Users/KETI/hannie/main.py
Hi, PyCharm
Files already downloaded and verified
Files already downloaded and verified
Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
plane
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

Train:
[12500/12500] | Loss: 0.59405

Validation
[2500/2500] | Loss: 0.68239

Epoch  4

Train:
[12500/12500] | Loss: 0.48832

Validation
[2500/2500] | Loss: 0.78060

Epoch  5

Train:
[12500/12500] | Loss: 0.40461

Validation
[2500/2500] | Loss: 0.76174

Epoch  6

Train:
[12500/12500] | Loss: 0.34993

Validation
[2500/2500] | Loss: 0.69767

Epoch  7

Train:
[12500/12500] | Loss: 0.30967

Validation
[2500/2500] | Loss: 0.86094

Epoch  8

Train:
[12500/12500] | Loss: 0.28769

Validation
[2500/2500] | Loss: 0.80631

Epoch  9

Train:
[12500/12500] | Loss: 0.27127

Validation
[2500/2500] | Loss: 0.83604

Epoch  10

Train:
[12500/12500] | Loss: 0.27028

Validation
[2500/2500] | Loss: 1.07764

Epoch  11

Train:
[12500/12500] | Loss: 0.25830

Validation
[2500/2500] | Loss: 1.30018

Epoch  12

Train:
[12500/12500] | Loss: 0.25812

Validation
[2500/2500] | Loss: 1.19477

Epoch  13

Train:
[12500/12500] | Loss: 0.26153

Validation
[2500/2500] | Loss: 0.89517

Epoch  14

Train:
[12500/12500] | Loss: 0.26290

Validation
[2500/2500] | Loss: 0.85531

Epoch  15

Train:
[12500/12500] | Loss: 0.25636

Validation
[2500/2500] | Loss: 0.87353

Epoch  16

Train:
[12500/12500] | Loss: 0.26459

Validation
[2500/2500] | Loss: 1.08304

Epoch  17

Train:
[12500/12500] | Loss: 0.26941

Validation
[2500/2500] | Loss: 1.27546

Epoch  18

Train:
[12500/12500] | Loss: 0.28042

Validation
[2500/2500] | Loss: 0.93558

Epoch  19

Train:
[12500/12500] | Loss: 0.28703

Validation
[2500/2500] | Loss: 1.04861

Epoch  20

Train:
[12500/12500] | Loss: 0.33408

Validation
[2500/2500] | Loss: 1.05905

Epoch  21

Train:
[12500/12500] | Loss: 0.34988

Validation
[2500/2500] | Loss: 1.27398

Epoch  22

Train:
[12500/12500] | Loss: 0.30363

Validation
[2500/2500] | Loss: 1.46341

Epoch  23

Train:
[12500/12500] | Loss: 0.47419

Validation
[2500/2500] | Loss: 1.37669

Epoch  24

Train:
[12500/12500] | Loss: 0.34994

Validation
[2500/2500] | Loss: 0.78363

Epoch  25

Train:
[12500/12500] | Loss: 0.34847

Validation
[2500/2500] | Loss: 1.27663

Epoch  26

Train:
[12500/12500] | Loss: 0.48376

Validation
[2500/2500] | Loss: 1.78389

Epoch  27

Train:
[12500/12500] | Loss: 0.41131

Validation
[2500/2500] | Loss: 3.90899

Epoch  28

Train:
[12500/12500] | Loss: 0.37094

Validation
[2500/2500] | Loss: 1.06383

Epoch  29

Train:
[12500/12500] | Loss: 0.80590

Validation
[2500/2500] | Loss: 0.91278

Epoch  30

Train:
[12500/12500] | Loss: 0.39265

Validation
[2500/2500] | Loss: 1.63820

Epoch  31

Train:
[12500/12500] | Loss: 0.40834

Validation
[2500/2500] | Loss: 3.81228

Epoch  32

Train:
[12500/12500] | Loss: 1.31910

Validation
[2500/2500] | Loss: 1.50599

Epoch  33

Train:
[12500/12500] | Loss: 0.50278

Validation
[2500/2500] | Loss: 0.76590

Epoch  34

Train:
[12500/12500] | Loss: 0.42849

Validation
[2500/2500] | Loss: 0.84705

Epoch  35

Train:
[12500/12500] | Loss: 0.59535

Validation
[2500/2500] | Loss: 0.83152

Epoch  36

Train:
[12500/12500] | Loss: 0.60598

Validation
[2500/2500] | Loss: 0.83340

Epoch  37

Train:
[12500/12500] | Loss: 0.50645

Validation
[2500/2500] | Loss: 3.92895

Epoch  38

Train:
[12500/12500] | Loss: 104.44523

Validation
[2500/2500] | Loss: 1.09040

Epoch  39

Train:
[12500/12500] | Loss: 11.20984

Validation
[2500/2500] | Loss: 1.70822

Epoch  40

Train:
[12500/12500] | Loss: 0.67160

Validation
[2500/2500] | Loss: 4.68893

Epoch  41

Train:
[12500/12500] | Loss: 1.14047

Validation
[2500/2500] | Loss: 0.77187

Epoch  42

Train:
[12500/12500] | Loss: 0.78836

Validation
[2500/2500] | Loss: 0.86322

Epoch  43

Train:
[12500/12500] | Loss: 20.20905

Validation
[2500/2500] | Loss: 0.86668

Epoch  44

Train:
[12500/12500] | Loss: 21.10520

Validation
[2500/2500] | Loss: 0.81077

Epoch  45

Train:
[12500/12500] | Loss: 0.73384

Validation
[2500/2500] | Loss: 0.84592

Epoch  46

Train:
[12500/12500] | Loss: 17.66244

Validation
[2500/2500] | Loss: 0.93870

Epoch  47

Train:
[12500/12500] | Loss: 0.75603

Validation
[2500/2500] | Loss: 0.76304

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
