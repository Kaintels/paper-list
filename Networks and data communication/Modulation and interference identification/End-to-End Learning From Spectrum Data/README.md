# Title

**End-to-End Learning From Spectrum Data: A Deep Learning Approach for Wireless Signal Identification in Spectrum Monitoring Applications** (대학원 2학기 유비쿼터스네트워크 과목 - 논문 구현) [[code]](https://github.com/Kaintels/paper-review/blob/master/Networks%20and%20data%20communication/Modulation%20and%20interference%20identification/Merima%20Kulin%20et%20al..ipynb)



## Contents

* Ⅰ. [Introduction](#Introduction)
* Ⅱ. [Characteristic use cases for End-to-End learning from spectrum data](#Characteristic-use-cases-for-End-to-End-learning-from-spectrum-data)
* Ⅲ. [Data-Driven end-to-end learning for wireless signal classification](#Data-Driven-end-to-end-learning-for-wireless-signal-classification)
* Ⅳ. [Evaluation Setup](#Evaluation-Setup)
* Ⅴ. [Conclusion](#Conclusion)
* [발표자료](#발표자료)



## Introduction
* 네트워크는 현재 발전하고 있으며 스펙트럼 수요가 증가함에 따라 무선 장치의 수와 다양성이 증가하고 있음.

* 스펙트럼 수요가 증가함에 따라 관리와 같은 모니터링이 필요해짐.

* 본 논문는 심층 신경망 기반의 스펙트럼 모니터링 애플리케이션에서 무선 신호 식별 방법을 위한 스펙트럼 데이터로부터 End-to-End 딥 러닝을 연구함.



<A. Scope and contributions>
본 논문은 두 가지 사례 연구를 수행

1. modulation recognition
2. wireless technology interference detection

변조 인식을 위해 다음과 같은 변조 기술이 고려.

* BPSK, QPSK, 8-PSK, 16-QAM and 64-QAM, CPFSK, GFSK, 4-PAM

* IEEE 802.11b/g, IEEE 802.15.4 and IEEE 802.15.1.

<B. Related work> 

1. Traditional signal identification
   신호 식별과 관련된 무선 통신에 대한 이전의 연구 노력은 기존의 머신 러닝 기술이 주로 사용됨 (e.g. support vector machines (SVM), decision trees, k-nearest neighbors (k-NN), neural networks (NNs), etc.)

2. Deep learning for signal classification

   CNN (Convolutional Neural Network)은 자동 변조 인식 영역에서 기존의 머신 러닝 분류기보다 성능이 뛰어남을 입증.

3. Deep learning for wireless networks
    본 논문은 스펙트럼 모니터링을위한 End-to-End 딥 러닝을 제공.

## Characteristic use cases for End-to-End learning from spectrum data
<A. Detecting spectral opportunities & spectrum sharing>

1. Cognitive Radio (CR)

   CR network (CRN) 은 무선 환경을 인식하는 지능형 무선 통신 시스템.
   
2. Cognitive IoT

   이기종 무선 기술과 그에 따른 표준은 라이센스가 없는 주파수 대역에서 작동하여 사용 가능한 스펙트럼에 막대한 pressure을 가함.

<B. Spectrum Management policy and regulation>

* 무선 간섭과 관련된 문제를 해결하기 위해 스펙트럼 관리자는 전통적으로 엔지니어링 분석과 스펙트럼 측정에서 얻은 데이터의 조합을 사용.

* 그러나 오늘날 다양한 서비스와 무선 기술이 동일한 주파수 대역을 공유하는 시대에는 무단 송신기를 식별하기가 매우 어려울 수 있음.

* 스펙트럼 데이터를 자동으로 마이닝하고 간섭 소스를 식별 할 수있는보다 지능적인 알고리즘이 필요.

<C. Deep Learning from spectrum data>

1. Data acquisition

   스펙트럼 데이터를 얻기 위해 무전기는 먼저 다양한 스펙트럼 대역에서 원시 데이터를 수집하여 환경을 감지.

   raw data는 수신 된 무선 신호의 complex envelope를 나타내는 데이터 벡터 rk에 적층 된 n 개의 샘플로 구성.
   
2. Data pre-processing

   데이터 벡터 rk로 구성된 raw sample은 주파수, 진폭, 위상 및 스펙트럼과 같이 분석하는 신호 처리 (SP) 도구를 위한 입력으로 변환.

3. Classification (spectrum learning)

   분류는 무선 신호의 존재를 감지하여 환경 무선 상황을 가능하게 함.

4. Decision (spectrum decision)

   ML 모델에 의해 계산 된 예측은 CR 애플리케이션에서 결정 모듈에 대한 입력으로 사용. 

   결정은 다른 사용자에게 간섭을 일으키지 않고 데이터 속도를 최대화하는 최상의 전송 전략 (예 : 주파수 대역 또는 전송 전력)과 관련 될 수 있음.

   

## Data-Driven end-to-end learning for wireless signal classification

<A. Wireless signal model>

1. Transmitter

2. Wireless channel

3. Receiver


<B. Wireless signal model>

무선 신호 식별을위한 기계 학습 모델을 도출하기 위해서는 적절한 train 데이터가 수집되어야 함.

<C. Wireless signal representation>

본 논문은 세 가지 간단한 데이터 표현을 고려, IQ vector, A/Φ Vector, FFT vector

## Evaluation Setup

<A. Datasets description>

1. Radio modulation recognition

   Dataset : RadioML 2016.10a Modulation dataset (https://www.deepsig.io/datasets)

   데이터 벡터 x는 128 개의 샘플 배치에서 수집.

   One-hot encoding은 11 개의 변조에 해당하는 11 개의 클래스 레이블의 discrete 세트를 생성하는 데 사용.

   전체적으로, I 및 Q 샘플로 구성된 220,000 개의 데이터 벡터 x ∈ R2 × 128이 사용.

2. Wireless interference identification in ISM bands

   Dataset : Wireless interference dataset from Signal Generation Software (https://www.deepsig.io/datasets)

   2.4GHz 주파수 대역에서 작동하는 IEEE 802.11b / g (WiFi), IEEE 802.15.4 (Zigbee) 및 IEEE 802.15.1 (Bluetooth) 표준

   데이터 세트는 할당 된 주파수 채널과 해당 무선 기술에 따라 레이블이 지정되어 15 가지 클래스로 구성. 

   총 225,225 개의 스냅 샷이 수집.

<B. CNN Network structure>

* 1D CNN 으로 진행하였고 filter size  제외하고 본 논문과 같은 파라미터 구현하도록 노력함.

<C. Implementation details>

* 두 예제 세트 모두 신호 대 잡음비 (SNR)가 -20dB ~ + 20dB로 균일하게 분배되고 특정 서브 세트에서 성능을 평가할 수 있도록 태그가 지정.

* CNN은 70 에포크에 대해 훈련됨

## Conclusion

* 본 논문은 다양한 무선 신호 식별 작업을 실현하기위한 딥 러닝 기반의 통합 된 접근 방식 인 스펙트럼 데이터로부터 End-to-End 딥 러닝에 대한 포괄적이고 체계적인 소개를 제공.

* Convolutional neural networks (CNN)는 자체 처리 능력을 갖춘 여러 계층으로 구성되어 있기 때문에 이러한 기능에 적합.
* 제시된 방법론은 두 개의 활성 무선 신호 식별 연구 문제에서 검증
* 또한, 무선 통신 도메인에 대해 제시된 결과는 표시된 분류에 대한 구별되는 특성을 나타내는 올바른 표현을 결정하는 데 중요.
* 구체적으로, medium-high SNR에 대한 변조-인식 사례에 대한 연구에서는 CNN 모델에 대해 훈련 된 A/Φ Vector 표현은 다른 2 개의 모델에서 2 % 및 10 %의 성능 향상이 있었음.

## 발표자료

![image](https://user-images.githubusercontent.com/38157496/75612184-969dc280-5b64-11ea-95f9-25b9c868eb5a.png)
![image](https://user-images.githubusercontent.com/38157496/75612358-02ccf600-5b66-11ea-9a7b-87ade7cf90c5.png)
![image](https://user-images.githubusercontent.com/38157496/75612197-addcb000-5b64-11ea-8add-0e14985e85c0.png)
![image](https://user-images.githubusercontent.com/38157496/75612201-b9c87200-5b64-11ea-97f4-93048b14a112.png)
![image](https://user-images.githubusercontent.com/38157496/75612202-c2b94380-5b64-11ea-98c2-048ce1f1e4e8.png)
![image](https://user-images.githubusercontent.com/38157496/75612208-d6fd4080-5b64-11ea-9b96-e0b242675e61.png)
![image](https://user-images.githubusercontent.com/38157496/75612209-dcf32180-5b64-11ea-8cef-c3804dc06d51.png)
![image](https://user-images.githubusercontent.com/38157496/75612231-08760c00-5b65-11ea-95d1-0e4ff8dc8119.png)
![image](https://user-images.githubusercontent.com/38157496/75612248-2b082500-5b65-11ea-9145-bfe52882ecc6.png)
![image](https://user-images.githubusercontent.com/38157496/75612257-39564100-5b65-11ea-828b-de12feae7172.png)
![image](https://user-images.githubusercontent.com/38157496/75612264-45420300-5b65-11ea-82c7-7d7950458c3c.png)
![image](https://user-images.githubusercontent.com/38157496/75612299-85a18100-5b65-11ea-94db-925c4587a757.png)
![image](https://user-images.githubusercontent.com/38157496/75612323-b1bd0200-5b65-11ea-9cd9-e452aa67e620.png)
![image](https://user-images.githubusercontent.com/38157496/75612331-c4cfd200-5b65-11ea-8645-27f3620bdc08.png)
![image](https://user-images.githubusercontent.com/38157496/75612333-cbf6e000-5b65-11ea-91f9-3ab7ce928655.png)
![image](https://user-images.githubusercontent.com/38157496/75612336-d4e7b180-5b65-11ea-8ee7-365313133b29.png)



## Reference

https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb