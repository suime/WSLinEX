# WSLinEX
"Building a Data Analytics Environment in the Company"

## WSL 설치
참고링크

```cardlink
url: https://readmedium.com/en/https:/medium.com/dawn-cau/wsl2-%EB%94%A5%EB%9F%AC%EB%8B%9D-%ED%99%98%EA%B2%BD-%EA%B5%AC%EC%B6%95%ED%95%98%EA%B8%B0-95d7b95d1f4b
title: "WSL2 딥러닝 환경 구축하기 (CUDA, CuDNN, Anaconda)"
description: "WSL 설치"
host: readmedium.com
favicon: https://readmedium.com/favicon.ico
```

```cardlink
url: https://velog.io/@cjkangme/WSL2%EB%A1%9C-CUDA-%ED%99%98%EA%B2%BD-%EC%84%A4%EC%A0%95%ED%95%98%EA%B8%B0-CUDAcuDNN-%EC%84%A4%EC%B9%98%EA%B9%8C%EC%A7%80
title: "WSL2로 CUDA 환경 설정하기 (CUDA+cuDNN 설치까지)"
description: "설치 참조 링크"
host: velog.io
favicon: https://static.velog.io/favicons/favicon-32x32.png
image: https://images.velog.io/velog.png
```

WSL이 설치가 되면 윈도우에서 우분투 터미널로 접속한다.

```sh title:"빌드 에센셜 패키지 설치"
sudo apt update
sudo apt install build-essential
```

build-essential 패키지는 개발에 필요한 기본 라이브러리와 헤더파일 등을 갖고 있는 패키지이다. 각종 실행파일 실행에 필요한 `cmake`, `gcc`, `g++` 등의 라이브러리를 제공한다.

## 분석 환경 설정
### 🚧 각종 SSL 우회 설정하기
- [!] 중요 회사컴에서는 SSL 인증이 안되서 우회를 해야된다.

```sh 
# wget 우회 
echo 'alias wget="wget --no-check-certificate"' >> ~/.bashrc

# pip 우회 
pip config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org"

# conda 우회
conda config --set ssl_verify false

# git 우회 
git config --global http.sslVerify false

# bash 새로 불러오기 
source ~/.bashrc
```

위에거 안하면 `wget` 뒤에 계속 저 옵션을 붙여야 하므로 해두면 좋다.

### nvidia 드라이버
- 기본적으로 WSL에서는 윈도우의 그래픽카드 드라이버를 따라가기 때문에 따로 리눅스 배포판에 맞는 그픽 카드를 **추가적으로 설치할 필요가 없다!**
- 오히려 설치하면 오류가 발생할 수 있다.

### 리눅스 기준 `cuda` 및 `cuDNN` 버전

| **YOLO 버전**                                                      | **PyTorch**                                             | **Tensorflow**                                                      | **Python 버전**  | **cuDNN 버전** | **CUDA 버전** |
| ---------------------------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------- | -------------- | ------------ | ----------- |
| [YOLOv4](https://github.com/AlexeyAB/darknet?tab=readme-ov-file) | [>=1.5.1](https://github.com/WongKinYiu/PyTorch_YOLOv4) | [==2.3.0rc0](https://github.com/hunglc007/tensorflow-yolov4-tflite) | >=3.8          | >=7.6        | >=10.1      |
| [YOLOv5](https://github.com/ultralytics/yolov5)                  | >=1.8.0                                                 | >=2.0.0                                                             | >=3.8          | >=7.4        | >=10.0      |
| [YOLOv6](https://github.com/meituan/YOLOv6)                      | >=1.8.0                                                 | >=2.0.0                                                             | >=3.8          | >=7.4        | >=10.0      |
| [YOLOv7](https://github.com/WongKinYiu/yolov7)                   | >=1.7.0,!=1.12.0                                        | >=2.4.1                                                             |                | >=8.0        | >=11.0      |
| [YOLOv8](https://github.com/ultralytics/ultralytics)             | >=1.8.0                                                 | >=2.0.0                                                             |                | >=7.4        | >=10.0      |
| [YOLOv9](https://github.com/WongKinYiu/yolov9)                   | >=1.7.0                                                 | >=2.4.1                                                             |                | >=8.0        | >=11.0      |
| [YOLOv10](https://github.com/THU-MIG/yolov10)                    | >=1.8.0                                                 | <=2.13.1                                                            | >=3.8 & <=3.11 | >=7.6        | >=10.1      |
| [YOLOv11](https://github.com/ultralytics/ultralytics)            | >=1.8.0                                                 | >=2.0.0                                                             | >=3.8          | >=7.4        | >=10.0      |

### CUDA 다운로드

```cardlink
url: https://developer.nvidia.com/cuda-toolkit-archive
title: "CUDA Toolkit Archive"
host: developer.nvidia.com
description: "CUDA 버전에 맞게 다운로드 링크"
favicon: https://dirms4qsy6412.cloudfront.net/assets/favicon-81bff16cada05fcff11e5711f7e6212bdc2e0a32ee57cd640a8cf66c87a6cbe6.ico
image: https://developer.download.nvidia.com/images/og-default.jpg
```

▽ 아래 코드만 실행해도됨

```sh title:"11.8 쿠다 설치"
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --override
```

### cuDNN 다운로드

```cardlink
url: https://developer.nvidia.com/cudnn-archive
title: "cuDNN Archive"
host: developer.nvidia.com
favicon: https://dirms4qsy6412.cloudfront.net/assets/favicon-81bff16cada05fcff11e5711f7e6212bdc2e0a32ee57cd640a8cf66c87a6cbe6.ico
image: https://developer.download.nvidia.com/images/og-default.jpg
description: "cuDNN 버전 모음"
```

▽ 아래 코드만 실행해도됨

```sh title:"cuDNN 9.6 설치"
cd ~
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.6.0.74_cuda11-archive.tar.xz
tar -xvf cudnn-linux-x86_64-9.6.0.74_cuda11-archive.tar.xz
```

```sh title:"cudnn 9.6을 cuda 경로에 복사하기"
cd ~/cudnn-linux-x86_64-9.6.0.74_cuda11-archive
sudo cp include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp lib/libcudnn* /usr/local/cuda-11.8/lib64
```

```sh title:"리눅스 경로 설정 11.8 버전"
echo 'export CUDA_HOME=/usr/local/cuda-11.8' >> ~/.bashrc 
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc 
echo 'export CUDADIR=$CUDADIR:$CUDA_HOME' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export TF_CPP_MIN_LOG_LEVEL=3' >> ~/.bashrc
```

```sh title:"버전 확인"
nvcc --version
```

### 아나콘다 설치

```cardlink
url: https://www.anaconda.com/download/success
title: "Download Now | Anaconda"
description: "Anaconda is the birthplace of Python data science. We are a movement of data scientists, data-driven enterprises, and open source communities."
host: www.anaconda.com
favicon: https://www.anaconda.com/wp-content/themes/berg-theme-child/assets/images/favicon/android-icon-192x192.png
image: https://www.anaconda.com/wp-content/uploads/2024/03/download-svgrepo-com-2.svg
```

```sh title:"anaconda 설치"
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
sh *sh
```

```sh title:"miniconda 설치"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh *.sh
```

### 텐서 설치 및 확인
콘다 환경 활성화 후 할 것들

```sh title:"텐서플로 설치 "
pip install 'tensorflow[and-cuda]'
```

```sh title:"텐서플로가 그래픽카드 잡고 있는지 확인"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

`[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]` ← 문구가 나오면 텐서 설치 확인

## vscode에서 접속
