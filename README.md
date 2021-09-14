# SScanner - Web版文档扫描器

# 项目简介
项目的目标功能是实现`web版的文档扫描器`。使用 `HED + Hough Transfrom` 的方法检测文档边缘，但改成用 MobileNet V2作为轻量级网络基底。自己采集数据并使用OpenCV合成数据（文档扫描版和背景）。

网络的训练是在租的云GPU服务器上进行，使用 `Docker` 在 `Ubuntu` 上快速配置 `Tensorflowf + CUDA + Jupyter Notebook` 训练环境，远程调参、训练。由于签证时间限制没来得及标记真实数据，只用了合成的数据来训练。


最后把训练好的文档边缘检测网络整合到 `React` 项目中，再使用 web 版的 `Tensorflow JS` 和 OpenCV.JS 来完善剩下的文档边界框的计算。同时去掉了OpenCV.js中不需要的模块并重新编译，生成轻量级的Web版OpenCV库，体积缩小了9倍。


由于时间比较赶，只做到了文档的边界框角点的计算。剩下的文档投影变换和彩色文档阴影去除还有待开发...


# 软件需求
    NodeJS
    Python 3.7 (Anaconda)
    Jupyter Notebook (Anaconda)
    vscode-styled-components (VS Code extension, Optional)


# 安装依赖
    npm install
    pip install tensorflow
    pip install import_ipynb
    pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
    
On `Ubuntu`, you need to install extra dependencies for OnpenCV to get it run:

    apt-get install ffmpeg libsm6 libxext6 -y

# 项目预览
    npm start

![preview](/public/images/result1.png)
![preview](/public/images/result2.png)
![preview](/public/images/result3.png)