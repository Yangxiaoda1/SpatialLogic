# SpatialLogic

1. 下载数据集：https://huggingface.co/datasets/yangxiaoda/SpacialLogic
2. 解压observation中的文件，删掉原文件
3. 把observation中的视频以10倍下采样切分成帧：python get_clips.py
4. 此时，文件结构为：
|-SpacialLogic
    |-clips
    |-observation
        |-327
            |-648642
        |-...
    |-task
    |-qwenvl
        |-full
            |-mycheckpoint
            |-train.py
            |-inference.py
        |-lora
            |-mycheckpoint
            |-train.py
            |-inference.py
    |-other methods...
如果要做拟合测试，可以只解压一个任务，构造类似结构：
|-SpacialLogic-Demo
    |-clips
    |-...
5. 拉取项目：
git clone https://github.com/Yangxiaoda1/SpatialLogic.git
