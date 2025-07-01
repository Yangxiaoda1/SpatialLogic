# SpatialLogic

1. 下载数据集：https://huggingface.co/datasets/yangxiaoda/SpacialLogic
2. 解压observation中的文件，删掉原压缩文件
3. 把observation中的视频以10倍下采样切分成帧，制作clips文件夹：python 1get_clips.py
4. 把clips文件夹的每个文件根据任务名称归类：python 2group.py
5. 由于“Place”指令对应的视频结尾部分多余，需要减除：python 3refine.py
6. 此时，文件结构为：
```bash
|-SpacialLogic
    |-clips
        |-327
            |-648642
                |-Place the held corn into the shopping cart's plastic bag.
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
```
如果要做拟合测试，可以只解压一个任务，构造类似结构：
```bash
|-SpacialLogic-Demo
    |-clips
    |-...
```
5. 拉取项目：
git clone https://github.com/Yangxiaoda1/SpatialLogic.git

[Optional]Cot数据生成：调用GPT接管，然后人工矫正
```bash
python datamaker.py
```

