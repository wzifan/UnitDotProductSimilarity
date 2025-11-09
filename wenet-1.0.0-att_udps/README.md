# mywenet

由wenet1.0.0修改，runtime参照最新版的wenet(2.0-3.0)并进行了测试和bug修复

以libtorch的部署方式为主

## Quick Start

#### docker环境搭建 (libtorch runtime)

1. **创建并启动一个linux的docker镜像 (可根据自己的实际情况适当修改)**

   以`ubuntu:20.04`为例(实际使用过程中请使用公司的docker镜像)

   ```shell
   docker pull ubuntu:20.04
   ```

   ```shell
   docker run -itd --gpus all --shm-size 4G --workdir /home/wzifan --entrypoint /bin/bash --name wenet_libtorch_v1 ubuntu:20.04
   ```

   ```shell
   docker exec -it --user root wenet_libtorch_v1 /bin/bash
   ```

2. **安装需要的软件包 (可自行换源)**

   ```shell
   apt update
   ```

   ```shell
   apt upgrade
   ```

   ```shell
   apt install -y git cmake wget build-essential vim
   ```

   安装python3，设置软连接

   ```shell
   apt install python3
   ln -s /usr/binpython3 /usr/bin/python
   ```

   也可安装miniconda，然后创建激活环境后`pip install -r [wenet根目录]/requirements.txt`

   设置vim

   ```shell
   vim /etc/vim/vimrc
   ```

   添加以下内容并保存退出

   ```shell
   set fileencodings=utf-8,ucs-bom,gb18030,gbk,gb2312,cp936
   set termencoding=utf-8
   set encoding=utf-8
   ```

   在用户配置文件~/.bashrc末尾添加以下内容以支持中文

   ```shell
   export LANG="C.UTF-8"
   export LANGUAGE="C.UTF-8"
   export LC_ALL="C.UTF-8"
   ```

3. **获取wenet代码**

   当前目录：/home/wzifan

   ```shell
   git clone -b release02 http://gitlab2.bitautotech.com/SHARESKILL/nlp-trainee/asr_mywenet2.git
   ```

4. **安装libtorch环境 (cmake)**

   请保证cmake版本大于等于3.14

   ```shell
   cd [wenet根目录]/runtime/libtorch
   ```

   ```shell
   mkdir build && cd build
   ```

   - **cpu**

   ```shell
   cmake .. -DCMAKE_BUILD_TYPE=Release -DFST_HAVE_BIN=ON
   ```

   ```shell
   cmake --build . --config Release
   ```

   - **gpu**

   **首先安装cuda和cudnn，并添加环境变量，具体步骤略**

   然后注意[wenet根目录]/runtime/core/cmake/libtorch.cmake中的**torch版本**、**cuda版本**、下载地址和**hash值**，可自行修改

   ```shell
   cmake .. -DCMAKE_BUILD_TYPE=Release -DFST_HAVE_BIN=ON -DGPU=ON
   ```

   ```shell
   cmake --build . --config Release
   ```

   注：遇到无法下载的可先wget下载，然后移到目标目录

5. **文件准备**

   可以在本地创建如下目录和文件结构的文件夹：

   ```
   work
   --make_graph
   ----lm
   ------lm.arpa
   ----dict
   ------lexicon.txt
   ------units.txt
   
   --model_dir
   ----conformer_without_uv.pt
   ----conformer_without_uv.zip
   ----global_cmvn
   ----train.yaml
   ----units.txt
   
   --wav_dir
   ----test1.wav
   ----test2.wav
   ----text
   ----wav.scp
   
   --path.sh
   ```

   **文件说明：**

   1. **`make_graph`**

   ```
   --make_graph
   ----lm
   ------lm.arpa
   ----dict
   ------lexicon.txt
   ------units.txt
   ```

   `make_graph`下存放用于构图的文件，用于语言模型的添加，包括语言模型`make_graph/lm/lm.apra`、词到字的文件`make_graph/dict/lexicon.txt`和建模单元文件`make_graph/dict/units.txt`

   - `lm.apra`

     `lm.apra`是一个关于词或字的n-gram语言模型，其名称必须为lm.arpa

   - `units.txt`

     `units.txt`是建模单元，也就是wenet数据准备产生的data/dict/lang_char.txt，其内容示例如下：

     ```
     <blank> 0
     <unk> 1
     ▁ 2
     A 3
     B 4
     C 5
     D 6
     E 7
     F 8
     G 9
     H 10
     I 11
     J 12
     K 13
     L 14
     M 15
     N 16
     O 17
     P 18
     Q 19
     R 20
     S 21
     T 22
     U 23
     V 24
     W 25
     X 26
     Y 27
     Z 28
     ○ 29
     一 30
     丁 31
     七 32
     万 33
     丈 34
     三 35
     ```

     第一列为建模单元，中文端到端语音识别一般以字为建模单元，第二列为建模单元序号。`<blank>`表示空，`<unk>`表示未在字表中的字，最后一个`<sos/eos>`表示开始或结束，其余的都为普通的字或字符。

   - `lexicon.txt`

     `lexicon.txt`是一个由词到字的文件，**当**`lm.apra`**是基于词的语言模型时**，`lexicon.txt`的第一列是所有的词，第二列到最后一列是这些词对应的那些字，示例内容如下：

     ```
     啊 啊
     啊啊啊 啊 啊 啊
     阿 阿
     阿尔 阿 尔
     阿根廷 阿 根 廷
     阿九 阿 九
     阿克 阿 克
     阿拉伯数字 阿 拉 伯 数 字
     阿拉法特 阿 拉 法 特
     阿拉木图 阿 拉 木 图
     阿婆 阿 婆
     阿文 阿 文
     阿亚 阿 亚
     阿育王 阿 育 王
     阿扎尔 阿 扎 尔
     ```

     **当**`lm.apra`**是基于字的语言模型时**，可以把字当作词，用`units.txt`中的所有的字作为`lexicon.txt`中的第一列和第二列的内容(`<blank>`、`<unk>`、`<sos/eos>`除外)，`lexicon.txt`的示例内容如下：

     ```
     乒 乒
     乓 乓
     乔 乔
     乖 乖
     乘 乘
     乙 乙
     乜 乜
     九 九
     乞 乞
     也 也
     ```

     可以待`work`文件夹上传至服务器后用类似如下的命令产生此时的`lexicon.txt`

     ```shell
     sed -n '3,5537p' units.txt | awk -F ' ' '{print $1,$1}' > lexicon.txt
     ```

   2. **`model_dir`**

      ```
      --model_dir
      ----conformer_without_uv.pt
      ----conformer_without_uv.zip
      ----global_cmvn
      ----train.yaml
      ----units.txt
      ```

      `model_dir`下存放的是模型相关的文件，包括模型训练并参数求平均后产生的checkpoint文件`conformer_without_uv.pt`，通过torch script导出的libtorch模型`conformer_without_uv.zip`，全局cmvn文件`global_cmvn`，训练配置文件`train.yaml`(在各例子训练产生的文件夹下，如`exp/conformer/train.yaml`)，建模单元文件`units.txt`(与各例子数据准备之后的`data/dict/lang_char.txt`一致)。

      其中`conformer_without_uv.zip`和`units.txt`是libtorch runtime环境下语音识别的必须文件，`conformer_without_uv.pt`、`global_cmvn`、`train.yaml`和`units.txt`是模型继续训练/参数微调的必须文件。

   3. **`wav_dir`**

      ```
      --wav_dir
      ----test1.wav
      ----test2.wav
      ----text
      ----wav.scp
      ```

      wav_dir下存放着待测试的音频文件，音频通道数为1，采样率为16000，采样大小为2，音频格式为wav，除了音频文件，还包括`text`和`wav.scp`，

      `text`为标签文件，每一行第一项为音频id，第二项为该音频对应的文本，中间用空格隔开，示例如下：

      ```
      001 甚至出现交易几乎停滞的情况
      002 一二线城市虽然也处于调整中
      ```

      `wav.scp`为音频路径文件，每行的第一项为音频id，第二项为音频路径，中间用空格隔开，示例如下：

      ```
      001 /home/wzifan/work/wav_dir/test1.wav
      002 /home/wzifan/work/wav_dir/test2.wav
      ```


   4. **`path.sh`**

      ```
      --path.sh
      ```

      `path.sh`为环境变量文件，供libtorch runtime环境下语音识别使用，其中`WENET_DIR`需指向wenet根目录，具体内容如下：

      ```shell
      export SRILM=/home/xxx/srilm
      export WENET_DIR=$PWD/../../..
      export BUILD_DIR=${WENET_DIR}/runtime/libtorch/build
      export OPENFST_PREFIX_DIR=${BUILD_DIR}/../fc_base/openfst-subbuild/openfst-populate-prefix
      export PATH=$PWD:${BUILD_DIR}/bin:${BUILD_DIR}/kaldi:${OPENFST_PREFIX_DIR}/bin:$PATH:$SRILM/bin/i686-m64:$SRILM/bin
      export MANPATH=$SRILM/man

      # NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
      export PYTHONIOENCODING=UTF-8
      export PYTHONPATH=../../:$PYTHONPATH
      ```

   **将包含了以上内容的work文件夹上传至服务器的docker中，目录为/home/wzifan/work）**

   将`asr_mywenet2`下的`tools`和`mytools`放入`/home/wzifan/work`目录

   ```shell
cd /home/wzifan/work
   ```

   ```shell
cp -r /home/wzifan/asr_mywenet2/tools .
   ```

   ```shell
cp -r /home/wzifan/asr_mywenet2/mytools .
   ```

   赋予权限

   ```shell
chmod 775 -R /home/wzifan
   ```

6. **无语言模型情况下的识别[非流式]**

   ```shell
   cd /home/wzifan/work
   ```

   - **激活环境变量**

   ```shell
   . ./path.sh
   ```

   - 识别单个音频

   ```shell
   decoder_main --ctc_weight 0.5 --reverse_weight 0.0 --chunk_size -1 --unit_path model_dir/units.txt --model_path model_dir/conformer_without_uv.zip --wav_path wav_dir/test1.wav
   ```

   - 识别多个音频(多线程)

   ```shell
   ./tools/decode.sh --nj 4 --warmup 30 --ctc_weight 0.5 --reverse_weight 0.0 --chunk_size -1 wav_dir/wav.scp wav_dir/text model_dir/conformer_without_uv.zip model_dir/units.txt result_without_lm
   ```

7. **加语言模型情况下的识别[非流式]**

   wenet的语言模型说明详见`[wenet根目录]/docs/lm/lm.md`，下图为`T.fst、L.fst、G.fst`的简要说明

   ![TLG](./images/TLG.png)

   - **加基于字的语言模型**

   使用如下命令，根据`units.txt`，生成`lexicon.txt`

   ```shell
   cd /home/wzifan/work/make_graph/dict
   ```

   ```shell
   mv lexicon.txt lexicon_backup.txt
   ```

   ```shell
   # 查看units.txt有多少行
   wc units.txt
   ```

   ```shell
   # 如果units.txt有5538行
   sed -n '3,5537p' units.txt | awk -F ' ' '{print $1,$1}' > lexicon.txt
   ```

   如前文所述，此时`lexicon.txt`示例如下

   ```
   乒 乒
   乓 乓
   乔 乔
   乖 乖
   乘 乘
   乙 乙
   乜 乜
   九 九
   乞 乞
   也 也
   ```

   根据`units.txt`和`lexicon.txt`构建`T.fst`和`L.fst`

   ```shell
   cd /home/wzifan/work/
   ```

   ```shell
   ./tools/fst/compile_lexicon_token_fst.sh make_graph/dict make_graph/temp make_graph/lang_temp
   ```

   `T.fst`和`L.fst`构建完成之后，使用`make_tlg.sh`构建`TLG.fst`

   ```shell
   ./tools/fst/make_tlg.sh make_graph/lm make_graph/lang_temp/ make_graph/lang
   ```

   使用`TLG.fst`进行加语言模型的语音识别

   - 识别单个音频

   ```shell
   decoder_main  --acoustic_scale 4.0 --ctc_weight 0.5 --reverse_weight 0.0 --chunk_size -1 --rescoring_weight 0.5 --beam 15.0 --lattice_beam 4.0 --max_active 7000 --blank_skip_thresh 0.98 --fst_path make_graph/lang/TLG.fst --dict_path make_graph/lang/words.txt --unit_path model_dir/units.txt --model_path model_dir/conformer_with_uv.zip --wav_path wav_dir/test1.wav
   ```

   - 识别多个音频(多线程)

   ```shell
   ./tools/decode.sh --nj 4 --warmup 30 --acoustic_scale 4.0 --ctc_weight 0.5 --reverse_weight 0.0 --chunk_size -1 --rescoring_weight 0.5 --beam 15.0 --lattice_beam 4.0 --max_active 7000 --blank_skip_thresh 0.98 --fst_path make_graph/lang/TLG.fst --dict_path make_graph/lang/words.txt wav_dir/wav.scp wav_dir/text model_dir/conformer_with_uv.zip model_dir/units.txt result
   ```

   `nj`为线程数，

   `wav_dir/wav.scp`为音频列表文件，

   `wav_dir/text`为标签文件，如果不需要计算字错率或没有标签文件，只需要指定一个存在的空文件即可。

   `model_dir/units.txt`为建模单元文件，其与`make_graph/dict/units.txt`一致。

   `make_graph/lang/TLG.fst`为构图得到的WFST图，`make_graph/lang/words.txt`为该`TLG.fst`对应的词典，是由`tools/fst/make_tlg.sh`生成的。

   `acoustic_scale`用于调整声学得分的大小，使得声学得分与语言模型等分数在同一个数量级上，`rescoring_weight`为语言模型重打分比重，即使重打分比重为0，也会因WFST图中的一些常数值而使得识别结果与不加语言模型不一样。

   - **加基于词的语言模型**

   查看准备好的`lexicon.txt`，建议使用语言模型中所有的词作为`lexicon.txt`第一列所有的词，如上文(5文件准备)所述，`lexicon.txt`示例如下：

   ```
   啊 啊
   啊啊啊 啊 啊 啊
   阿 阿
   阿尔 阿 尔
   阿根廷 阿 根 廷
   阿九 阿 九
   阿克 阿 克
   阿拉伯数字 阿 拉 伯 数 字
   阿拉法特 阿 拉 法 特
   阿拉木图 阿 拉 木 图
   阿婆 阿 婆
   阿文 阿 文
   阿亚 阿 亚
   阿育王 阿 育 王
   阿扎尔 阿 扎 尔
   ```

   使用`mytools/remove_oov_in_lexicon.py`去除`lexicon.txt`中含 ”超纲字“ 的词及其对应的字，“超纲字”指`units.txt`中没有的字

   ```
   mv make_graph/dict/lexicon.txt make_graph/dict/lexicon_backup.txt
   ```

   ```shell
   python3 mytools/remove_oov_in_lexicon.py --units_path model_dir/units.txt --lexicon_path make_graph/dict/lexicon_backup.txt --new_lexicon_path make_graph/dict/lexicon.txt
   ```

   得到符合要求的`lexicon.txt`后，下面的步骤与加基于字的语言模型一样

   根据`units.txt`和`lexicon.txt`构建`T.fst`和`L.fst`

   ```shell
   cd /home/wzifan/work/
   ```

   ```shell
   ./tools/fst/compile_lexicon_token_fst.sh make_graph/dict make_graph/temp make_graph/lang_temp
   ```

   `T.fst`和`L.fst`构建完成之后，使用`make_tlg.sh`构建`TLG.fst`

   ```shell
   ./tools/fst/make_tlg.sh make_graph/lm make_graph/lang_temp/ make_graph/lang
   ```

   使用`TLG.fst`进行加语言模型的语音识别

   - 识别单个音频

   ```shell
   decoder_main  --acoustic_scale 4.0 --ctc_weight 0.5 --reverse_weight 0.0 --chunk_size -1 --rescoring_weight 0.5 --beam 15.0 --lattice_beam 4.0 --max_active 7000 --blank_skip_thresh 0.98 --fst_path make_graph/lang/TLG.fst --dict_path make_graph/lang/words.txt --unit_path model_dir/units.txt --model_path model_dir/conformer_with_uv.zip --wav_path wav_dir/test1.wav
   ```

   - 识别多个音频(多线程)

   ```shell
   ./tools/decode.sh --nj 4 --warmup 30 --acoustic_scale 4.0 --ctc_weight 0.5 --reverse_weight 0.0 --chunk_size -1 --rescoring_weight 0.5 --beam 15.0 --lattice_beam 4.0 --max_active 7000 --blank_skip_thresh 0.98 --fst_path make_graph/lang/TLG.fst --dict_path make_graph/lang/words.txt wav_dir/wav.scp wav_dir/text model_dir/conformer_with_uv.zip model_dir/units.txt result
   ```

8. **流式语音识别[不加语言模型]**

   

9. **流式语音识别[加语言模型]**



















------

# WeNet

[**中文版**](https://github.com/wenet-e2e/wenet/blob/main/README_CN.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/wenet-e2e/wenet)

[**Roadmap**](ROADMAP.md)
| [**Docs**](https://wenet-e2e.github.io/wenet/)
| [**Papers**](https://wenet-e2e.github.io/wenet/papers.html)
| [**Runtime (x86)**](https://github.com/wenet-e2e/wenet/tree/main/runtime/server/x86)
| [**Runtime (android)**](https://github.com/wenet-e2e/wenet/tree/main/runtime/device/android/wenet)
| [**Pretrained Models**](docs/pretrained_models.md)

**We** share neural **Net** together.

The main motivation of WeNet is to close the gap between research and production end-to-end (E2E) speech recognition models,
to reduce the effort of productionizing E2E models, and to explore better E2E models for production.

Note: please read `modify_list.txt` before use, please ensure that the `wenet` folder in each example is consistent with the `wenet` in the root directory

## Highlights

* **Production first and production ready**: The core design principle of WeNet. WeNet provides full stack solutions for speech recognition.
  * *Unified solution for streaming and non-streaming ASR*: [U2 framework](https://arxiv.org/pdf/2012.05481.pdf)--develop, train, and deploy only once.
  * *Runtime solution*: built-in server [x86](https://github.com/wenet-e2e/wenet/tree/main/runtime/server/x86) and on-device [android](https://github.com/wenet-e2e/wenet/tree/main/runtime/device/android/wenet) runtime solution.
  * *Model exporting solution*: built-in solution to export model to LibTorch/ONNX for inference.
  * *LM solution*: built-in production-level [LM solution](docs/lm.md).
  * *Other production solutions*: built-in contextual biasing, time stamp, endpoint, and n-best solutions.

* **Accurate**: WeNet achieves SOTA results on a lot of public speech datasets.
* **Light weight**: WeNet is easy to install, easy to use, well designed, and well documented.

## Performance Benchmark

Please see `examples/$dataset/s0/README.md` for benchmark on different speech datasets.

## Installation(Python Only)

If you just want to use WeNet as a python package for speech recognition application,
just install it by `pip`, please note python 3.6+ is required.
``` sh
pip3 install wenet
```

And please see [doc](runtime/binding/python/README.md) for usage.


## Installation(Training and Developing)

- Clone the repo
``` sh
git clone https://github.com/wenet-e2e/wenet.git
```

- Install Conda: please see https://docs.conda.io/en/latest/miniconda.html
- Create Conda env:

``` sh
conda create -n wenet python=3.8
conda activate wenet
pip install -r requirements.txt
conda install pytorch=1.10.0 torchvision torchaudio=0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- Optionally, if you want to use x86 runtime or language model(LM),
you have to build the runtime as follows. Otherwise, you can just ignore this step.

``` sh
# runtime build requires cmake 3.14 or above
cd runtime/server/x86
mkdir build && cd build && cmake .. && cmake --build .
```

## Discussion & Communication

Please visit [Discussions](https://github.com/wenet-e2e/wenet/discussions) for further discussion.

For Chinese users, you can aslo scan the QR code on the left to follow our offical account of WeNet.
We created a WeChat group for better discussion and quicker response.
Please scan the personal QR code on the right, and the guy is responsible for inviting you to the chat group.

If you can not access the QR image, please access it on [gitee](https://gitee.com/robin1001/qr/tree/master).

| <img src="https://github.com/robin1001/qr/blob/master/wenet.jpeg" width="250px"> | <img src="https://github.com/robin1001/qr/blob/master/binbin.jpeg" width="250px"> |
| ---- | ---- |

Or you can directly discuss on [Github Issues](https://github.com/wenet-e2e/wenet/issues).

## Contributors

| <a href="https://www.chumenwenwen.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/chumenwenwen.png" width="250px"></a> | <a href="http://lxie.npu-aslp.org" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/colleges/nwpu.png" width="250px"></a> | <a href="http://www.aishelltech.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/aishelltech.png" width="250px"></a> | <a href="http://www.ximalaya.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/ximalaya.png" width="250px"></a> | <a href="https://www.jd.com" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/jd.jpeg" width="250px"></a> |
| ---- | ---- | ---- | ---- | ---- |
| <a href="https://horizon.ai" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/hobot.png" width="250px"></a> | <a href="https://thuhcsi.github.io" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/colleges/thu.png" width="250px"></a> | <a href="https://www.nvidia.com/en-us" target="_blank"><img src="https://raw.githubusercontent.com/wenet-e2e/wenet-contributors/main/companies/nvidia.png" width="250px"></a> | | | |

## Acknowledge

1. We borrowed a lot of code from [ESPnet](https://github.com/espnet/espnet) for transformer based modeling.
2. We borrowed a lot of code from [Kaldi](http://kaldi-asr.org/) for WFST based decoding for LM integration.
3. We referred [EESEN](https://github.com/srvk/eesen) for building TLG based graph for LM integration.
4. We referred to [OpenTransformer](https://github.com/ZhengkunTian/OpenTransformer/) for python batch inference of e2e models.

## Citations

``` bibtex
@inproceedings{yao2021wenet,
  title={WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit},
  author={Yao, Zhuoyuan and Wu, Di and Wang, Xiong and Zhang, Binbin and Yu, Fan and Yang, Chao and Peng, Zhendong and Chen, Xiaoyu and Xie, Lei and Lei, Xin},
  booktitle={Proc. Interspeech},
  year={2021},
  address={Brno, Czech Republic },
  organization={IEEE}
}

@article{zhang2022wenet,
  title={WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit},
  author={Zhang, Binbin and Wu, Di and Peng, Zhendong and Song, Xingchen and Yao, Zhuoyuan and Lv, Hang and Xie, Lei and Yang, Chao and Pan, Fuping and Niu, Jianwei},
  journal={arXiv preprint arXiv:2203.15455},
  year={2022}
}
```
