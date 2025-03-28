<h1 id="BQDMY">AReaL: Ant Reasoning RL </h1>
![](https://intranetproxy.alipay.com/skylark/lark/0/2025/jpeg/163056495/1743133178438-df2b3c40-9d0f-4c7b-8118-7327df90ccc8.jpeg)

<font style="color:rgb(31, 35, 40);">AReaL (Ant Reasoning RL) is an open-sourced and efficient reinforcement learning training system for large reasoning models developed at </font>**<font style="color:rgb(31, 35, 40);">the RL Lab, Ant Research</font>**<font style="color:rgb(31, 35, 40);">, built upon the open-source project </font>[RealHF](https://github.com/openpsi-project/ReaLHF)<font style="color:rgb(31, 35, 40);">. With a 100% open-source commitment, including data, training details, infra, and models, AReaL aims to help everyone build their own AI agents easily with a low cost.  Our team likes milk tea. We hope people will like our project just like a-real-milk tea. </font>

<font style="color:rgb(31, 35, 40);"></font>

**<font style="color:rgb(31, 35, 40);">AReaL Highlights</font>**

+ <font style="color:rgb(31, 35, 40);">🛠️</font><font style="color:rgb(31, 35, 40);"> </font>**<font style="color:rgb(31, 35, 40);">Open & Reproducible</font>**<font style="color:rgb(31, 35, 40);">: We will continuously release </font>_<font style="color:rgb(31, 35, 40);">all code, datasets, and training recipes</font>_<font style="color:rgb(31, 35, 40);"> for RL training LLMs .</font>
+ <font style="color:rgb(31, 35, 40);">🚀</font><font style="color:rgb(31, 35, 40);"> </font>**<font style="color:rgb(31, 35, 40);">Scalability</font>**<font style="color:rgb(31, 35, 40);">: AReaL can seamlessly adapt to different computational resource settings, ranging from 1 single node to 1K GPUs.</font>
+ 🔪 **<font style="color:rgb(31, 35, 40);">Cutting-Edge Performances:</font>**<font style="color:rgb(31, 35, 40);"> AReaL can produce models with cutting-edge reasoning capabilities. We are actively working on other domains, such as coding and agent, as well. </font>

<h2 id="news"><font style="color:rgb(31, 35, 40);">News</font></h2>
**<font style="color:rgb(31, 35, 40);">[2025/03/31]</font>**<font style="color:rgb(31, 35, 40);"> </font>**<font style="color:rgb(31, 35, 40);">(v0.2, nickname Boba)</font>**<font style="color:rgb(31, 35, 40);"> Our milestone release Boba! Please call it A-ReaL-Boba! This release includes much accelerated training with SGLang support and SOTA 7B and 32B models on math reasoning. </font>

**<font style="color:rgb(31, 35, 40);">[2025/02/24] (v0.1)</font>**<font style="color:rgb(31, 35, 40);"> Our initial release includes reproducible results for 1.5B and 7B LRMs. Check our </font>[v0.1 technical blog](/blog/AReaL_v0_1.md)<font style="color:rgb(31, 35, 40);">.</font>

<h2 id="training-a-1.5b-lrm-from-the-distilled-model"><font style="color:rgb(31, 35, 40);">AReaL-boba Milestones</font></h2>
<font style="color:rgb(31, 35, 40);">In our boba release, we highlight the 3 most important milestones:</font>

+ <font style="color:rgb(31, 35, 40);">SGLang support</font>
+ <font style="color:rgb(31, 35, 40);">A SOTA 7B math reasoning model</font>
+ <font style="color:rgb(31, 35, 40);">A particularly competitive 32B model that can be trained with extremely low cost.</font>

For the complete training and model details, please check [our v0.2 technical blog](/blog/AReaL_v0_2.md). 

<h4 id="kEJjy"><font style="color:rgb(31, 35, 40);">SGLang support with 1.5x speedup on 7B Training</font></h4>
<font style="color:rgb(31, 35, 40);">note: vs v0.1</font>

<h4 id="k4xsF"><font style="color:rgb(31, 35, 40);">SOTA 7B model using RL on math reasoning</font></h4>
| <font style="color:rgb(31, 35, 40);">Model </font> | <font style="color:rgb(31, 35, 40);">AIME 2024</font> | <font style="color:rgb(31, 35, 40);">AIME 2025</font> | <font style="color:rgb(31, 35, 40);">GPQA</font> |
| :---: | :---: | :---: | :---: |
| <font style="color:rgb(31, 35, 40);">R1-Distill-Qwen-7B</font> | <font style="color:rgb(31, 35, 40);">55.0</font> | <font style="color:rgb(31, 35, 40);">39.7</font> | <font style="color:rgb(31, 35, 40);"></font> |
| <font style="color:rgb(31, 35, 40);">Light-R1-7B-DS</font> | <font style="color:rgb(31, 35, 40);">59.1</font> | <font style="color:rgb(31, 35, 40);">44.3</font> | <font style="color:rgb(31, 35, 40);"></font> |
| <font style="color:rgb(31, 35, 40);">AReaL-boba-RL-7B (ours)</font> | **61.9** | **48.3** | **** |


<h4 id="orhVc"><font style="color:rgb(31, 35, 40);">Approaching QwQ-32B performances using only 200 data samples</font></h4>
|  | <font style="color:rgb(31, 35, 40);">QwQ-32B</font> | <font style="color:rgb(31, 35, 40);">AReaL-boba-SFT-32B</font> |
| --- | :---: | :---: |
| <font style="color:rgb(31, 35, 40);">AIME 2024</font> | <font style="color:rgb(31, 35, 40);">78.9</font> | 78.8 |


<h2 id="getting-started"><font style="color:rgb(31, 35, 40);">Getting Started</font></h2>
<h3 id="Trych">Quick Start</h3>
```markdown
git clone https://github.com/inclusionAI/AReaL
cd AReaL

# Train the distilled 7B model
REAL_NO_EXT=1 pip3 install -e . --no-build-isolation
python3 -m realhf.apps.quickstart ppo-math --config examples/configs/7B-distill/areal-7B-distill-gpus-128.yaml

# Evaluate the 7B model
python3 evaluation/eval_and_aggregate.py --model_path $MODEL_PATH --max_gen_tokens 32768
```

<h3 id="jWpIm">Resources</h3>
+ [Tutorial](/examples/README.md)
+ [Tutorial (中文)](/examples/README_zh.md)

<h2 id="future-plan">Future Plan</h2>
AReaL is under active development. We will have major releases in a weekly manner. We also highly appreciate efforts from the community as well. Here we highlight our future research and development plan. 

<h3 id="fM6oh">System Development</h3>
- [x] Support for SGLang.
- [ ] Support for the latest vLLM and megatron-core packages.
- [ ] RL training with coding problems.
- [ ] Asynchronous generation and RL training.
- [ ] Optimizations for distributed training: expert parallel and zero-bubble pipelining.
- [ ] RL for vision-language models (VLM).
- [ ] Function calling and agent capabilities.

<h3 id="PCbLW">Algorithm Development</h3>
- [ ] The training receipe for 32B models.
- [ ] Multi-task RL training.
- [ ] Agentic capabilities with end-to-end RL.
- [ ] Stable RL training for larger MOE models.

<h2 id="acknowledgement"><font style="color:rgb(31, 35, 40);">Acknowledgement</font></h2>
<font style="color:rgb(31, 35, 40);">We would like to remark that major contributors are from </font>**<font style="color:rgb(31, 35, 40);">RL Lab at Ant Research</font>**<font style="color:rgb(31, 35, 40);"> and </font>**<font style="color:rgb(31, 35, 40);">Institute for Interdisciplinary Information Sciences, Tsinghua University</font>**<font style="color:rgb(31, 35, 40);">.</font>

<font style="color:rgb(31, 35, 40);">Our team has also received invaluable assistance from the Super Computing Technology (SCT) team at Ant Group, particularly in the realm of large-scale cluster operations and maintenance. </font>

<font style="color:rgb(31, 35, 40);">We also appreciate all the pioneer works from the community, particularly the </font>[ReaLHF](https://github.com/openpsi-project/ReaLHF)<font style="color:rgb(31, 35, 40);"> project from OpenPsi Inc. and those other projects, including but not limited to, </font>[DeepScaleR](https://github.com/agentica-project/deepscaler)<font style="color:rgb(31, 35, 40);">, </font>[Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main)<font style="color:rgb(31, 35, 40);">, </font>[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)<font style="color:rgb(31, 35, 40);">, </font>[VeRL](https://github.com/volcengine/verl)<font style="color:rgb(31, 35, 40);">, and </font>[SGLang](https://github.com/sgl-project/sglang)<font style="color:rgb(31, 35, 40);">.</font>

<h2 id="citation"><font style="color:rgb(31, 35, 40);">Citation</font></h2>
```plain
@article{mei2024realhf,
  title={ReaLHF: Optimized RLHF Training for Large Language Models through Parameter Reallocation},
  author={Mei, Zhiyu and Fu, Wei and Li, Kaiwei and Wang, Guangju and Zhang, Huanchen and Wu, Yi},
  journal={arXiv preprint arXiv:2406.14088},
  year={2024}
}
```

```plain
@misc{areal2025,
  author = {RL Lab, Ant Research},
  title = {AReaL: Ant Reasoning RL},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/inclusionAI/AReaL}},
}
```

