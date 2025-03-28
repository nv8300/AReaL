<h1 align="center">
<em>AReaL</em>: A fully open-sourced and inclusive RL project for large reasoning models
</h1>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/assets/logo.png">
    <img alt="ReaL" src="/assets/logo.png" width="20%">
  </picture>
</p>

AReaL (Ant Reasoning RL) is an open-sourced and efficient reinforcement learning training system for large reasoning models developed at **the RL Lab, Ant Research**, built upon the open-source project [RealHF](https://github.com/openpsi-project/ReaLHF). With a 100% open-source commitment, including data, training details, infra, and models, AReaL aims to help everyone build their own AI agents easily with a low cost.  Our team likes milk tea. We hope people will like our project just like a-real-milk tea.

**AReaL Highlights**

+ 🛠️ **Open & Reproducible**: We will continuously release _all code, datasets, and training recipes_ for RL training LLMs .
+ 🚀 **Scalability**: AReaL can seamlessly adapt to different computational resource settings, ranging from 1 single node to 1K GPUs.
+ 🔪 **Cutting-Edge Performances:** AReaL can produce models with cutting-edge reasoning capabilities. We are actively working on other domains, such as coding and agent, as well. 

## News

**[2025/03/31] (v0.2, nickname Boba)** Our milestone release Boba! Please call it A-ReaL-Boba! This release includes much accelerated training with SGLang support and SOTA 7B and 32B models on math reasoning. 

**[2025/02/24] (v0.1)** Our initial release includes reproducible results for 1.5B and 7B LRMs. Check our [v0.1 technical blog](/blog/AReaL_v0_1.md).

## AReaL-boba Milestones

In our boba release, we highlight the 3 most important milestones:

+ SGLang support
+ A SOTA 7B math reasoning model
+ A particularly competitive 32B model that can be trained with extremely low cost.

For the complete training and model details, please check our [v0.2 technical blog](/blog/AReaL_v0_2.md). 

### SGLang support with 1.5x speedup on 7B Training

![throughput_comparision_with_v0.1.0.png](/assets/thpt_comparison.png) 


### SOTA 7B model using RL on math reasoning
| Model  | AIME 2024 | AIME 2025 | GPQA |
| :---: | :---: | :---: | :---: |
| R1-Distill-Qwen-7B | 55.0 | 39.7 |  |
| Light-R1-7B-DS | 59.1 | 44.3 |  |
| AReaL-boba-RL-7B (ours) | **61.9** | **48.3** | **** |


### Approaching QwQ-32B performances using only 200 data samples
|  | QwQ-32B | AReaL-boba-SFT-32B |
| --- | :---: | :---: |
| AIME 2024 | 78.9 | 78.8 |


## Getting Started

### Quick Start
```markdown
git clone https://github.com/inclusionAI/AReaL
cd AReaL

# Train the distilled 7B model
REAL_NO_EXT=1 pip3 install -e . --no-build-isolation
python3 -m realhf.apps.quickstart ppo-math --config examples/configs/7B-distill/areal-7B-distill-gpus-128.yaml

# Evaluate the 7B model
python3 evaluation/eval_and_aggregate.py --model_path $MODEL_PATH --max_gen_tokens 32768
```

### Resources
+ [Tutorial](/examples/README.md)
+ [Tutorial (中文)](/examples/README_zh.md)

## Future Plan
AReaL is under active development. We will have major releases in a weekly manner. We also highly appreciate efforts from the community as well. Here we highlight our future research and development plan. 

### System Development
- [x] Support for SGLang.
- [ ] Support for the latest vLLM and megatron-core packages.
- [ ] RL training with coding problems.
- [ ] Asynchronous generation and RL training.
- [ ] Optimizations for distributed training: expert parallel and zero-bubble pipelining.
- [ ] RL for vision-language models (VLM).
- [ ] Function calling and agent capabilities.

### Algorithm Development
- [ ] The training receipe for 32B models.
- [ ] Multi-task RL training.
- [ ] Improving agentic capabilities with end-to-end RL.
- [ ] Stable RL training for larger MOE models.

## Acknowledgement

We would like to remark that major contributors are from **RL Lab at Ant Research** and **Institute for Interdisciplinary Information Sciences, Tsinghua University**.

Our team has also received invaluable assistance from the Super Computing Technology (SCT) team at Ant Group, particularly in the realm of large-scale cluster operations and maintenance. 

We also appreciate all the pioneer works from the community, particularly the [ReaLHF](https://github.com/openpsi-project/ReaLHF) project from OpenPsi Inc. and those other projects, including but not limited to, [DeepScaleR](https://github.com/agentica-project/deepscaler), [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [veRL](https://github.com/volcengine/verl), and [SGLang](https://github.com/sgl-project/sglang).

## Citation
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

