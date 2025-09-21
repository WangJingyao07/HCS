# VLM-HCS: Plug-in Hierarchical Coresets Selection for Scene Understanding

[![Awesome](https://img.shields.io/badge/Demo-green)](https://wangjingyao07.github.io/HCS.github.io/)
![Static Badge](https://img.shields.io/badge/ACMMM25-yellow)
![Static Badge](https://img.shields.io/badge/to_be_continue-orange)
![Stars](https://img.shields.io/github/stars/WangJingyao07/HCS)

This repository provides strong open-source VLM baselines (CLIP, LLaVA, Qwen2-VL) and
a **Hierarchical Coresets Selection (HCS)** module that can be plugged into these baselines
to select informative regions in a coarse→fine manner for scene understanding tasks.

## Setup
```bash
conda create -n vlm-hcs python=3.10 -y
conda activate vlm-hcs
pip install -r requirements.txt
```

## Quickstart (Example)
```bash
bash scripts/demo_clip_hcs.sh
```

## Notes
- Backbones are **frozen** by default; HCS acts as a pre-selection module.
- You can optionally train a tiny MLP scorer (`train/train_hcs_scorer.py`) to stabilize scores.



### Citation

If you find our work and codes useful, please consider citing our paper and star our repository (🥰🎉Thanks!!!):

```bibtex
@misc{wang2025advancing,
      title={Advancing Complex Wide-Area Scene Understanding with Hierarchical Coresets Selection}, 
      author={Jingyao Wang and Yiming Chen and Lingyu Si and Changwen Zheng},
      year={2025},
      eprint={2507.13061},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.13061}, 
}
