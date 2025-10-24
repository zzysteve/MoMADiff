# Towards Robust and Controllable Text-to-Motion via Masked Autoregressive Diffusion (ACM MM 2025)
### [[Paper]](https://arxiv.org/abs/2505.11013)  [[Project Page]](https://zzysteve.github.io/project_pages/momadiff)

If you find our code or paper helpful, please consider starring our repository and citing:
```
@article{zhang2025towards,
  title={Towards Robust and Controllable Text-to-Motion via Masked Autoregressive Diffusion},
  author={Zhang, Zongye and Kong, Bohan and Liu, Qingjie and Wang, Yunhong},
  journal={arXiv preprint arXiv:2505.11013},
  year={2025}
}
```

## Open Source TODOs
- [x] Code release
- [x] Pre-trained models
- [x] Training scripts
- [x] Evaluation scripts
- [ ] Keyframe-guided generation

## Quick Start

If you only want to generate motions using a specified description or keyframes, you can follow the steps in this section without needing to download datasets or train models.

### 1. Conda Environment
```bash
conda create -n momadiff python=3.10.14
conda activate momadiff
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```
We test our code on Python 3.10.14 and PyTorch 2.4.0

### 2. Models and Dependencies

#### Download Pre-trained Models from Huggingface
We provide the pre-trained models for generation and evaluation. You could run the following script to download them from [this repo](https://huggingface.co/SteveZh/momadiff_models).

```
python prepare/hf_download.py
```

#### Manually Download (Optional)
If you cannot download from Huggingface, please manually download the models from [Baidu Netdisk](https://pan.baidu.com/s/1-g_H8r06zXBWYxFAisKkpw?pwd=v2rg) (extraction code: `v2rg`) and place them in `./checkpoints`.

Directory structure:
```
checkpoints
├── kit
│   ├── kl_vae_ver0
│   │   ├── args_for_pretrained_klvae.pkl
│   │   ├── data_opt.txt
│   │   ├── meta
│   │   │   ├── mean.npy
│   │   │   └── std.npy
│   │   └── net_last.pth
│   ├── text_mot_match
│   │   └── model
│   │       └── finest.tar
│   └── Trans-B_EMA_klv0-stable_masked-only_diff1000
│       └── model
│           └── latest_ema.tar
└── t2m
    ├── kl_vae_ver0-stable
    │   ├── args_for_pretrained_klvae.pkl
    │   ├── data_opt.txt
    │   ├── meta
    │   │   ├── mean.npy
    │   │   └── std.npy
    │   └── net_last.pth
    ├── text_mot_match
    │   └── model
    │       └── finest.tar
    ├── text_mot_match_IDEA400
    │   └── model
    │       └── finest.tar
    ├── text_mot_match_kungfu
    │   └── model
    │       └── finest.tar
    └── Trans-B_EMA_klv0-stable_masked-only_diff1000
        └── model
            └── latest_ema.tar
```

#### Generate motions by text

Please refer to `text2motion_demo.ipynb`.

## Evaluation and Training

If you just want to generate motions using your own text and keyframes using our pre-trained models, please refer to the [Quick Start](#quick-start) section above.

If you want to evaluate our pretrained models or train your own models, please follow the additional instructions below to get the datasets.

### 3. Get Data

Due to license restrictions, we cannot directly provide the datasets used in our experiments. Follow the instructions below to obtain the datasets for model training and evaluation.

**(a). Standard Benchmarks**

**HumanML3D** - Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then place the produced dataset in `./dataset/HumanML3D`.

**KIT** - Download from [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then place result in `./dataset/KIT-ML`

**(b). Out-of-distribution Datasets for Keyframe-guided Generation**

We download data from [Motion-X++](https://huggingface.co/datasets/YuhongZhang/Motion-Xplusplus) in SMPL format, and process the data following [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git) protocols. The processed data is placed to `./dataset/KIT-ML` and `./dataset/kungfu` respectively.

**(c). Glove Embeddings for Evaluators**

Please refer to `prepare/download_glove.sh` to download the glove embeddings used by the evaluators.


### 4. Evaluate Pre-trained Models

#### Evaluate VAE Reconstruction

HumanML3D
```bash
python eval_klvae.py --out_dir checkpoints/t2m/kl_vae_ver0-stable_eval --resume_pth checkpoints/t2m/kl_vae_ver0-stable/net_last.pth --data_root dataset/HumanML3D --evaluator text_mot_match
```

Generalization on IDEA400
```bash
python eval_klvae.py --out_dir checkpoints/t2m/kl_vae_ver0-stable_eval --resume_pth checkpoints/t2m/kl_vae_ver0-stable/net_last.pth --data_root dataset/IDEA400 --evaluator text_mot_match_IDEA400
```

Generalization on Kungfu
```bash
python eval_klvae.py --out_dir checkpoints/t2m/kl_vae_ver0-stable_eval --resume_pth checkpoints/t2m/kl_vae_ver0-stable/net_last.pth --data_root dataset/kungfu --evaluator text_mot_match_kungfu
```

The evaluators for IDEA400 and Kungfu are trained using the code in [text-to-motion](https://github.com/EricGuo5513/text-to-motion).

#### Evaluate text-to-motion generation on standard benchmarks

Evaluation on HumanML3D dataset.
```bash
python eval_t2m.py --name Trans-B_EMA_klv0-stable_masked-only_diff1000 --config configs/t2m.yaml --use_ema true --diffusion_steps 1000 --which_model_for_eval latest_ema --loss_strategy masked --ddim_steps 100 --trans_infer_timesteps 9 --guidance_param 3
```

Evaluation on KIT-ML dataset.
```bash
python eval_t2m.py --name Trans-B_EMA_klv0-stable_masked-only_diff1000 --config configs/kit.yaml --use_ema true --diffusion_steps 1000 --which_model_for_eval latest_ema --loss_strategy masked --ddim_steps 100 --trans_infer_timesteps 9 --guidance_param 3
```

#### Evaluate keyframe-guided text-to-motion generation on out-of-distribution datasets.

Evaluation on Kungfu dataset.
```bash
python eval_t2m_cross_dataset.py --name Trans-B_EMA_klv0-stable_masked-only_diff1000 --config configs/t2m.yaml --use_ema true --which_model_for_eval latest_ema --inference_setting keyframe --diffusion_steps 1000 --ddim_steps 100 --evaluator text_mot_match_kungfu --data_dir ./dataset/kungfu
```

Evaluation on IDEA400 dataset. All samples (~12,000) are used for testing, it would take some time. You can set `repeat_time` to a smaller value for quick test.
```bash
python eval_t2m_cross_dataset.py --name Trans-B_EMA_klv0-stable_masked-only_diff1000 --config configs/t2m.yaml --use_ema true --which_model_for_eval latest_ema --inference_setting keyframe --diffusion_steps 1000 --ddim_steps 100 --evaluator text_mot_match_IDEA400 --data_dir ./dataset/IDEA400
```

### 5. Train your own models

Make sure you have downloaded and preprocessed datasets as described in Step 3.

There are two stages of training: (1) VAE stage, (2) Masked Autoregressive Transformer stage.

#### Train VAE

```bash
# HumanML3D dataset
python train_klvae.py --out_dir checkpoints/reprod_humanml3d_klvae --dataset_name t2m --learning_rate 5e-5 --lr 5e-5

# KIT-ML dataset
python train_klvae.py --out_dir checkpoints/reprod_kit_klvae --dataset_name kit --learning_rate 5e-5 --lr 5e-5
```

#### Train Masked Autoregressive Transformer

```bash
# HumanML3D dataset
python train_latentar_transformer.py --name reprod_t2m --config configs/reprod/t2m.yaml --eval_every_e 100 --trans_infer_timesteps 9 --guidance_param 3 --use_ema true --loss_strategy masked --diffusion_steps 1000 --ddim_steps 100

# KIT-ML dataset
python train_latentar_transformer.py --name reprod_kit --config configs/reprod/kit.yaml --eval_every_e 100 --trans_infer_timesteps 9 --guidance_param 3 --use_ema true --loss_strategy masked --diffusion_steps 1000  --ddim_steps 100
```

#### Evaluate your trained models
```bash
python eval_t2m.py --name reprod_t2m --config configs/reprod/t2m.yaml --use_ema true --diffusion_steps 1000 --which_model_for_eval latest_ema --loss_strategy masked --ddim_steps 100 --trans_infer_timesteps 9 --guidance_param 3

python eval_t2m.py --name reprod_kit --config configs/reprod/kit.yaml --use_ema true --diffusion_steps 1000 --which_model_for_eval latest_ema --loss_strategy masked --ddim_steps 100 --trans_infer_timesteps 9 --guidance_param 3
```


## Acknowlegements

Our code is mainly based on [MoMask](https://github.com/EricGuo5513/momask-codes), thanks to the authors for their great work and open-sourced code. We also thank the open-sourcing of these works:

[Guided-Diffusion](https://github.com/openai/guided-diffusion), [MotionGPT](https://github.com/qiqiApink/MotionGPT), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [MDM](https://github.com/GuyTevet/motion-diffusion-model/tree/main), [ParCo](https://github.com/qrzou/ParCo), and [MLD](https://github.com/ChenFengYe/motion-latent-diffusion/tree/main).

## License
This project is released under the MIT License (see `LICENSE`).

Third-party components and their licenses are summarized in `NOTICE`:
- MoMask (MIT)
- text-to-motion (MIT)
- MDM (MIT)
- MotionGPT (see upstream repo)

Datasets (HumanML3D, KIT-ML, Motion-X++, etc.) and body models (SMPL / SMPL-X) have separate licenses or terms of use. You must comply with their respective licenses independently.

If you redistribute modified versions, retain the original copyright notice and this license.
