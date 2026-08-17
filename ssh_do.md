# RunPod SSH Setup

## 1. Transfer files (run on Mac)

```bash
# Code
scp -P <port> -r /Users/box/src/research2026/model/ root@<ip>:/workspace/

# Data
scp -P <port> -r /Users/box/src/research2026/data/data_full/ root@<ip>:/workspace/data/data_full/
```

## 2. On the pod — create directories

```bash
mkdir -p /workspace/data/data_full
```

## 3. Install dependencies

```bash
pip install transformers peft accelerate pandas pyarrow pyyaml torch
```

## 4. Login to HuggingFace

```bash
hf auth login
```

## 5. Run training in tmux

```bash
tmux new -s train
python3 /workspace/model/sft_bias.py --dataset gt --config /workspace/model/cfgs/train_config_cloud.yaml
```

Detach with `Ctrl+B + D`

Reattach later:
```bash
tmux attach -t train
```

## 6. Queue PS after GT finishes

```bash
until [ -f /workspace/model/adapters/llama-sft-gt_*/adapter_model.safetensors ]; do sleep 15; done && \
echo "GT done, starting PS..." && \
python3 /workspace/model/sft_bias.py --dataset ps --config /workspace/model/cfgs/train_config_cloud.yaml
```

## 7. Run inference

```bash
python3 /workspace/model/infer.py --config /workspace/model/cfgs/train_config_cloud.yaml
```
