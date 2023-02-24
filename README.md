## Installation

1. Install `python==3.10 pytorch==1.12.1 torchvision==0.13.1`
2. Install [trojanzoo](https://github.com/ain-soph/trojanzoo)
    ```
    git clone https://github.com/ain-soph/trojanzoo.git
    cd trojanzoo
    pip install -e .
    ```
3. Install [AutoDL](https://github.com/D-X-Y/AutoDL-Projects) and [NATSBench](https://github.com/D-X-Y/NATS-Bench)


## Configuration

1. Download NATSBench (NATS-tss-v1_0-3ffb9-full) following the official guidance.
2. Set `nats_path` in `./configs/trojanvision/model.yml`

## Experiment

The command to run the code is listed at the top of each file. For example, to run the genetic search for vulnerable architectures, use the following command:

```
    CUDA_VISIBLE_DEVICES=1 python ./projects/nas_backdoor/search_genetic.py --color --verbose 1 --model nats_bench --attack input_aware_dynamic --validate_interval 1 --train_mask_epochs 10 --epochs 10 --lr 1e-2 --natural --total_resume --save_suffix 1
```

> `--download` option may be necessary for the first run to download datasets and models.
