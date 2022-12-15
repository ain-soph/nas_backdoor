
> **NOTE:** This is an implementation for anonymous ICLR submission [The Dark Side of AutoML: Towards Architectural Backdoor Search](https://openreview.net/forum?id=bsZULlDGXe) upon requests from reviewers. It doesn't stand for the final code release.

## Installation

1. Install `python==3.10 pytorch==1.12.1 torchvision==0.13.1`
2. Install trojanzoo
    ```
    git clone https://github.com/ain-soph/trojanzoo.git
    cd trojanzoo
    pip install -e .
    ```
3. Install [AutoDL](https://github.com/D-X-Y/AutoDL-Projects) and [NATSBench](https://github.com/D-X-Y/NATS-Bench)


## Configuration

1. Download NATSBench (NATS-tss-v1_0-3ffb9-full) following their official guidance.
2. Set `nats_path` in `./configs/trojanvision/model.yml`

## Experiment
The command to run each script is at the top of file (commented).
> `--download` option might be necessary for the first run to download datasets and models.
