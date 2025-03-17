#  ✨ Castle Game AI ✨  

Documents link: https://www.fit.uet.vnu.edu.vn/wp-content/uploads/2023/04/De-PROCON_2023.pdf

## Install Python Libraries and Packages

``` bash
pip install -e .
```

## Run script with your machine using random steps for testing

``` bash
usage: test_game.py [-h] [--show-screen SHOW_SCREEN] [--render RENDER]

options:
  -h, --help    show this help message and exit
  --show-screen SHOW_SCREEN
  --render      RENDER
```

Example command for testing game with 10 episodes:

```python
python3 test_game.py --show-screen
```

## Test game using pretrained model

```python
python3 test_model.py \
    --show-screen \
    --n-evals 10 \
    --model-path-1 ./trained_models/model.pt \
    --model-path-2 trained_models/model.pt \
    --load-model \
    --device cuda
```

## How to train model

Full options:
``` bash
usage: dqn_main.py [-h] [--show-screen SHOW_SCREEN] [-v] [--figure-path FIGURE_PATH] [--n-evals N_EVALS] [--gamma GAMMA] [--tau TAU] [--n-step N_STEP] [--lr LR] [--batch-size BATCH_SIZE]
                   [--optimizer OPTIMIZER] [--memory-size MEMORY_SIZE] [--num-episodes NUM_EPISODES] [--model-path MODEL_PATH] [--load-model]

options:
  -h, --help            show this help message and exit
  --show-screen SHOW_SCREEN
  -v, --verbose
  --figure-path FIGURE_PATH
  --n-evals N_EVALS
  --gamma GAMMA
  --tau TAU
  --n-step N_STEP
  --lr LR
  --batch-size BATCH_SIZE
  --optimizer OPTIMIZER
  --memory-size MEMORY_SIZE
  --num-episodes NUM_EPISODES
  --model-path MODEL_PATH
  --load-model
```

Example command for training model with 1M episodes:

```python
python3 dqn_main.py \
        --show-screen \
        --figure-path ./figures/ \
        --model-path ./trained_models/model.pt \
        --num-episodes 1000000 \
        --memory-size 64384 \
        --batch-size 256 \
        --gamma 0.99 \
        --tau 0.01 \
        --lr 1e-6 \
        --optimizer adamw \
        --n-step 3 \
        --verbose
```
