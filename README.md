<div  align="center">
    <h1>wd1:  Weighted Policy Optimization for Diffusion Language Models Reasoning</h1>
  <p>We introduce wd1, a novel policy optimization approach that reformulates the objective as a weighted likelihood, requiring only a single approximation for the current parametrized policy likelihood</p>
</div>



## Environment Setup
```conda create -n py39 python=3.9```

And 
```pip install -e .``` Here we need the CUDA_HOME for Deepspeed and Flashattention compilation.


## wd1++
```
bash start_train.sh <num-of-gpus> <per-device-batch-size> wd1 countdown
```

## Baseline: MDPO
```
bash start_train_mdpo.sh <num-of-gpus> <per-device-batch-size> grpo countdown
```

Dataset choices include `numina` (OpenR1), `math` (MATH500), `gsm8k`, `sudoku`.

