
<div  align="center">
    <h1>wd1:  Weighted Policy Optimization for Diffusion Language Models Reasoning</h1>
  <p>We introduce wd1, a novel policy optimization approach that reformulates the objective as a weighted likelihood, requiring only a single approximation for the current parametrized policy likelihood</p>
</div>



![Results](media/comparison.pdf)


## Environment Setup

To setup the environment, run;
```
python -m venv .venv
pip install -r requirements.txt
```


## SFT
```bash
# First go to the SFT directory
cd SFT

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file ddp_config.yaml --main_process_port 29500 --num_processes 4 sft_train.py
```

## wd1
You must change the data directory for all the bash scripts. Change it based on your path. Or you could just export it before the run with this command.
```
export BASE_DATA=/home/diffusion-rl/data
```
Otherwise the code will use the default.


### RL only 
To run direct RL without SFT     
```
# Pattern
bash run/wll_NP_{datasetname}.sh
# Example
bash run/wll_NP_countdown.sh
```
### RL on top of SFT
To run RL on top of SFT     
```
# Pattern
bash run/wll_SFT_NP_{datasetname}.sh
# Example
bash run/wll_SFT_NP_countdown.sh
```

## Evaluation

The evaluation code is inside the `eval` directory.

- Run with `bash eval/run_eval_all.sh`
- Make sure to point to the correct checkpoint.
- The evaluation file will only save the generations; use the parser to calculate accuracy
- For example, baseline generations are in the `eval_baselines` directory. Use `python parse_and_get_acc.py` to print the accuracy.
