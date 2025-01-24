# Fine-tunning Gemma (pretrained from HuggingFace) witih LoRA
^ training made compatable with Mac Intel (CPU)

## Highlights 
* Use Gemma (transformer-based) for text generation task
* Fine-tune using quantization method (LoRA) 
* Adapt the code to run with CPU-only PC 

 
### Project Structure 
This project directory is as followed: 
```
    .
    ├── .env                                <- store huggingface key for usage later
    ├── gemma_finetune.py                   <- main script
    ├── outputs                             <- output folder (being ignored by .gitignore)
    └── README.md
```


## Installation
git clone  https://github.com/yslidet/LiDETry_Gemma.git

### Prerequisite - HuggingFace Key
* Get the [read] key from your huggingface account
    * Note: click **acknowledge** at [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b) if you use it for the first time. 

* In terminal, login using HuggingFace CLI 
```bash
huggingface-cli login
```
- Create `.env` file (if not exist) and add your huggingface keytoken in the note below. 
```note
MY_HUGGINGFACE_KEY=[your READ token from Hugging Face]
```

### Prerequisite - Environment 
```bash
pip install -r requirement.txt
```


## Usage

```bash
#!/usr/bin/env bash
### ------ SETUP:start ------
# >> Activate conda ENV (e..g mac_intel)
conda activate <env_name> 

# >> Setup $PYTHONPATH
# cd [project_dir]
cd ..
export PYTHONPATH="$PWD"

# >> Name Tmux Session (?:optional)
# export tmux_sess = ?
### ------ SETUP:end ------

python gemma_finetune.py > '<log_filename>.txt'
```

## Acknowledgments
`gemma_finetune.py` : I learn of this script from Krish Naik (https://www.youtube.com/watch?v=UWo9r6flDjk), all credit of the code goes to him. 
* I adapt this script to run on my personal Mac Intel (CPU only).
