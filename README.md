# About

This project is an AI-powered application that checks text for the usage of AI (as funny as it may sound).

![application](https://raw.githubusercontent.com/haritonn/ai-detection/main/assets/example_1.png)

Also implemented:
+ `Streamlit` interface (with model cache);
+ `ClearML` integration;
+ Under the hood here is fine-tuned (`LoRA`) `BERT` architecture;


## Dataset

The model was trained on a labeled dataset of software requirements,
where each entry is marked as written by a **Human** or **ChatGPT**.
The dataset covers various scenarios (e.g. Travel Planning, Voting Systems,
Education Platforms) and includes both Functional and Nonfunctional requirements.

## Threshold

Model uses custom (currently hardcoded, sorry) probability threshold with value 30%. If probability of `AI` class
ge than 30%, the text is classified as AI-written.

# Usage 
## Installation

Prerequirements:
- `Python` (ofc);
- `uv` package manager;
- Installed & configured `clearml`.

```bash
git clone https://github.com/haritonn/ai-detection
cd ai-detection
uv sync
```

## Running
```sh
uv run streamlit run app.py
```

## Todo
+ Find & fine-tune on better quality dataset;
+ Add some plots to the `streamlit` interface;
+ Add configuration file;
+ ...
