# _Semixup_

By Huy Hoang Nguyen, Simo Saarakkala, Matthew Blaschko, and Aleksei Tiulpin.

Implementation of the paper (https://arxiv.org/abs/2003.01944),

(c) Huy Hoang Nguyen, University of Oulu, Finland.

## Introduction
_Semixup_ is a semi-supervised learning method based on in/out-of-manifold regularization. In the task of automatic grading Kellgren-Lawrence (KL) score for knee osteoathritis diagnosis, Semixup shows its data-efficiency as achieving a comparable performance with over 12 times less labeled data than a well-tuned SL requires.

![Semixup](https://github.com/MIPT-Oulu/semixup/blob/master/docs/semixup.png "Semixup")

![Knee OA samples](https://github.com/MIPT-Oulu/semixup/blob/master/docs/kneeoa_samples.png "Knee OA samples")


## Setup
Create a virtual environment using `conda`
```bash
conda create -n semixup python=3.7
conda activate semixup
```
Install Collagen framework (v0.0.1) by
```bash
pip install git+https://github.com:MIPT-Oulu/Collagen.git@0.0.1
```
Then, clone and setup Semixup by
```bash
git clone https://github.com/MIPT-Oulu/semixup.git
cd semixup/
pip install -e .
```

## Prepare Data
Training/validation data: the Osteoarthritis Initiative (OAI, https://nda.nih.gov/oai/).

Independent test data: the Multicenter Osteoarthritis Study (MOST, http://most.ucsf.edu/)

The script below:
 - crops lateral and medial sides of all knee images,
 - divides OAI data into labeled and unlabeled parts, and splits each of them into 5 folds,
 - prepares the full OAI data setting,
 - processes and cleans MOST data for evaluation.

```bash
cd scripts/
./prepare_data.sh
```
Default arguments assume the following directory structure
```
data/
├── MOST_OAI_FULL_0_2           # Preprocessed whole knee images
├── X-Ray_Image_Assessments_SAS # OAI metadata
└── most_meta                   # MOST metadata
```

## Training and Evaluation
### Supervised Learning
Train 5 folds of each architecture
```bash
cd <root>
./scripts/run_sl.sh <batch_size> <model> <comment>
```

Run archiecture selection using
```bash
cd <root>
./scripts/run_arch_selection.sh <batch_size>
```
### Ablation Study for Semixup's regularizers
```bash
cd <root>
./scripts/run_ablation_losses.sh <n_labels_per_klg> # 100 or 500
```

### Semi-Supervised Learning Methods
You can run a common command for training _Semixup_ or other SSL baselines.

Method name can be either `semixup`, `mixmatch`, `ict`, `pimodel` (with case-sensitive).
The amount of labeled data per KL grade can be 50, 100, 500, 1000.
```bash
cd <root>
./scripts/run_ssl.sh <n_labels_per_klg> <method_name> <comment>
```

### SL and SSL Comparisons
The evaluation to compare best SL and SSL models were independently done on MOST data. 

You need to prepare an intermediate file of the best models by
```bash
cd common/
python prepare_models_eval.py
```
then run evaluation using either `eval.py` or `eval_with_ci.py`. Please check those files for appropriate arguments.

### Statistical Tests
#### Mixed-effects model
File `mixed_effects_models.py` in `scripts/significance_tests/` aims to prepare data for running generalized mixed-effects in R.

#### One-sided Wilcoxon Signed-rank Test 
Code for Wilcoxon test is also in `scripts/significance_tests/`.
You first need to prepare data using the `prepare_data.py` file, then run `wilcoxon_test.py`.

### Pretrained Models
Download the zip file at [link](https://login.microsoftonline.com/9f9ce49a-5101-4aa3-8c75-0d5935ad6525/oauth2/authorize?client_id=00000003-0000-0ff1-ce00-000000000000&response_mode=form_post&protectedtoken=true&response_type=code%20id_token&resource=00000003-0000-0ff1-ce00-000000000000&scope=openid&nonce=50CE31E11C0D7C803D197586882F66E3E9F7007ECCFDAE62-375C0C7C640D48F6875F80867DFFA3C4FB0EC062D9DA64727134CAEEDCFFF282&redirect_uri=https:%2F%2Funioulu-my.sharepoint.com%2F_forms%2Fdefault.aspx&wsucxt=1&cobrandid=11bd8083-87e0-41b5-bb78-0bc43c8a8e8a&client-request-id=96343d9f-20d7-a000-c15b-09bcbc4f73d8).

## Citation
Please cite the paper below if you find repo useful.
```
@article{hoang2020semixup,
  title={Semixup: In-and Out-of-Manifold Regularization for Deep Semi-Supervised Knee Osteoarthritis Severity Grading from Plain Radiographs},
  author={Hoang Nguyen, Huy and Saarakkala, Simo and Blaschko, Matthew and Tiulpin, Aleksei},
  journal={arXiv},
  pages={arXiv--2003},
  year={2020}
}
```

## License
The codes and the pretrained models are not available for any commercial use including research for commercial purposes.
