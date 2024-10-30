# tf-matcomp
Code for "Abrupt Learning in Transformers: A Case Study on Matrix Completion" (NeurIPS 2024). [arXiv link](https://arxiv.org/abs/2410.22244)

Requirements can be found in the `env.yaml` file generated from the conda environment used for experiments. All experiments in this repository were done on a single (L40S / A100 / V100 / A40) GPU. 

Please update the `modeling_bert.py` file in your HuggingFace transformers library code (located at `miniconda3/envs/env-name/lib/python3.8/site-packages/transformers/models/bert/` if using miniconda) with the file provided here - this includes changes to remove `token_type_embeddings`, and for causal intervention (activation patching) experiments.

## Training

`$ python3 train.py --config configs/train.yaml`

- Training metrics can be tracked using W&B by setting the `wandb` flag to `True` in `train.yaml`, and updating your W&B credentials at line 188 in `train.py`. 

- Data sampling is done in `data.py`, the model is defined in `model.py`, and utility functions are in `utils.py`.

- For training individual components, initializing the other components to the weights at converged model, modify `train_embed.py` (this version trains embeddings only, keeping Attention and MLP layers fixed).



Code for GPT training is in `src/gpt/`; config file is `/src/configs/gpt.yaml`.

## Interpretability tests

All interpretability tests are through standalone scripts; please see below the experiment - code mapping:

| Task Description                                              | Python Script                                      |
|---------------------------------------------------------------|-------------------------------------------------------|
| Get attention maps [1]                                            | `att_maps.py`                                 |
| Comparing nuclear norm minimization and BERT                   | `compare_nuc_bert.py`                                 |
| Token intervention for verifying copying                      | `copy_check.py`                                       |
| Nuclear norm minimization (utils and experiments)                   | `cvx_nuc_norm.py`                                 |
| Embedding visualization / permuting positional embeds [2] | `embed.py`                                      |
| Out-of-distribution tests                                 | `ood.py`                                      |
| Activation patching (Attention layers)                                          | `patch.py`                        |
| Probe for input information                                 | `probe.py`                                            |
| Switching pre- and post-shift models                           | `switch_models.py`                                    |
| Uniform ablations                                             | `uniform_ablation.py`                                 |


`.yaml` config file for each of these (except `cvx_nuc_norm.py` that does not need one) has the same file name as the script in `configs/` directory.

[1] Options for attention maps includes type of masking (`mask_type` in config): random masking (`none`), masking specific rows (`row`), columns (`col`). For `row` and `col`, the same mask will be used for the whole batch, while for `none`, the masks will be sampled randomly for each matrix in the input batch.

[2] Specify the experiment required in `exp_type` in `configs/embeds.yaml`: 

| Variable     | Description                                    |
|--------------|------------------------------------------------|
| `token_norm` | $\ell_2$ norm of token embeddings              |
| `token_pca`  | PCA of token embeddings                        |
| `pos_tsne`   | TSNE of positional embeddings                  |
| `token_progress`   | Track token embedding progress                 |
| `pos_progress`   | Track positional embedding progress                 |
| `permute`    | Checking loss after randomly permuting positional embeddings |
