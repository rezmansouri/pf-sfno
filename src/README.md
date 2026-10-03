# Bridging the Training-Inference Gap in Steady-State Solar Wind Neural Surrogates

Code for training and evaluating an autoregressive Spherical Fourier Neural Operator (SFNO) surrogate for steady-state solar wind propagation from $30\,R_{\odot}$ to 1 AU.

<table>
  <tr>
    <td style="width: 50%; vertical-align: top;">
      <img src="assets/prediction_cr2149_hmi_masp_mas_std_0201.gif" width="100%">
    </td>
    <td style="width: 50%; vertical-align: top;">
      <img src="assets/rmse_cr2149_hmi_masp_mas_std_0201.gif" width="100%">
    </td>
  </tr>
</table>

*Solar wind radial velocity estimates and errors from Carrington Rotation 2149*

The repository includes implementations for:
- SFNO model training with teacher forcing and push-forward training
- Autoregressive radial propagation
- HUX-f numerical baseline
- Evaluation metrics and visualization utilities

## Repository Structure

```text
.
├── model.py              # SFNO model architecture
├── train.py              # Model training
├── predict.py            # Autoregressive prediction
├── test.py               # Model evaluation
├── metrics.py            # Evaluation metrics
├── training_utils.py     # Training utilities
├── utils.py              # General utilities
├── common_grid_utils.py  # Grid utilities
└── hux/                  # HUX-f implementation and evaluation
    ├── evaluate.py
    ├── predict_instance.py
    ├── metrics.py
    ├── hux_utils.py
    └── hux_code/
```


Training

`train.py` expects the following command-line arguments:

```python
data_path
split_csv_path
training_dataset_stats_path
batch_size
n_epochs
lift_project_ratio
hidden_channels
n_layers
prediction_horizon
warmup_pf_epochs
p_push_forward
save_test_preds
feed_grid_embeddings
radial_pushforward
```

### Prediction and Evaluation

Use `predict.py` for autoregressive radial propagation and test.py for evaluation using RMSE, RMGSE, and ACC.

The `hux/` directory contains the HUX-f baseline and its associated evaluation tools.

### Requirements

The experiments were developed using Python 3.10, PyTorch 2.2.1, and CUDA 12.1.