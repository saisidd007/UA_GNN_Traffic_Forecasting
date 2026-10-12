# Lab run: main model, one seed, 50 epochs on both datasets

Use Linux, or Ubuntu under WSL2 on Windows, with Python 3.10 or newer.
These commands run the corrected manuscript architecture interpretation, seed 42,
all training/validation/test windows, and 10 MC-dropout passes at final evaluation.
The manuscript omits some architecture details; this is not an exact reproduction
of its published results. The two datasets run sequentially.

## 1. Get the latest code

For the first download:

```bash
git clone https://github.com/saisidd007/UA_GNN_Traffic_Forecasting.git
cd UA_GNN_Traffic_Forecasting
```

For an existing checkout, enter its folder and run:

```bash
git pull --ff-only origin main
```

## 2. Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

For an NVIDIA GPU, `CUDA available` must say `True`. If it says `False`,
check the NVIDIA driver with `nvidia-smi` and install the PyTorch build recommended
for that machine at https://pytorch.org/get-started/locally/ before starting.
The program automatically chooses CUDA when available. For a CPU-only machine,
you can install the CPU wheel instead using:

```bash
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

The CSV datasets and graphs are already in the repository.

## 3. Start and leave running

Use `tmux` so closing SSH or the terminal does not interrupt training.
Install tmux using the lab's normal package installation procedure if necessary.

```bash
tmux new-session -d -s ua_gnn 'bash run_lab_50.sh'
tmux attach -t ua_gnn
```

Detach without stopping: press **Ctrl+B**, release, then **D**.
Reattach with `tmux attach -t ua_gnn`.
Keep the lab machine powered on and disable automatic sleep.
To deliberately stop, attach and press **Ctrl+C**.

If tmux is unavailable, run `bash run_lab_50.sh` in a terminal and keep that
terminal open until completion.

Defaults: batch size 64, four CPU threads, 50 epochs, early-stopping patience 50.
This forces all 50 epochs, while the validation-best checkpoint is used for test
evaluation. METR-LA finishes training and testing before PEMS-BAY starts.
GPU memory requirements depend on the card. If a fresh run fails with CUDA
out-of-memory, start a new output folder with a smaller batch size:

```bash
BATCH_SIZE=8 OUTPUT_ROOT=paper_results/lab_full_50_batch8_seed42 bash run_lab_50.sh
```

Do not change batch size or other model/training settings partway through a run.
More CPU threads are not automatically faster; four was faster than two on our
machine. Override using `THREADS=8` only if appropriate for the lab machine.

## 4. Watch progress

```bash
tail -f paper_results/lab_full_50_seed42/METR-LA.log
```

For the second dataset:

```bash
tail -f paper_results/lab_full_50_seed42/PEMS-BAY.log
```

Logs show the selected CPU/GPU, epoch number, training/validation/test phase,
batch count, percentage, elapsed time, and estimated remaining time for the
current phase. Default interval is 25 batches; the first and last batch are also
logged. A long test phase is expected: it uses 10 forward passes per test batch.
Progress is also saved in each dataset's `progress.jsonl`.

## 5. Resume after interruption

Use the same checkout, output root, batch size, thread count, and environment:

```bash
tmux new-session -d -s ua_gnn 'bash run_lab_50.sh resume'
```

The script resumes any dataset with `latest.pt`, and starts the next dataset
fresh if it has not started yet. Checkpoints save after each completed epoch;
an interrupted partial epoch is repeated. A dataset whose 50 epochs already
finished is evaluated again when resumed. Existing incomplete folders without
a checkpoint require inspection or a new output root; they are not overwritten.

## 6. Collect results

Each dataset has a folder under `paper_results/lab_full_50_seed42/` containing:

- `metrics.json`: final test scores, uncertainty metrics, device, and run settings.
- `persistence_baseline.json`: matched persistence comparison.
- `training.json`: completed epochs, training loss, validation error, epoch time.
- `best.pt`: validation-best model.
- `latest.pt`: checkpoint for resuming training.
- `predictions.npz`: saved test predictions and uncertainties.
- `progress.jsonl`: detailed progress events.

Check that `epochs_completed` is 50 in both `metrics.json` files before reporting
the runs as complete. Preserve the entire output directory.

## Runtime

On our CPU machine with four threads and batch size 64, measured first epochs
including full validation took 55 minutes for METR-LA and 2 hours 19 minutes
for PEMS-BAY. Extrapolated training time for both 50-epoch runs is about 162 hours,
plus final test evaluation. These timings do not predict the lab GPU's runtime.
Use the first completed epoch's `epoch_seconds` on the lab machine to estimate
its remaining training time; final MC evaluation adds extra time.
