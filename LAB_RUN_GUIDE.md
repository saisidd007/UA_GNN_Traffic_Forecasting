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

### Native Windows PowerShell

The diagnostic and training Python scripts also run in native Windows. Use
`python` from your chosen environment instead of `.venv/bin/python`.
For the lab NVIDIA T1000 8 GB with driver 596.59, replace CPU-only PyTorch with:

```powershell
python -m pip install --upgrade --force-reinstall torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python check_lab_hardware.py --hardware-only
python check_lab_hardware.py --device cuda --batch-size 8 --output lab_gpu_report.json
```

`CUDA available` must be true. The driver version reported by `nvidia-smi` does
not need to match PyTorch's bundled CUDA runtime. The benchmark reports Windows
CPU/core and RAM information as well as NVIDIA driver details. Run it from the
repository folder. Send the JSON report before choosing final training settings.
The completed T1000/i9-10900K benchmark selected two CPU threads at batch size 8.
For this lab machine, use the native launcher in your activated Windows venv:

```powershell
python -u run_lab_50.py
```

Keep the terminal open and disable computer sleep. Both datasets run sequentially
on CUDA using seed 42, 50 epochs, all windows, batch size 8 and two CPU threads.
Logs appear in the terminal and under `paper_results/lab_t1000_full50_seed42/`.
After an interruption, use `python -u run_lab_50.py --resume`. Datasets whose
training and testing already completed are skipped. The current partial epoch
is repeated. This launcher does not restart itself after Windows reboots.
For explicit GPU training, add `--device cuda` to `paper_experiment.py`; it
refuses to fall back to CPU when CUDA is unavailable. The Bash/tmux commands
below are for Linux/WSL, not native PowerShell.

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

Before choosing thread counts, run the short hardware benchmark and share its
JSON report. It reports logical CPUs, process CPU limits, RAM and GPU details,
and compares actual model training/validation batches at several thread counts.
It does not run a full epoch or change your saved training runs.

```bash
.venv/bin/python -u check_lab_hardware.py
```

Send `lab_hardware_report.json`. Default test batch size is 2 to reduce GPU memory
requirements. The fastest thread count depends on batch size; after choosing
the proposed training batch, repeat with (for example):

```bash
.venv/bin/python -u check_lab_hardware.py --batch-size 8 --output lab_hardware_batch8.json
```

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

The latest checkpoint is replaced atomically after each completed epoch.
Resume checks saved model and training settings and refuses changed settings.
GPU RNG state is restored using CPU tensors as required by PyTorch. Exceptions
are printed in the log and recorded in `last_error.json` when possible. Progress
logs do not save optimizer state within an epoch: power loss or an interruption
can lose the current epoch's work. No script can guarantee against hardware,
disk, driver, or memory failures. GPU behavior must be tested on the lab machine.

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
