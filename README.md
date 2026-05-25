# Desktop AI Trainer

Desktop AI Trainer is a local open-source project for learning how desktop AI agents work.

> Important: this project includes an AI agent that can automatically control your computer in live-control mode.
> It can send real keyboard and mouse input through normal desktop automation APIs.
> Treat it like any other automation software that can click buttons, press keys, and affect the active application.

The project lets you:

- record screenshots from your screen;
- record allowed keyboard and mouse actions;
- build a dataset from your demonstrations;
- train a PyTorch imitation-learning model;
- run the model in dry-run mode without controlling your computer;
- optionally run live-control mode where the AI can automatically send keyboard and mouse actions.

The idea is simple: you demonstrate actions, the project saves screen frames and input events, and a model learns to predict similar actions from screenshots. In dry-run mode the model only logs predictions. In live-control mode the model can execute allowed actions on your desktop.

## For Developers

This repository is a desktop AI automation scaffold.

The codebase contains components for:

- screen capture;
- input event recording;
- supervised imitation learning;
- dry-run prediction logging;
- live keyboard/mouse control;
- safety checks around the active window, confidence, hotkeys, and action rate.

If you are reviewing or extending this project, assume that live-control code can affect the user's machine. Keep the action space small, explicit, and profile-driven. Do not add hidden integrations, memory readers, network sniffers, injection, or bypass logic.

## What This Project Is Not

Desktop AI Trainer is not a cheat, exploit, or anti-cheat bypass tool.

It does not use:

- process memory reading;
- packet sniffing;
- network traffic analysis;
- code injection;
- anti-cheat bypasses;
- hidden game or application APIs.

It only uses normal screen capture and normal keyboard/mouse input.

Use it only in safe, local, and allowed environments.

## Safety

Live-control mode can send real keyboard and mouse input to your computer automatically.

That means it may click, type, press hotkeys, or interact with the active allowed window.

Do not start live-control first.

Recommended flow:

1. Record demonstrations.
2. Build the dataset.
3. Inspect the dataset preview.
4. Train the model.
5. Run dry-run mode.
6. Analyze prediction logs.
7. Use live-control only if dry-run looks safe.

Safety controls include:

- dry-run mode;
- emergency stop hotkey;
- pause hotkey;
- allowed window title check;
- action rate limit;
- confidence threshold;
- full action logging.

Default hotkeys:

```text
Ctrl+Shift+Q  emergency stop
Ctrl+Shift+P  pause/resume
```

## Requirements

- Python 3.11+
- Windows is recommended for desktop input recording
- PyTorch
- OpenCV
- Pillow
- mss
- pynput
- pyautogui
- pandas
- matplotlib
- pydantic
- pyyaml
- pytest

## Installation

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check that the project starts:

```powershell
python -m src.main
```

Expected output:

```text
Desktop AI Trainer is ready.
```

## Profiles

A profile defines what the project is allowed to record or execute.

Profiles are stored in:

```text
configs/profiles/
```

Inspect a profile:

```powershell
python -m src.input_control.action_space --profile lineage2_private
```

A profile contains:

- allowed window title;
- screen capture region;
- FPS;
- confidence threshold;
- action delay;
- emergency stop hotkey;
- pause hotkey;
- allowed actions.

Only actions listed in the profile are allowed.

If an action is not in the profile, the recorder ignores it and the live agent must not execute it.

## Recording Demonstrations

Start recording:

```powershell
python -m src.recorder.record_demo --profile lineage2_private --episode demo_001
```

During the countdown, switch to the target window.

The recorder saves:

```text
data/demos/{profile}/{episode}/frames/
data/demos/{profile}/{episode}/actions.csv
data/demos/{profile}/{episode}/metadata.json
```

Before recording:

- close private windows;
- do not record passwords;
- do not record sensitive documents;
- start with short test recordings;
- check that `total_actions` is greater than 0.

Debug input capture:

```powershell
python -m src.recorder.debug_input --profile lineage2_private --seconds 10
```

This command helps check whether Python can see your allowed keyboard and mouse input.

## Building A Dataset

Build a dataset:

```powershell
python -m src.dataset.build_dataset --profile lineage2_private
```

Create a visual preview:

```powershell
python -m src.dataset.visualize_dataset --profile lineage2_private --samples 20
```

Preview output:

```text
outputs/debug/{profile}_dataset_samples.png
```

Always inspect the preview before training.

If labels are wrong, record better demonstrations or fix the data first.

## Training

Train the imitation model:

```powershell
python -m src.training.train_imitation --profile lineage2_private --epochs 10
```

Training outputs:

```text
models/{profile}_imitation_policy.pt
outputs/reports/{profile}_training_metrics.csv
outputs/reports/{profile}_training_curves.png
```

Evaluate the model:

```powershell
python -m src.training.evaluate_imitation --profile lineage2_private
```

Evaluation output:

```text
outputs/reports/{profile}_evaluation_report.txt
```

High accuracy can be misleading if most frames are `no_op`.

Always check:

- class distribution;
- confusion matrix;
- top-3 predictions;
- dry-run logs.

## Dry-Run Mode

Dry-run mode predicts actions but does not press keys, move the mouse, or click.

Run dry-run:

```powershell
python -m src.inference.dry_run_agent --profile lineage2_private
```

Log output:

```text
outputs/logs/{profile}_dry_run_predictions.csv
```

Use dry-run to check:

- predicted action;
- confidence;
- top-3 predictions;
- forced `no_op`;
- wrong-window behavior.

Do not use live-control if dry-run looks unstable.

## Analyzing Predictions

Analyze dry-run logs:

```powershell
python -m src.analysis.analyze_predictions --profile lineage2_private
python -m src.analysis.plot_action_distribution --profile lineage2_private
```

Outputs:

```text
outputs/reports/{profile}_prediction_analysis.txt
outputs/reports/{profile}_action_distribution.png
```

Do not run live-control if:

- `no_op` is almost everything;
- one action dominates;
- confidence is low;
- predictions switch too often;
- predictions are clearly wrong.

## Live-Control Mode

Live-control sends real keyboard and mouse input.

Use it only after dry-run looks safe.

Start live-control:

```powershell
python -m src.inference.run_agent --profile lineage2_private
```

Live log:

```text
outputs/logs/{profile}_live_actions.csv
```

Emergency stop:

```text
Ctrl+Shift+Q
```

If something goes wrong, stop the agent, inspect the logs, improve the data, and return to dry-run.

## Running Tests

```powershell
python -m pytest
```

## Project Structure

```text
configs/      YAML profiles
data/         local demos, videos, datasets
models/       trained model files
outputs/      debug images, logs, reports
src/          source code
tests/        pytest tests
```

Generated data, trained models, logs, and reports are ignored by Git.

## Contributing

Pull requests are welcome.

Good contributions include:

- safety improvements;
- documentation improvements;
- tests;
- bug fixes;
- dataset tools;
- evaluation tools;
- usability improvements.

Please do not contribute features for memory reading, packet sniffing, injection, anti-cheat bypasses, or hidden application APIs.

## License

This project is licensed under the MIT License.

See [LICENSE](LICENSE).
