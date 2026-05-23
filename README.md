# Desktop AI Trainer

Desktop AI Trainer is a local educational desktop-agent project. The program is designed for a user to record their own computer actions, build training data from those demonstrations, train an imitation-learning model, and later run an AI agent that chooses keyboard and mouse actions from screen images.

This is not a bot for a specific game or application. The project is a general learning scaffold for desktop automation research and experimentation.

## Safety Boundaries

Desktop AI Trainer uses only normal screen capture and keyboard/mouse input. It must not use:

- process memory reading;
- packet sniffing or network traffic analysis;
- code injection;
- anti-cheat or protection bypasses;
- hidden game or application APIs.

Live-control mode can be dangerous because it sends real input to the desktop. Always validate behavior in dry-run mode first, inspect the action logs, and only then consider enabling live control in a safe, reversible environment.

Planned and configured safety controls include:

- dry-run mode;
- emergency stop hotkey;
- pause hotkey;
- allowlist window title guard;
- action rate limit;
- confidence threshold;
- full action logging.

## Profiles

A profile is a YAML file in `configs/profiles/` that describes how the agent is allowed to interact with a specific desktop task. Profiles are intentionally generic: they are not game-client integrations and they do not add memory reading, packet sniffing, injection, hidden APIs, or protection bypasses.

Each profile defines:

- `profile_name`;
- `allowed_window_title`;
- `capture_region`;
- `fps`;
- `action_delay`;
- `confidence_threshold`;
- `emergency_stop_hotkey`;
- `pause_hotkey`;
- `actions`.

The `actions` list is the action space. It is the complete list of actions the model may learn to predict for that profile. Keep it small, explicit, and task-specific. Always include `no_op` with `type: none` so the agent can safely decide to do nothing.

Example action:

```yaml
- name: press_f1
  type: key
  key: f1
```

To create a new profile, copy `configs/profiles/default.yaml` to a new file, change `profile_name`, set the target `allowed_window_title`, adjust the capture region and safety settings, then replace `actions` with only the actions you want the agent to use.

Restricting the action space matters because it limits the blast radius of model mistakes, makes demonstrations easier to label, improves training signal, and keeps dry-run review readable. If an action is not in the profile, the agent should not be able to choose it.

## Install

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```powershell
python -m src.main
```

Expected output:

```text
Desktop AI Trainer is ready.
```

Inspect a profile action space:

```powershell
python -m src.input_control.action_space --profile lineage2_private
```

## Record Demonstrations

Demonstration recording captures:

- PNG screen frames from the profile `capture_region`;
- allowed keyboard and mouse events that exist in the profile `actions`;
- timestamps and frame ids for each recorded action;
- episode metadata in JSON.

Start a recording:

```powershell
python -m src.recorder.record_demo --profile lineage2_private --episode demo_001
```

After running the command, switch focus to the target window during the 5 second countdown. The recorder checks `allowed_window_title` after the countdown, so the target window must be active at that moment. If you run the command inside VS Code, VS Code is usually the active window until you switch away.

For `lineage2_private`, the expected window title is currently:

```text
Lineage II
```

When recording starts, a small status overlay appears near the right side of the capture area. It shows that screen recording is active, the current episode, frame/action counters, and the hotkeys for stop and pause/resume.

The recorder writes:

```text
data/demos/{profile}/{episode}/frames/
data/demos/{profile}/{episode}/actions.csv
data/demos/{profile}/{episode}/metadata.json
```

Before recording, close password fields, private chats, personal documents, browser tabs with sensitive information, and anything else that should not appear in screenshots. The recorder ignores actions that are not listed in the selected profile, but it still captures the configured screen region as images.

The active window title must match the profile `allowed_window_title`. Recording starts after a 5 second countdown.

Stop or pause recording:

```text
Ctrl+Shift+Q  emergency stop and save the episode
Ctrl+Shift+P  pause or resume recording
```

## Import Video

Video import extracts frames from an existing video file and stores them like a demo episode. It does not record keyboard or mouse input, and it does not infer user actions from the video.

Import frames from a video:

```powershell
python -m src.video.import_video --profile lineage2_private --video-path data/videos/demo.mp4 --every-n-frames 5
```

The importer writes:

```text
data/demos/{profile}/video_import_{timestamp}/frames/
data/demos/{profile}/video_import_{timestamp}/actions.csv
data/demos/{profile}/video_import_{timestamp}/metadata.json
```

Imported video rows start with:

```text
action_name=unknown
action_id=-1
```

This is intentional. A video by itself is not a full demonstration because it does not contain the user's input events. Use video import when you have useful visual states, want to bootstrap frame collection, or want to manually label a small set of examples.

Manually label an imported video episode:

```powershell
python -m src.dataset.manual_label_tool --profile lineage2_private --episode video_import_YYYYMMDD_HHMMSS
```

The tool prints each frame path, shows the available actions from the profile, and asks for an `action_id`. Enter `u` to keep a frame as `unknown`, press Enter to keep the current label, or enter the action id such as `0` for `no_op`.

Before training, inspect or manually label video imports. Unknown rows are not real action demonstrations; if you build a dataset without useful labels, many frames may effectively become `no_op`, which can train a passive model.

## Build Dataset

After recording one or more episodes, build a supervised dataset:

```powershell
python -m src.dataset.build_dataset --profile lineage2_private
```

The builder reads:

```text
data/demos/{profile}/{episode}/frames/
data/demos/{profile}/{episode}/actions.csv
data/demos/{profile}/{episode}/metadata.json
```

It writes:

```text
data/datasets/{profile}/dataset.csv
```

`dataset.csv` contains `frame_path`, `action_id`, `action_name`, `timestamp`, `episode`, and `split`. Splits are assigned as `train`, `val`, and `test`.

The builder matches each frame to an allowed action from the profile. If no action is close enough to that frame, the frame is labeled `no_op`.

`no_op` is often the largest class because most screen frames happen between deliberate user actions. This is normal, but too much `no_op` can teach the model to do nothing. The builder prints class distribution and warns when `no_op` dominates. It does not delete or rebalance data automatically.

Create a dataset preview:

```powershell
python -m src.dataset.visualize_dataset --profile lineage2_private --samples 20
```

The preview is saved to:

```text
outputs/debug/{profile}_dataset_samples.png
```

Use the preview to check whether frame labels make sense. If actions are poorly matched to frames, inspect `actions.csv`, verify frame ids and timestamps, record shorter episodes with clearer actions, reduce idle time, or adjust the recorder timing before training.

## Train Imitation Policy

Train a CNN policy from `data/datasets/{profile}/dataset.csv`:

```powershell
python -m src.training.train_imitation --profile lineage2_private --epochs 10
```

Training reads screenshots, resizes them to a small square image, and learns to predict the profile `action_id` with supervised imitation learning. It does not control the computer.

Training writes:

```text
models/{profile}_imitation_policy.pt
outputs/reports/{profile}_training_metrics.csv
outputs/reports/{profile}_training_curves.png
```

The trainer prints class distribution and uses class weights to reduce action imbalance. This matters because desktop datasets often contain many more `no_op` frames than active actions.

Evaluate the saved model:

```powershell
python -m src.training.evaluate_imitation --profile lineage2_private
```

Evaluation writes:

```text
outputs/reports/{profile}_evaluation_report.txt
```

Accuracy means the percentage of test frames where the top predicted action equals the dataset label. High accuracy can be misleading when `no_op` dominates: a model that predicts `no_op` almost everywhere can look good while being useless. Always inspect class distribution, top-3 accuracy, prediction distribution, and the confusion matrix.

The confusion matrix shows which true actions are being confused with which predicted actions. It is the fastest way to catch cases like `press_f1` being learned as `no_op`, directional actions being mixed up, or a class never being predicted.

## Dry-Run Inference

Dry-run mode lets the trained model watch the active screen region and predict actions without controlling the computer. It does not press keys and does not move or click the mouse. It only prints predictions and writes a CSV log.

Run dry-run:

```powershell
python -m src.inference.dry_run_agent --profile lineage2_private
```

The dry-run log is written to:

```text
outputs/logs/{profile}_dry_run_predictions.csv
```

Each row includes the raw predicted action, confidence, effective action after safety rules, top-3 actions, active window title, and whether the prediction was forced to `no_op`.

Safety behavior:

- if confidence is below `confidence_threshold`, effective action is forced to `no_op`;
- if active window title does not match `allowed_window_title`, effective action is forced to `no_op`;
- mismatches and forced `no_op` decisions are logged for review;
- `Ctrl+Shift+P` pauses or resumes dry-run;
- `Ctrl+Shift+Q` stops dry-run.

Do not jump from training directly to live-control. First, dry-run the model and inspect whether predictions are stable, aligned with what you would do, and not dominated by `no_op` or a single action. A model is behaving badly if confidence is low, top-3 predictions change randomly, the active action is usually wrong, the model predicts actions on the wrong window, or the CSV shows repeated forced `no_op` decisions.

## Analyze Dry-Run

Before live-control, analyze the dry-run prediction log:

```powershell
python -m src.analysis.analyze_predictions --profile lineage2_private
python -m src.analysis.plot_action_distribution --profile lineage2_private
```

These tools read:

```text
outputs/logs/{profile}_dry_run_predictions.csv
```

They write:

```text
outputs/reports/{profile}_prediction_analysis.txt
outputs/reports/{profile}_action_distribution.png
```

The analysis report includes:

- percentage of `no_op`;
- average confidence;
- low confidence rate;
- most frequent actions;
- action switches per minute.

Bad model signs:

- `no_op` is more than 95%;
- one action is more than 90%;
- average confidence is below `confidence_threshold`;
- actions switch too frequently;
- `forced_no_op` appears constantly in the dry-run CSV;
- predictions change randomly while the screen is stable.

Do not run live-control when any warning appears in `{profile}_prediction_analysis.txt`. Return to dataset preview, evaluation, more demonstrations, manual labels, or a higher `confidence_threshold` first.

## Live-Control

Live-control sends real keyboard and mouse input to your computer. Use it only after recording demonstrations, building the dataset, training/evaluating the model, and running dry-run long enough to trust the predictions.

Start live-control:

```powershell
python -m src.inference.run_agent --profile lineage2_private
```

The agent prints a warning and waits through a countdown before it can act. It writes a complete log to:

```text
outputs/logs/{profile}_live_actions.csv
```

Every row includes timestamp, predicted action, executed action, confidence, active window, safety state, top-3 predictions, and execution result.

Safety states:

- `OK`: active window matches, confidence is high enough, action may execute;
- `LOW_CONFIDENCE_NO_OP`: confidence is below threshold, only `no_op` executes;
- `WRONG_WINDOW_NO_OP`: active window does not match, only `no_op` executes;
- `PAUSED_NO_OP`: pause mode is active, only `no_op` executes;
- `EMERGENCY_STOP`: emergency stop was triggered.

Controls:

```text
Ctrl+Shift+Q  emergency stop
Ctrl+Shift+P  pause or resume
```

If the agent presses the wrong keys, hit `Ctrl+Shift+Q` immediately. Then inspect `outputs/logs/{profile}_live_actions.csv`, remove risky actions from the profile, raise `confidence_threshold`, increase `action_delay`, and return to dry-run before trying live-control again.

Raise confidence threshold in `configs/profiles/{profile}.yaml`:

```yaml
confidence_threshold: 0.95
```

Reduce action rate by increasing `action_delay`:

```yaml
action_delay: 0.5
```

Mouse clicks only work if the mouse action is explicitly listed in the profile. Actions that are not in the profile action space are rejected.

## Tests

```powershell
python -m pytest
```

In an activated virtual environment where the pytest executable is on PATH, `pytest` is equivalent.

## Project Flow

1. Record user demonstrations with screen frames and synchronized keyboard/mouse actions.
2. Build datasets from recorded demonstrations.
3. Train a policy model with imitation learning.
4. Run the trained agent in dry-run mode and inspect logs.
5. Only after dry-run validation, run live control with all safety controls enabled.
