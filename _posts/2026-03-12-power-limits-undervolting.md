---
title: "GPU Power Limits and Undervolting on Linux"
---

# Undervolting on Linux

This guide is based on [this excellent post](https://shelbyjenkins.github.io/blog/power-limit-nvidia-linux/) by Shelby Jenkins, extended with clock locking and V/F curve offsets for undervolting.

## Why bother?

I have 3 3090s, and I've had them set to 280W basically as long as I've had them (I tried a bunch of different wattages, and that's what I found was pareto optimal for my setup), and was happy with that. BUT! Ever since I moved a few months ago, I've been experiencing crashes (system reboots) when trying to use them all at the exact same time: torch.compile across all three gpus for DDP or vLLM warmups (which also torch.compiles), that kind of thing. Staggering compilation and vLLM start times were fine, and sustained 280W loads were fine. 3090s are well known for using a ton of power instantaneously, and today, I decided I'd had enough: it was time to try to fix this. 

Shelby's fantastic post shows us how to cap the *average* wattage that our gpus use, not going above that wattage. The average part is important: apparently the regulation is over a certain time interval, so instantaneously, GPUs can pull much more power than we set our limit to. 
Enter undervolting. 
I'm told that for undervolting, you "shift the voltage-frequency curve so the GPU runs at a lower voltage for the same clock speed" (from Claude). 
So, at a certain voltage, we can get the same performance. As we know, watts = volts * amps, so we can use fewer watts for the same clock speed. To keep my system from crashing, it turns out that I had to lock my clocks to 1850: anything above that would still crash my machine. This tells me that the problem was probably really a voltage issue (fact check me on this, I'm not sure at all!). 

Remember that it's super important to try a quick training run and monitor the loss and or gradient norms. After I got my setup stable, I had to lower the offset to make sure that I wasn't getting NaNs. [Apparently this is because not enough voltage can make transistors not switch cleanly](https://ieeexplore.ieee.org/document/7401681)

I would've been happy to just have my system not crash every time I try to use them all at once. But it turns out that I'm actually getting a little more performance!

Here's what my results look like on my 3090s:

|               | Previous (280W power limit)| Final (undervolted) |
|---------------|----------------------------|---------------------|
| Power limit   | 280W                       | 250W                |
| Clock         | unmanaged (boost to 2100)  | locked 210-1850 MHz |
| V/F offset    | 0                          | +175 MHz            |
| FP16 TFLOPS   | ~59.8                      | ~62                 |
| Stability     | crashes during vLLM warmup | stable              |


## The script

This script handles three things:
1. **Power limits** — caps each GPU's power draw via `nvidia-smi`
2. **Clock locking** — locks the GPU clock to a range, preventing it from boosting into high-voltage frequency bins
3. **V/F curve offset** — shifts the voltage-frequency curve via `pynvml`, so your target clock runs at the voltage normally used for a lower clock (this is the actual undervolting)

Place it at `/usr/local/sbin/nv-power-limit.sh`:

```bash
#!/usr/bin/env bash
# Persistent NVIDIA GPU power limits, clock locking, and undervolting.
# Intended to be run at boot via systemd.
#
# Based on: https://shelbyjenkins.github.io/blog/power-limit-nvidia-linux/
# Extended with clock locking and V/F curve offsets for undervolting.
#
# HOW TO CONFIGURE:
#   1. Set gpu_enabled — 1 for each GPU you want to manage, 0 to skip
#   2. Set gpu_power_limits — desired power limit in watts per GPU
#   3. Set LOCK_CLOCK_MIN/MAX — GPU clock range (prevents high-voltage boost bins)
#   4. Set CLOCK_OFFSET — V/F curve shift in MHz (higher = more undervolt)
#      e.g. +175 means 1850 MHz runs at the voltage normally used for ~1675 MHz
#      Note that the values below are what worked for me on my machine to solve the problem I was having, your mileage will vary.
#
set -euo pipefail

command -v nvidia-smi &>/dev/null || { echo >&2 "nvidia-smi not found, exiting."; exit 1; }

# ── Configuration ────────────────────────────────────────────────────
# Enable/disable per GPU (0-indexed)
declare -A gpu_enabled=(
    [0]=1
    [1]=1
    [2]=1
)

# Power limits in watts per GPU
declare -A gpu_power_limits=(
    [0]=250
    [1]=250
    [2]=250
)

# Clock locking range in MHz (0 to disable)
LOCK_CLOCK_MIN=210
LOCK_CLOCK_MAX=1850

# V/F curve clock offset in MHz (0 to disable)
# Shifts the voltage-frequency curve so your locked clock runs at lower voltage.
CLOCK_OFFSET=175
# ─────────────────────────────────────────────────────────────────────

echo "=== NVIDIA GPU Power & Undervolt Setup ==="

# Step 1: Set power limits
for gpu_id in "${!gpu_enabled[@]}"; do
    if [[ ${gpu_enabled[$gpu_id]} -ne 1 ]]; then
        echo "GPU $gpu_id: skipped (disabled)"
        continue
    fi

    /usr/bin/nvidia-smi -i "$gpu_id" --persistence-mode=1

    max_pl=$(/usr/bin/nvidia-smi -i "$gpu_id" -q -d POWER | grep 'Max Power Limit' | awk '{print $5}' | grep -oE '[0-9]+([.][0-9]+)?')
    if [[ -z "$max_pl" || "$max_pl" == "N/A" ]]; then
        echo "GPU $gpu_id: could not read max power limit, skipping"
        continue
    fi

    desired=${gpu_power_limits[$gpu_id]}
    if [[ "$desired" -le $(printf "%.0f" "$max_pl") ]]; then
        echo "GPU $gpu_id: power limit -> ${desired}W"
        /usr/bin/nvidia-smi -i "$gpu_id" --power-limit="$desired"
    else
        echo "GPU $gpu_id: ERROR desired ${desired}W exceeds max ${max_pl}W"
    fi
done

# Step 2: Lock GPU clocks (prevents boosting to high-voltage frequency bins)
if [[ "$LOCK_CLOCK_MAX" -gt 0 ]]; then
    for gpu_id in "${!gpu_enabled[@]}"; do
        if [[ ${gpu_enabled[$gpu_id]} -eq 1 ]]; then
            echo "GPU $gpu_id: clock lock -> ${LOCK_CLOCK_MIN}-${LOCK_CLOCK_MAX} MHz"
            /usr/bin/nvidia-smi -i "$gpu_id" -lgc "$LOCK_CLOCK_MIN","$LOCK_CLOCK_MAX"
        fi
    done
fi

# Step 3: Apply V/F curve clock offset via NVML
# This requires pynvml (pip install nvidia-ml-py) accessible to root.
# Adjust the PYTHONPATH below if pynvml is installed in a user site-packages.
if [[ "$CLOCK_OFFSET" -gt 0 ]]; then
    enabled_ids=()
    for gpu_id in "${!gpu_enabled[@]}"; do
        if [[ ${gpu_enabled[$gpu_id]} -eq 1 ]]; then
            enabled_ids+=("$gpu_id")
        fi
    done

    # Try to find pynvml — check common locations
    PYNVML_PATHS=(
        "/usr/lib/python3/dist-packages"
        "/usr/local/lib/python3/dist-packages"
    )
    # Also check all user site-packages
    for d in /home/*/.[Ll]ocal/lib/python3*/site-packages; do
        [[ -d "$d" ]] && PYNVML_PATHS+=("$d")
    done
    EXTRA_PATH=$(IFS=:; echo "${PYNVML_PATHS[*]}")

    gpu_count=${#enabled_ids[@]}
    PYTHONPATH="$EXTRA_PATH:${PYTHONPATH:-}" /usr/bin/python3 -c "
import pynvml, sys
pynvml.nvmlInit()
offset = $CLOCK_OFFSET
gpu_ids = [${enabled_ids[*]// /,}]
for i in gpu_ids:
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    pynvml.nvmlDeviceSetGpcClkVfOffset(handle, offset)
    actual = pynvml.nvmlDeviceGetGpcClkVfOffset(handle)
    print(f'GPU {i}: clock offset -> +{actual} MHz')
pynvml.nvmlShutdown()
" || echo "WARNING: clock offset failed (is pynvml installed? pip install nvidia-ml-py)"
fi

echo "=== Done ==="
exit 0
```

Make it executable:

```bash
sudo chmod 744 /usr/local/sbin/nv-power-limit.sh
```

## Systemd service

Create `/usr/local/etc/systemd/nv-power-limit.service`:

```ini
[Unit]
Description=NVIDIA GPU power limit, clock lock, and undervolt
After=syslog.target systemd-modules-load.service
ConditionPathExists=/usr/bin/nvidia-smi

[Service]
User=root
ExecStart=/usr/local/sbin/nv-power-limit.sh

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo chmod 644 /usr/local/etc/systemd/nv-power-limit.service
sudo ln -s /usr/local/etc/systemd/nv-power-limit.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start nv-power-limit.service
sudo systemctl status nv-power-limit.service
# If it looks good:
sudo systemctl enable nv-power-limit.service
```

## Finding stable values
The values in the script above (250W power limit, 1850 MHz max clock, +175 MHz offset) are specific to my setup. I'd imagine they're a pretty good place to start, as (I think?) they're pretty conservative, but you'll need to find your own stable values.
You might just have [your](https://claude.com/product/claude-code) [favorite](https://openai.com/codex/) [coding](https://opencode.ai/) [agent](https://mistral.ai/products/vibe) write up a script for you to try little increments, logging each combination of these until you crash (DM or email me if you'd like mine). It seems reasonable to me to pick a wattage and go from there, but they're your GPUs. 

Start conservative and work your way up:
1. Set a power limit first and run your workload. Check temps and performance.
2. Lock clocks to your GPU's typical boost clock (check with `nvidia-smi dmon -s c`).
3. Add clock offset in small increments (+50 MHz at a time). Run a stress test at each step (eg vLLM warmup/serving).

If the system crashes under load, back off the offset. If it crashes at idle, your minimum clock might be too low for the offset you're applying.

## Modifying settings

After editing the script:

```bash
sudo systemctl daemon-reload
sudo systemctl restart nv-power-limit.service
sudo systemctl status nv-power-limit.service
```
