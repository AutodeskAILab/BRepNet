from pathlib import Path

def find_checkpoint(logs_dir, month_day, hour_min_second):
    logs_dir = Path(logs_dir)
    checkpoint_path = logs_dir / month_day / hour_min_second / "checkpoints"
    checkpoint_files = [ f for f in checkpoint_path.glob("*.ckpt")]
    min_loss_checkpoints = []
    for file in checkpoint_files:
        # Check if this is a tmp_end file
        if not "tmp_end" in file.stem:
            min_loss_checkpoints.append(file)
    assert len(min_loss_checkpoints) > 0

    # Always return the last checkpoint we found
    return min_loss_checkpoints[-1]
