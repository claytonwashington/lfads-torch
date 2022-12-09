import logging
import shutil
from datetime import datetime
from pathlib import Path

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from lfads_torch.run_model import run_model

logger = logging.getLogger(__name__)

# ---------- OPTIONS -----------
LOCAL_MODE = False
OVERWRITE = True
RUN_TAG = datetime.now().strftime("%Y%m%d-%H%M%S")
RUNS_HOME = Path("/snel/share/runs/lfads-torch/validation")
RUN_DIR = RUNS_HOME / "multi" / RUN_TAG
CONFIG_PATH = Path("../configs/multi.yaml")
# ------------------------------

# Initialize the `ray` server in local mode if necessary
if LOCAL_MODE:
    ray.init(local_mode=True)
# Overwrite the directory if necessary
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Run the hyperparameter search
tune.run(
    tune.with_parameters(run_model, config_path=CONFIG_PATH),
    metric="valid/recon_smth",
    mode="min",
    name=RUN_DIR.name,
    config=dict(
        model=dict(
            dropout_rate=tune.uniform(0.3, 0.4),
            kl_co_scale=tune.loguniform(1e-8, 1e-6),
            kl_ic_scale=tune.loguniform(1e-8, 1e-6),
            weight_decay=1.5e-3,
            ic_enc_dim=tune.randint(50, 100),
            ci_enc_dim=tune.randint(60, 100),
            con_dim=tune.randint(60, 100),
            co_dim=tune.randint(1, 5),
            ic_dim=tune.randint(50, 150),
            gen_dim=tune.randint(160, 200),
            fac_dim=tune.randint(60, 100),
        ),
    ),
    resources_per_trial=dict(cpu=2, gpu=0.3),
    num_samples=90,
    local_dir=RUN_DIR.parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=FIFOScheduler(),
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=["valid/recon_smth", "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)
