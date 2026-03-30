# luau

**Latent Unified Adaptive Upskilling** — a research framework for accelerating reinforcement learning in novel domains via teacher-student transfer.

## Overview

LUAU investigates how a pretrained "teacher" agent can selectively guide a "student" agent learning a new task, improving sample efficiency without blindly imitating the teacher. The core insight is **introspection**: the student queries whether the teacher's knowledge is still relevant for the current state before deciding to follow its advice.

### Algorithms

- **Baseline SAC** — Soft Actor-Critic with torch.compile and vectorized envs, used as the transfer learning baseline.
- **Finetune** — SAC initialized from pretrained teacher weights; simple fine-tuning baseline.
- **IAA (Introspection-Aware Agent)** — SAC student that uses teacher actions when the teacher's Q-value confidence exceeds a threshold, with exponential decay on teacher reliance over training.
- **DIAA (Discriminative IAA)** — IAA variant that measures divergence between a frozen teacher Q-network and a trainable copy; uses teacher actions only when the divergence is low (i.e., the teacher's knowledge still applies).

### Environments

| Domain | Environment | Action Space |
|---|---|---|
| Grid world | MiniGrid Empty-5x5, FourRoomDoorKey | Discrete |
| Continuous control | BipedalWalker-v3 (hardcore), LunarLander-v3 (wind) | Continuous |
| Autonomous driving | CyCurveMerge-v0, MergeTurn-v0, TIntersection-v0 (MetaDrive) | Continuous |

### Key Files

```
luau/
├── sac_torchcompile.py     # Baseline SAC
├── sac_finetune.py         # Fine-tuning from pretrained weights
├── sac_iaa.py              # IAA (continuous)
├── sac_diaa.py             # DIAA (continuous)
├── ppo.py / iaa.py         # PPO + IAA (discrete, MiniGrid)
├── inference.py            # Run a trained policy for one episode
├── evaluate.py             # Compare teacher vs. student action distributions
├── driving-envs/           # MetaDrive variants of all the above
└── analysis/               # Learning curve plots, ablations, transfer efficacy
```

Experiments are tracked with [Weights & Biases](https://wandb.ai). Model checkpoints (actor.pt, qnet.pt) are saved as WandB artifacts.

### Transfer Efficacy Metric

Performance is measured by **Transfer Efficacy (TE)**:

```
TE = (AUC_method - AUC_baseline) / AUC_baseline
```

along with jumpstart (initial performance advantage) and asymptotic reward.

## TODO

### Easy
- [ ] Fix Figure 3 caption: "SAC" appears twice — one should read "Baseline"
- [ ] Add confidence intervals / shading to Figure 3 learning curves (4 seeds are already run)
- [ ] Explain the large MetaDrive jumpstart values in Table 2 (670%, 935%, 599%) — add a sentence in Results
- [ ] Explain why DIAA still outperforms SAC in BWHM even after the threshold collapses to zero (Discussion, p.8)
- [ ] Clean up boilerplate sections below (tooling, Getting Started, Contributing — leftover from project template)

### Medium
- [ ] Ablation: burn-in (δ) and decay (λ) parameters for DIAA — the Discussion already attributes CM's slow learning to burn-in being too long, so this is a known gap
- [ ] Ablation: coefficient learning rate (μ) — the one hyperparameter unique to DIAA's threshold update is never ablated
- [ ] Disentangle DIAA's two changes from IAA: (1) student Q-functions in the introspection criterion vs. (2) dynamic threshold — ideally with an ablation showing each change's individual contribution
- [ ] Add a 2D PCA action distribution plot for at least one environment to strengthen the interpretability contribution

### Hard
- [ ] Formalize the "threshold as task difficulty metric" contribution — correlate steady-state threshold value with an independent source-target similarity measure (e.g., behavioral distance or reward gap)
- [ ] Add a direct TGRL baseline on at least one task — TGRL is the primary algorithmic inspiration and reviewers will expect an empirical comparison
- [ ] Ablation: coefficient learning rate (μ) sensitivity across all five tasks

* [Poetry](https://python-poetry.org/)
    * For dependency management, packaging, and publishing
* [Ruff](https://github.com/astral-sh/ruff)
    * For linting/formatting (it's FAST)
* GitHub Actions
    * For CI/CD
* [Pytest](https://docs.pytest.org/en/8.2.x/)
    * For testing
* [pre-commit](https://pre-commit.com/)
    * For pre-commit hooks
* [PyInvoke](http://www.pyinvoke.org/)
    * For task running, because I hate `make`


## To get started
1. [Create a repository from a template](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-repository-from-a-template).
1. Clone the new repo
2. If poetry isn't installed, [you need to install it](https://python-poetry.org/docs/#installation).  
3. terminal `cd` into the project
3. run `poetry install`
3. Run `poetry run inv setup`
    

The setup will  
* Setup the poetry environment (or use the existing one you're activated to)
* Install the dependencies
* Setup the pre-commit hooks
* Ask you for some project details (name, author, etc) and update the pyproject.toml file

## Features
This sets up a basic set of checks to run.  If you already have a virtual environment setup for this project, you can skip all the `poetry run` parts of the command as long as that environment is active. Example `poetry run inv checks` would be `inv checks` if the environment is active. I won't be putting `poetry run` in front of every command, but if you don't have a virtual environment setup, you'll need to run `poetry run inv checks` instead of `inv checks`.

### Pre-commit
Pre-commit is used to run checks before you commit.  You can run `pre-commit run --all-files` to run all the checks.  The configuration for pre-commit is located in [.pre-commit-config.yaml](.pre-commit-config.yaml). If you find yourself unable to commit, this is almost certainly the reason. You need to install it for it to work on the client side. You can do this by running `pre-commit install`.

### Ruff
Ruff is used for linting and formatting. You can run 
`ruff check --fix` to check and fix the code. `ruff format` will only format the code.
The configuration for ruff is located in [ruff.toml](ruff.toml).

### Pytest
Pytest is used for testing.  You can run `pytest` to run all the tests. The CI is configured to run `pytest -m "not skipci"` so any test marked with `@pytest.mark.skipci` will not run in the CI pipeline. See [main_test.py](tests/main_test.py) for an example of how to use this.

### PyInvoke
PyInvoke is used for task running, and chosen because make is black magic to me.  You can run `inv --list` to see all the available tasks.  The tasks are located in the [tasks](tasks) folder.  The checks task will run all the checks. 

* `inv --list` will show you all the available tasks
* `inv checks` will run all the checks located in [checks.py](tasks/checks.py)

### GitHub
For pull requests, the pipeline will run `inv checks` and run all the formatting checks.  It will run the all the pytests, let you know what fails and succeeds in the pull request itself as well as give you a code coverage report.  The pipeline is located in [.github/workflows/ci.yml](.github/workflows/ci.yml).  

All pytests marked `@pytest.mark.skipci` will not run in the pipeline.  This is useful for tests that are slow, or require a specific environment to run.  You can run these tests locally, but they will not run in the pipeline.  You can see an example of this in [main_test.py](tests/main_test.py).

In order to get true coverage numbers in your report, the checks look for files in the src folder with a matching `_test.py` file in the `tests` folder.  If it doesn't have one, it creates a skeleton to just import.  
For example, [main.py](python_template/main.py) has a matching [main_test.py](tests/main_test.py) file.  

Theres also [issue templates](.github/ISSUE_TEMPLATE/bug_report.yml) and [rulesets](.github/rulesets/Require-Merge-Request.json) for the repository.  

## Contributing
If you have any suggestions, please open an issue.  If you'd like to contribute, please open a pull request.  I'm always looking for ways to improve this template. I'm open to suggestions, but I'm also very opinionated.  I'm trying to keep it as simple as possible while remaining good enough for production code.

## Updating from template
If you want to update your project from the template, or add the template to an existing project. 
There's a handy inv task. Just run `inv setup.update-from-template`.

or you can do it manually with the following commands

```bash
git remote add template https://github.com/lite-dsa/python-template.git
git fetch template
git merge template/main --allow-unrelated-histories
```

# Packages

```
pip install stable-baselines3[extra]
pip install swig
pip install gymnasium[box2d]
pip install minigrid
pip install ffio
pip install wandb
pip install scikit-image
pip install h5py
pip install seaborn
```

# Making a video from image frames

```
ffmpeg -framerate 25 -i frame_%06d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ../output.mp4
```
