# HPC Workshop

This is an example repository you can use to learn Purdue's HPC resources.

## Overarching Objective

[SLURM](https://slurm.schedmd.com/overview.html) is a cluster management utility that is used to send __jobs__ to the server. This job includes specifications about:
1. The actual code to run.
2. Constraints for: \
    a. Walltime: the maximum runtime of the job \
    b. Resources: specify GPU (generally any accelerator) count/type \
    c. Account: the queue under which the job should run \
    d. Node Type: the node under which the job should run \
    e. + a ton more; [RTFM](https://wiki.rc.usf.edu/index.php/SLURM_Using_Features_and_Constraints)
You can use `sfeatures` to list out the nodelist, CPUs, Memory, Available Features (resource type) as well as n-GPUs you can request at a time.

After we meet the above specifications, our job is added to a queue. It is then run in weighted priority order, which is calculated based on your past usage, resources and walltime.

The goal of this workshop is to understand and utilize HPC via SLURM.

## Logging into Clusters

Start by logging in to `<username>@gilbreth.rcac.purdue.edu` -- you can login with your career account password and Duo's 2FA.

You should now have access to a shell, that you can use to remotely run commands within the linux system.

Protips:
1. Append `ServerAliveInterval 60` to `~/.ssh/config` to prevent frozen sessions after long idle periods.
2. `curl https://raw.githubusercontent.com/dylanaraps/neofetch/master/neofetch | bash` runs neofetch!

## Configuration

### Adding your SSH Key

Using 2FA everytime gets annoying, especially if you log in and out often. Instead: create an [SSH Key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) that you can use to securely authenticate yourself into various clusters.

Once you create a keypair, __append the public key__ to `~/.ssh/authorized_keys`. If you add your SSH keys to GitHub, one easy way of doing this is to run `curl https://github.com/<username>.keys >> ~/.ssh/authorized_keys` in the cluster. From your local session, you can also run `ssh-copy-id <username>@<cluster-address>` and it should automatically update `~/.ssh/authorized_keys for you`.

### Redirecting Cache

The home directory of your gilbreth account is limited to [25GB](https://www.rcac.purdue.edu/knowledge/faqs/ncdu). Things like your cache, datasets and large models can significantly use up this storage.

However, there is exists scratch space at `/scratch/gilbreth/<username>`. This storage may be deleted at any time, but has a significantly larger limit than your home directory. We setup the following split:
1. Code: `$HOME/path/to/repo/`
2. Data, Model, Cache: `/scratch/gilbreth/username/`

We can do this by updating `$XDG_CACHE_HOME`. In your `~/.bashrc`, add:
```bash
export XDG_CACHE_HOME=/scratch/gilbreth/<username>/
```
Replacing `<username>` with your alias. Then, run `source ~/.bashrc`. Your cache is now under scratch!

You can run `mv ~/.cache/* /scratch/gilbreth/<username>`, or `rm -rf ~/.cache` to relinquish your storage. `ncdu` can be used to show you where your storage quota is being used.

### IPython Override

For some reason, the incorrect instance of IPython (incredible debugging tool) is run, which does not contain utilize the active environment.

However, running `python -m IPython` works, so we can override this behavior by creating a wrapper in `~/.local/bin/`:
```bash
#!/usr/bin/bash

python -m IPython
```
Name this `ipython`, mark it as executable with `chmod +x ipython`, and ensure that `~/.local/bin` is present under `$PATH` (you can add `export PATH=~/.local/bin:$PATH` to `~/.bashrc` to ensure this).

## Interactive Jobs

Next, we want to configure an interactive job to explore our actual runtime environment to test the workflow before it runs non-interactively.

Run `sinteractive -A <account> -N<number of nodes> -n<number of tasks> -c<cpus/task> -G<number of GPUs> -t <job walltime>`

Run `nvidia-smi` to check out your new A100!

## Creating a Runtime Environment

### Loading Data

For our speific task, we can load the data in two ways:
1. If you are on an RCAC cluster: `scripts/prepare.sh` will link the directory from the MLP data depot directly.
2. Otherwise, you can use `scripts/download.sh`. It will pull the dataset from our remote and set it up, this might take a little longer and storage will count against your quota.

### Loading Modules
Once we have the compute, our next objective is to setup runtime dependencies. Let's start by checking if we have python:
```bash
jsetpal@gilbreth-fe02:~ $ python --version
Python 2.7.5

mc17 151 $ python --version
-bash: python: command not found
```

This is because dependencies are configurable using `module`:
```bash
module load cuda cudnn anaconda
```
Should be all that you need for the current experiment. You can :
1. Find the loaded modules with `module list`
2. Reset modules using `module purge`
3. List all available module using `module av`
4. Obtain an extended description using `module spider <name/version>`

### Setting up Python Dependencies

You can install an updated python instance using conda.
```bash
conda create -n <name> python=<version>
conda activate <name>
(/home/jsetpal/.conda/envs/cent7/2020.11-py38/llm-finetuning) jsetpal@gilbreth-fe02:~ $ python --version
Python 3.11.5
```

Optionally, install [uv](https://github.com/astral-sh/uv) to *drastically* reduce package installation time.

Finally, we can clone the repository and install package dependencies:
```bash
uv pip install -r requirements.txt  # if you installed uv (you are cool :D)
pip install -r requirements.txt     # if you didn't install uv (you are still cool)
```

We are ready to begin the training run!

## Preparing the Run

The final step is to create a bash script that:
1. Lists SLURM constraints: Add `#SBATCH --<constraint>=<value>` lines.
2. Loads required modules and dependencies.
3. Carries out the training run.

You can find an example at `scripts/sbatch.sh`. An example of a script that works with LWSN servers is also made available at `scripts/sbatch-lwsnqueues.sh`.

We can run this script using `sbatch scripts/sbatch.sh`!

Note: you may need to update the `cd` path to the relevant project root.

## Monitoring the Job

You can check `squeue` for a status regarding your job.
```bash
(/home/jsetpal/.conda/envs/cent7/2020.11-py38/llm-finetuning) jsetpal@gilbreth-fe02:~ $ squeue -u jsetpal
JOBID        USER      ACCOUNT      NAME             NODES   CPUS  TIME_LIMIT ST TIME
000001       jsetpal   gpu          sbatch.sh            1      1     4:00:00  R 0:10
000002       jsetpal   gpu          sbatch.sh            1      1     4:00:00 PD 0:00
```
Here, the first job is running, while the second is pending (awaiting free resources).

On gilbreth, you can check the available resources per account using `qlist`:
```bash
jsetpal@gilbreth-fe02:~/.local/bin $ qlist 

                      Current Number of GPUs                        Node
Account           Total    Queue     Run    Free    Max Walltime    Type
==============  =================================  ==============  ======
debug               182        1       4     178        00:30:00   B,D,E,F,G,H,I
mlp-n                 1        0       0       1     14-00:00:00       N
standby             230     1038      94      72        04:00:00   B,D,E,F,G,H,I,K
```

You can use `srun` to log into your currently running jobs as well:
```bash
srun --jobid=000001 --pty /usr/bin/bash -i
```

You can cancel a queued or running job using `scancel <jobid>`.

## Multi-GPU Runs

Using multiple GPUs can speed up training significantly, and the example code in `src/model/train.py` shows the setup process for DDP, including logging and syncing metrics across devices.

## Checkpointing

To ensure training runs that are terminated early don't end up being wasted, checkpointing is very critical to any long-running job. Usually SLURM is very good with preventing resource conflicts that can trigger random crashes, however this is helpful to guardrail your own code.

## Timed / `KeyBoardInterrupt`ed Early Stopping

Since SLURM jobs are timed, it is crucial to ensure that runs end before the alloted time. A simple way to guardrail against that is using a timed shutdown of training, which we can also double with `KeyBoardInterrupt` to manually trigger a clean training exit.

# Some MLOps Detail

Crucially, this is a non-interactive training setup. This means you can update constants, and rerun training setups without worrying about having your local machine on during these runs.

While it is possible to port forward Jupyter Notebooks and use them for training, it's not recommended practice because it **lacks reproduciblity**.

Instead, it is recommended to use **module-driven development**, using:
1. [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) for package-based structure for machine learning code.
2. [MLFlow](https://mlflow.org/) / [WandB](https://wandb.ai/) for experiment tracking.

# Fantastic Resources
1. [RCAC SLURM Guide](https://www.rcac.purdue.edu/knowledge/gilbreth/)
2. [Cheatsheet](https://slurm.schedmd.com/pdfs/summary.pdf)
3. [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
