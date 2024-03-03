# HPC Workshop

This is an example repository you can use to learn Purdue's HPC resources.

## Overarching Objective

[SLURM](https://slurm.schedmd.com/overview.html) is a cluster management utility that is used to send __jobs__ to the server. This job includes specifications about:
1. The actual code to run.
2. Constraints for:
    a. Walltime: the maximum runtime of the job
    b. Resources: specify GPU (generally any accelerator) count/type
    c. Account: 
    d. Node Type: 

After we meet the above specifications, our job is added to a queue. It is then run in weighted priority order, which is calculated based on your past usage, resources and walltime.

The goal of this workshop is to understand and utilize HPC via SLURM.

## Logging into Clusters

Start by logging in to `<username>@scholar.rcac.purdue.edu` / `<username>@queues.cs.purdue.edu` -- you can login with your career account password and Duo's 2FA.

You should now have access to a shell, that you can use to remotely run commands within the linux system.

Protips:
1. Append `ServerAliveInterval 60` to `~/.ssh/config` to prevent frozen sessions after long idle periods.
2. `curl https://raw.githubusercontent.com/dylanaraps/neofetch/master/neofetch | bash` runs neofetch!

### Adding your SSH Key

Using 2FA everytime gets annoying, especially if you log in and out often. Instead: create an [SSH Key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) that you can use to securely authenticate yourself into various clusters.

Once you create a keypair, __append the public key__ to `~/.ssh/authorized_keys`. If you add your SSH keys to GitHub, one easy way of doing this is to run `curl https://github.com/<username>.keys >> ~/.ssh/authorized_keys` in the cluster. From your local session, you can also run `ssh-copy-id <username>@<cluster-address>` and it should automataically update `~/.ssh/authorized_keys for you`.

## Interactive Jobs

Next, we want to configure an interactive job to explore our actual runtime environment to test the workflow before it runs non-interactively.

### Scholar

Scholar does this better. There's two ways to access an interactive runtime:
1. GPU endpoint: log in to `gpu.scholar.rcac.purdue.edu`
2. Run `sinteractive -A <account>`

### LWSN Cluster

You can access the list of queues available from the [CS servers homepage](https://www.cs.purdue.edu/resources/facilities/lwsnservers.html).

Once you decide on a queue, run:
```bash
srun --partition=<queue-name> --gres=gpu:<n> --pty /bin/bash -i
```

`--pty /bin/bash -i` runs bash interactively. It effectively is `sinteractive`.

## Creating a Runtime Environment

### Loading Modules
Once we have the compute, our next objective is to setup runtime dependencies. Let's start by checking if we have python:
```bash
jsetpal@scholar-fe06:~ $ python --version
Python 2.7.5

mc17 151 $ python --version
-bash: python: command not found
```

This is because dependencies are configurable using `module`:
```bash
module load cuda anaconda
module load cudnn  # only on scholar
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
(/home/jsetpal/.conda/envs/cent7/2020.11-py38/lint) jsetpal@scholar-fe06:~ $ python --version
Python 3.11.5
```

Optionally, install [uv](https://github.com/astral-sh/uv) to *drastically* reduce package installation time.

Finally, we can clone the repository and install package dependencies:
```bash
uv pip install -r requirements.txt  # if you installed uv
pip install -r requirements.txt     # if you didn't install uv
```

We are ready to begin the training run!

## Preparing the Run

The final step is to create a bash script that:
1. Lists SLURM constraints: Add `#SBATCH --<constraint>=<value>` lines.
2. Loads required modules and dependencies.
3. Carries out the training run.

You can find an example at `scripts/sbatch.sh`.

We can run this script using `sbatch scripts/sbatch.sh`!

## Monitoring the Job

You can check `squeue` for a status regarding your job.
```bash
(/home/jsetpal/.conda/envs/cent7/2020.11-py38/lint) jsetpal@scholar-fe06:~ $ squeue                                 
JOBID        USER      ACCOUNT      NAME             NODES   CPUS  TIME_LIMIT ST TIME
000001       jsetpal   gpu          sbatch.sh            1      1     4:00:00  R 0:10
000002       jsetpal   gpu          sbatch.sh            1      1     4:00:00 PD 0:00
```
Here, the first job is running, while the second is pending (awaiting free resources).

On scholar, you can check the available resources per account using `qlist`:
```bash
(/home/jsetpal/.conda/envs/cent7/2020.11-py38/lint) jsetpal@scholar-fe06:~/git/lint $ qlist 

                      Current Number of Cores                       Node
Account           Total    Queue     Run    Free    Max Walltime    Type
==============  =================================  ==============  ======
debug                32        0       0      32        00:30:00   A,B,G,H
gpu                 196        0      16     180        04:00:00     G,H
gpu-mig             128        0       0     128        04:00:00       H
long                128        0       0     128      3-00:00:00       A
scholar             576        2     208     368        04:00:00     A,B
```

You can use `srun` to log into your currently running jobs as well:
```bash
srun --jobid=000001 --pty /usr/bin/bash -i
```

You can cancel a queued or running job using `scancel <jobid>`.

# Some MLOPs Detail

Crucially, this is a non-interactive training setup. This means you can update constants, and rerun training setups without worrying about having your local machine on during these runs.

While it is possible to port forward Jupyter Notebooks and use them for training, it's not recommended practice because it **lacks reproduciblity**.

Instead, it is recommended to use **module-driven development**, using:
1. [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) for package-based structure for machine learning code.
2. [DVC](https://dvc.org/) for data versioning and pipelining.
3. [MLFlow](https://mlflow.org/) for experiment tracking.

# Fantastic Resources
1. [RCAC SLURM Guide](https://www.rcac.purdue.edu/knowledge/scholar/run)
2. [Cheatsheet](https://slurm.schedmd.com/pdfs/summary.pdf)
3. [SLURM Documentation](https://slurm.schedmd.com/documentation.html)
