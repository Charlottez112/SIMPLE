# Meta-Learning Conservation Laws for Fast Molecular Dynamics Simulation

## Setup in GreatLakes

`setup.sh` will download the Physics 101 dataset, install conda, create a new conda environment named `noether`, and install dependencies.

```console
$ mkdir workspace; cd workspace
$ git clone git@github.com:Charlottez112/eecs545_project.git
$ bash misc/setup.sh
```

## Submit GPU Jobs

See `job-sample.sh` for a reference job submission that works. Note that the sample limits the runtime of the job to 7 minutes (guessing that a shorter runtime job will be preferred by the scheduler). 7 minutes is enough to complete one epoch with noether networks and will be a good time for debugging. When submitting actual jobs, make sure to increase the job duration.

```console
$ sbatch misc/job-sample.sh
```
