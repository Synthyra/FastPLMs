import sys
from typing import List
import hydra
import omegaconf


from boltzgen_flat.task_task import Task


def main(config: str, args: List) -> None:
    """
    This is just a wrapper for running the .run() function of our `Task` class.
    If you run the pipeline (for example via `boltzgen run design_spec.yaml ...`) then this function reads the yaml files of the individual pipeline steps and executes the pipeline steps.

    The possible tasks are:
        - Train (GPU: BoltzGen diffusion model or inverse folding model training)
        - Predict (GPU: Running BoltzGen diffusion, inverse folding, refolding, designfolding, or affinity prediction)
        - Analyze (CPU: Compute CPU Metrics and aggregate metrics from GPU steps)
        - Filter (CPU: Very fast (20s) computes ranking and writes final output files)

    The files for these are:
        - src/boltzgen/task/train/train.py
        - src/boltzgen/task/predict/predict.py
        - src/boltzgen/task/analyze/analyze.py
        - src/boltzgen/task/filter/filter.py

    Parameters
    ----------
    config : str
        Path to the configuration yaml file. The yaml file contains something like `_target_: boltzgen.task.predict.predict.Predict` at the beginning which tells it which Task class to run
    args : List
        List of arguments to override the configuration.
    """
    # Load the configuration
    args = omegaconf.OmegaConf.from_dotlist(args)
    config = omegaconf.OmegaConf.load(config)
    config = omegaconf.OmegaConf.merge(config, args)

    # Instantiate the task
    task = hydra.utils.instantiate(config)

    if not isinstance(task, Task):
        msg = "Config must be an instance of Task."
        raise TypeError(msg)

    # Run the task
    task.run(config)


if __name__ == "__main__":
    config = sys.argv[1]
    args = sys.argv[2:]
    main(config, args)
