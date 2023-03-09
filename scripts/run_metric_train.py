import fire
import os
import gin

from stmetric.model import run_metric_learning_ptl
from stmetric.utils import gin_register_and_parse

@gin.configurable
def run_train(gin_file: str = "gin/sol_metric.gin"):
    gin_config_path = os.path.join(os.getcwd(), gin_file)
    gin_register_and_parse(gin_config_path)

    run_metric_learning_ptl()

def main():
  fire.Fire(run_train)

if __name__ == "__main__":
    main()