import sys
from pathlib import Path

import torch


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent
    repo_dir = project_dir / "d2l-zh-master"

    if repo_dir.exists():
        sys.path.insert(0, str(repo_dir))

    from d2l import torch as d2l

    print(f"project_dir={project_dir}")
    print(f"repo_exists={repo_dir.exists()}")
    print(f"repo_readme_exists={(repo_dir / 'README.md').exists()}")
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")

    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    print(f"x_shape={tuple(x.shape)}")
    print(f"d2l_reduce_sum={d2l.reduce_sum(x).item()}")


if __name__ == "__main__":
    main()
