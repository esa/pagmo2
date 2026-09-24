---
description: "Use when managing, creating and using virtual environments, conda environments, package installation, dependency changes in this workspace. Activate the repository's project environment before running install, build, test, or Python commands."
---

- Use conda package manager whenever possible.
- Only add packages to the environment if they are necessary for the project.
- Check that added dependencies are actually used in the project.
- If a project-specific conda environment already exists for this repository, prefer using it.
- Do not assume the environment name is always pagmo2_devel; check the repository configuration or ask the user if needed.
- Do not hard-code a personal path such as a local miniconda installation.
- Before running any install, build, test, configure, or Python command in this repo, activate the relevant project environment first.
- Activate the environment before using repo scripts, package installation, CMake configuration, compilation, testing, or Python tooling tied to this repository.
- Use the appropriate conda activation command for the current machine, for example:
  source <path-to-conda>/etc/profile.d/conda.sh && conda activate <project-env>
- Do not run cmake, ninja, ctest, make, pip, or Python scripts in this repo until the environment is active.