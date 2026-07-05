# Decision Log

## 2026-07-01 — Get started
**Decision**: Use venv to create an isolated Python environment for this project.

**Reason**: Keeps this project's dependencies separate from the global / conda base environment, 
so package versions across projects can't collide.
venv is lightweight and pairs cleanly with pip + requirements.txt for
reproducibility. Chose venv over conda create because the project
dependencies are pip-installable and I want requirements.txt to be the
single source of truth.

**Scope**: this project 
