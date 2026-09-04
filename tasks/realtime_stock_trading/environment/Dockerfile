# Shared sandbox image: same for every curated + generated task.
# Singularity needs a registry image (can't build locally); Docker pulls
# it under the default flow too. Per-task data lands in /data via
# Harbor's mounts mechanism, set per-trial by
# openfinai_harbor.runner.build_task_data_mount().
FROM nihao0630/openfinai-base:v1

WORKDIR /workspace
