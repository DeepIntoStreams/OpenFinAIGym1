# Kubernetes deployment for OpenFinGym RL training

Runs the SkyRL RL pipeline (`scripts/rl/*.sh`) as a single-Pod Kubernetes Job.
The training image carries the *environment* (CUDA, SkyRL, Harbor, deps); the
*code* lives on a shared ReadWriteMany PVC and is refreshed with `git`, so the
image only needs rebuilding when dependencies change.

## Files

| File | Purpose |
|---|---|
| `pvc.yaml` | One RWX PVC (`openfingym-workspace`) with subdirs `code/`, `sif_cache/`, `run_output/`, `hf_cache/`, `podman_storage/`. |
| `bootstrap_repo.yaml` | Job that clones `OpenFinGym1` + `SkyRL-main` into `code/` on the PVC (idempotent). |
| `run_bootstrap.sh` | Wrapper: create the bootstrap Job, follow its logs, delete it. |
| `transfer_pod.yaml` | Fallback when the cluster cannot reach github.com: a sleeping Pod that mounts the PVC for `kubectl cp` (instructions in the file header). |
| `openfinai_rl_job.tpl.yml` | envsubst template for the training Job (namespace, image, GPUs, resources, launcher script, sandbox provider). |
| `launch_rl.sh` | Renders the template, creates the Job, waits for the Pod and follows its logs. |
| `update_repo.yaml` / `run_update.sh` | Job (and wrapper) that `git fetch` + `reset --hard` both checkouts on the PVC between runs. |
| `../docker/` | Dockerfile + build script for the training image. |

## Values to edit

1. **Namespace** - the static manifests hard-code `openfingym` (marked `# EDIT:` at the top of each file); replace it before applying. The shell wrappers and the Job template read `NAMESPACE` instead (default `openfingym`).
2. **Storage class** - `pvc.yaml` uses `csi-cephfs-sc`; set any class that supports `ReadWriteMany`.
3. **Image** - pass your image to `launch_rl.sh` (the published one is `ghcr.io/deepintostreams/openfinai-rl:v1`; build your own with `deploy/docker/build_and_push.sh`).

Optional: the `kueue.x-k8s.io/queue-name` label (`<namespace>-user-queue`) can be removed on clusters without Kueue, and the `nodeSelector` in the Job template pins a GPU product - edit or remove it to match your node pool.

## Order of operations

```bash
export NAMESPACE=openfingym                                              # your namespace
kubectl -n "$NAMESPACE" apply -f deploy/k8s/pvc.yaml                     # 1. storage (once)
bash deploy/k8s/run_bootstrap.sh                                         # 2. clone code onto the PVC
kubectl -n "$NAMESPACE" create secret generic openfinai-secrets \
    --from-literal=huggingface=<hf-token> --from-literal=wandb=<wandb-key> # 3. optional (both keys optional)
bash deploy/k8s/launch_rl.sh ghcr.io/deepintostreams/openfinai-rl:v1 smoke  # 4. 1-GPU smoke run
bash deploy/k8s/launch_rl.sh ghcr.io/deepintostreams/openfinai-rl:v1 train  #    full 8-GPU run
bash deploy/k8s/run_update.sh                                            # 5. after pushing new commits
```

`launch_rl.sh` accepts `--provider singularity` (default) or `--provider podman`.
Use `podman` when the namespace runs the `baseline` PodSecurity profile, which
blocks the `SYS_ADMIN` capability Apptainer needs: the Pod then installs rootless
Podman at start-up and the trial config switches to
`config/harbor_trial_rl_podman.yaml`.

## Layout inside the training Pod

- `/workspace` = PVC `code/` -> `/workspace/OpenFinGym1`, `/workspace/SkyRL-main`
- `/workspace/OpenFinGym1/data/run_output` = PVC `run_output/` (trials, verifier registry, checkpoints)
- `/workspace/sif_cache`, `/workspace/podman_storage` = sandbox image caches
- `/root/.cache/huggingface` = PVC `hf_cache/`
