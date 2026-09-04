#!/bin/bash
# Launch an RL training Job on Kubernetes: render openfinai_rl_job.tpl.yml
# with envsubst, create it, then follow the Pod's logs. No file transfer
# happens here — the code is already on the PVC (see bootstrap_repo.yaml).
#
# Run from any machine with kubectl access to the cluster, after:
#   1) kubectl -n <ns> apply -f pvc.yaml                (one-time)
#   2) bash run_bootstrap.sh                            (one-time; clones code onto the PVC)
#   3) kubectl -n <ns> create secret generic openfinai-secrets ...   (optional; HF/W&B keys)
#   4) docker push <your-image>                         (one-time per image rev)
#   5) (if code changed since last run) bash run_update.sh
#
# Usage:
#   NAMESPACE=<ns> bash launch_rl.sh <image> [smoke|train] [job-name-prefix] [--provider singularity|podman]
#
# Defaults: namespace=openfingym, variant=smoke (1 GPU, Qwen3-1.7B,
#           offline_crypto_forecasting only), prefix=ofg-rl-<variant>,
#           provider=singularity.

set -euo pipefail

export NAMESPACE="${NAMESPACE:-openfingym}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/openfinai_rl_job.tpl.yml"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <image> [smoke|train] [job-name-prefix] [--provider singularity|podman]" >&2
    exit 1
fi

export IMAGE="$1"
VARIANT="${2:-smoke}"
export JOB_NAME_PREFIX="${3:-ofg-rl-${VARIANT}}"

# Optional --provider flag. Default singularity (any cluster where apptainer
# can obtain the capabilities it needs inside the Pod). In namespaces running
# the `baseline` PodSecurity profile (which blocks SYS_ADMIN), pass
# `--provider podman` to swap in the rootless Podman provider
# (config/harbor_trial_rl_podman.yaml).
PROVIDER="singularity"
for arg in "${@:4}"; do
    case "${arg}" in
        --provider=*)
            PROVIDER="${arg#--provider=}"
            ;;
        --provider)
            : # next loop iteration would consume the value; handled below
            ;;
        singularity|podman)
            # Bare value following --provider in a separate arg.
            PROVIDER="${arg}"
            ;;
    esac
done

case "${PROVIDER}" in
    singularity)
        export HARBOR_TRIAL_CONFIG="config/harbor_trial_rl.yaml"
        ;;
    podman)
        export HARBOR_TRIAL_CONFIG="config/harbor_trial_rl_podman.yaml"
        ;;
    *)
        echo "unknown --provider: ${PROVIDER} (use singularity|podman)" >&2
        exit 1
        ;;
esac
export PROVIDER

# Variant defines resources + launcher path. Edit here to tune.
case "${VARIANT}" in
    smoke)
        export NUM_GPUS=1
        export CPU_REQ=4
        export CPU_LIM=8
        export RAM_REQ_GB=64
        export RAM_LIM_GB=96
        export LAUNCHER_SCRIPT="scripts/rl/smoke_rl_single_gpu.sh"
        ;;
    train)
        export NUM_GPUS=8
        export CPU_REQ=16
        export CPU_LIM=32
        export RAM_REQ_GB=160
        export RAM_LIM_GB=200
        export LAUNCHER_SCRIPT="scripts/rl/train_rl.sh"
        ;;
    *)
        echo "unknown variant: ${VARIANT} (use smoke|train)" >&2
        exit 1
        ;;
esac

RENDERED="/tmp/${JOB_NAME_PREFIX}.yml"
# Pass an explicit variable list so envsubst leaves bash-only $-references
# (e.g. $f in shell loops inside container args) untouched.
envsubst '${NAMESPACE} ${IMAGE} ${JOB_NAME_PREFIX} ${NUM_GPUS} ${CPU_REQ} ${CPU_LIM} ${RAM_REQ_GB} ${RAM_LIM_GB} ${LAUNCHER_SCRIPT} ${HARBOR_TRIAL_CONFIG} ${PROVIDER}' \
    < "${TEMPLATE}" > "${RENDERED}"

echo "=== Rendered manifest: ${RENDERED} ==="
echo "    image:      ${IMAGE}"
echo "    variant:    ${VARIANT}"
echo "    launcher:   ${LAUNCHER_SCRIPT}"
echo "    provider:   ${PROVIDER}"
echo "    trial-yaml: ${HARBOR_TRIAL_CONFIG}"
echo "    GPUs:       ${NUM_GPUS}"
echo "    namespace:  ${NAMESPACE}"
echo ""

kubectl -n "${NAMESPACE}" create -f "${RENDERED}"

# Find the just-created Job (most recent with our prefix).
sleep 2
JOB_NAME="$(kubectl -n "${NAMESPACE}" get jobs \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
    | grep "^${JOB_NAME_PREFIX}-" | tail -1)"
echo "Job: ${JOB_NAME}"

# Wait for a pod to exist (Kueue may queue us).
POD_WAIT_TIMEOUT="${POD_WAIT_TIMEOUT:-3600}"
elapsed=0
POD_NAME=""
while [[ ${elapsed} -lt ${POD_WAIT_TIMEOUT} ]]; do
    POD_NAME="$(kubectl -n "${NAMESPACE}" get pods \
        -l job-name="${JOB_NAME}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
    if [[ -n "${POD_NAME}" ]]; then
        break
    fi
    echo "Waiting for pod (elapsed=${elapsed}s; Kueue may be queueing)..."
    sleep 30
    elapsed=$((elapsed + 30))
done

if [[ -z "${POD_NAME}" ]]; then
    echo "Timed out waiting for pod for ${JOB_NAME}" >&2
    kubectl -n "${NAMESPACE}" describe job "${JOB_NAME}" >&2
    exit 2
fi

echo "Pod: ${POD_NAME}"
echo "Following logs (Ctrl+C detaches; Job keeps running):"
echo ""

# Wait for Ready before attaching logs (image pull / scheduling).
kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${POD_NAME}" --timeout=1800s || true
exec kubectl -n "${NAMESPACE}" logs -f "${POD_NAME}"
