#!/bin/bash
# Convenience wrapper: create the bootstrap Job (with generateName-suffix
# discovery), tail its logs, then delete it. Run from any machine with
# kubectl access to the cluster.
#
# Usage:
#   NAMESPACE=<your-namespace> bash deploy/k8s/run_bootstrap.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-openfingym}"
PREFIX="bootstrap-openfingym-"

echo "=== Creating bootstrap Job ==="
kubectl -n "${NAMESPACE}" create -f "${SCRIPT_DIR}/bootstrap_repo.yaml"

# Find the just-created Job
sleep 2
JOB="$(kubectl -n "${NAMESPACE}" get jobs \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
    | grep "^${PREFIX}" | tail -1)"
echo "Job: ${JOB}"

# Wait for a pod
for i in $(seq 1 20); do
    POD="$(kubectl -n "${NAMESPACE}" get pods -l job-name="${JOB}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"
    [[ -n "${POD}" ]] && break
    echo "Waiting for pod (${i}/20)..."
    sleep 5
done

if [[ -z "${POD:-}" ]]; then
    echo "Timed out waiting for pod" >&2
    kubectl -n "${NAMESPACE}" describe job "${JOB}" >&2
    exit 2
fi
echo "Pod: ${POD}"

kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${POD}" --timeout=300s || true
kubectl -n "${NAMESPACE}" logs -f "${POD}" || true

# Wait for completion (may already be done by the time we got logs)
kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/${JOB}" --timeout=600s

echo "=== Bootstrap done. Cleaning up Job ==="
kubectl -n "${NAMESPACE}" delete "job/${JOB}"
