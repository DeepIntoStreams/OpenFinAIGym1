#!/bin/bash
# Convenience wrapper: git fetch + reset both checkouts on the PVC. Run
# from any machine with kubectl access after pushing new commits to
# origin/main.
#
# Usage:
#   NAMESPACE=<your-namespace> bash deploy/k8s/run_update.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-openfingym}"
PREFIX="update-openfingym-"

kubectl -n "${NAMESPACE}" create -f "${SCRIPT_DIR}/update_repo.yaml"
sleep 2
JOB="$(kubectl -n "${NAMESPACE}" get jobs \
    --sort-by=.metadata.creationTimestamp \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' \
    | grep "^${PREFIX}" | tail -1)"
echo "Job: ${JOB}"

kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/${JOB}" --timeout=180s
kubectl -n "${NAMESPACE}" logs "job/${JOB}"
kubectl -n "${NAMESPACE}" delete "job/${JOB}"
