#!/usr/bin/env bash
# One-shot GCP T4 VM for the TensorRT/Triton numbers (your Mac has no NVIDIA GPU).
# Cost: a T4 VM is roughly well under $1/hour; delete it when done (last command).
#   PROJECT=my-proj ZONE=us-central1-a bash scripts/gpu_vm.sh
set -euo pipefail
: "${PROJECT:?}"; ZONE=${ZONE:-us-central1-a}; VM=${VM:-pathrank-gpu}
gcloud compute instances create "$VM" --project "$PROJECT" --zone "$ZONE" \
  --machine-type n1-standard-8 --accelerator type=nvidia-tesla-t4,count=1 --maintenance-policy TERMINATE \
  --image-family common-cu123-ubuntu-2204-py310 --image-project deeplearning-platform-release \
  --boot-disk-size 100GB --metadata install-nvidia-driver=True
cat <<MSG
VM created. Next:
  gcloud compute scp --recurse artifacts/careerbuilder/v1/model_repository scripts $VM:~/pathrank/ --zone $ZONE --project $PROJECT
  gcloud compute ssh $VM --zone $ZONE --project $PROJECT -- 'cd pathrank && bash scripts/build_tensorrt.sh model_repository && bash scripts/triton_perf.sh model_repository'
  gcloud compute scp --recurse $VM:~/pathrank/triton_perf . --zone $ZONE --project $PROJECT
  gcloud compute instances delete $VM --zone $ZONE --project $PROJECT --quiet
MSG
