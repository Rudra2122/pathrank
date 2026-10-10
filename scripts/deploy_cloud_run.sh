#!/usr/bin/env bash
# Deploy the API to GCP Cloud Run with a canary.
#   PROJECT=my-proj bash scripts/deploy_cloud_run.sh                 # new revision, 0% traffic, tag "canary"
#   PROJECT=my-proj CANARY=10 bash scripts/deploy_cloud_run.sh       # ...and send it 10% of traffic
#   PROJECT=my-proj bash scripts/deploy_cloud_run.sh promote         # 100% to the canary revision
#   PROJECT=my-proj bash scripts/deploy_cloud_run.sh rollback        # 100% back to the previous revision
# Watch the canary on its tag URL (https://canary---pathrank-...run.app) and in Prometheus before promoting.
set -euo pipefail
: "${PROJECT:?set PROJECT}"; REGION=${REGION:-us-central1}; REPO=${REPO:-pathrank}; SVC=${SVC:-pathrank}
case "${1:-deploy}" in
  promote)
    REV=$(gcloud run services describe "$SVC" --region "$REGION" --project "$PROJECT" \
          --format='value(status.traffic.filter(tag=canary).revisionName)' | head -1)
    gcloud run services update-traffic "$SVC" --to-revisions "$REV=100" --region "$REGION" --project "$PROJECT"
    exit 0;;
  rollback)
    PREV=$(gcloud run revisions list --service "$SVC" --region "$REGION" --project "$PROJECT" \
           --format='value(metadata.name)' --sort-by=~metadata.creationTimestamp | sed -n 2p)
    gcloud run services update-traffic "$SVC" --to-revisions "$PREV=100" --region "$REGION" --project "$PROJECT"
    exit 0;;
esac
IMG="$REGION-docker.pkg.dev/$PROJECT/$REPO/pathrank:$(git rev-parse --short HEAD 2>/dev/null || date +%s)"
gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION" --project "$PROJECT"
gcloud builds submit --tag "$IMG" --project "$PROJECT" .
FIRST=$(gcloud run services describe "$SVC" --region "$REGION" --project "$PROJECT" >/dev/null 2>&1 && echo no || echo yes)
gcloud run deploy "$SVC" --image "$IMG" --region "$REGION" --project "$PROJECT" \
  --memory 4Gi --cpu 2 --min-instances 0 --max-instances 3 --timeout 120 --allow-unauthenticated \
  --set-env-vars PATHRANK_CONFIG=configs/careerbuilder.yaml \
  $( [ "$FIRST" = no ] && echo "--no-traffic --tag canary" )
# for the LLM agent: --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest
# for Triton on GKE/GPU VM: --set-env-vars TRITON_URL=<host>:8000 ; for Redis features: PATHRANK_SET=featurestore.online=redis ...
if [ "$FIRST" = no ] && [ -n "${CANARY:-}" ]; then
  gcloud run services update-traffic "$SVC" --to-tags "canary=$CANARY" --region "$REGION" --project "$PROJECT"
fi
