# ── K8s deploy targets ───────────────────────────────────
# Used via `just kube::<target>` from main justfile (mod kube)
set export := true

CHART := ".helm-chart"
RELEASE := "system"
KIND_CLUSTER := "my-kind-cluster"
NAMESPACE := "system"               # override: just kube::helm-install NAMESPACE=foo
REMOTE_NAMESPACE := "polarity-science"
IMAGES := "nginx server service philo integrator linker classifier web redis queue worker"
KUBECONFIG := "/home/ich/2-5-1/__kyber__/k3s-config.yaml"

# ── Kind ──────────────────────────────────────────────────

# create kind cluster
kind-create:
    kind create cluster --name {{KIND_CLUSTER}}

# delete kind cluster
kind-delete:
    kind delete cluster --name {{KIND_CLUSTER}}

# ── Images ────────────────────────────────────────────────

# build all images (nginx + web via compose, rest with lib build-context)
kind-build:
    docker compose build nginx web
    DOCKER_BUILDKIT=1 bash -c ' \
        for svc in server service philo integrator linker classifier worker; do \
            docker build --build-context lib=./lib --build-context integrator=./integrator --build-context classifier=./classifier -f $svc/Dockerfile -t $svc:latest $svc/; \
        done; \
        docker build --build-context lib=./lib --build-context integrator=./integrator --build-context classifier=./classifier -f worker/Dockerfile -t queue:latest worker/'

# pull external images (redis)
kind-pull:
    docker pull redis:latest

# load images into kind cluster (tagged with localhost/ prefix)
kind-load: kind-build kind-pull
    for img in {{IMAGES}}; do \
        docker tag $img:latest localhost/$img:latest 2>/dev/null; \
        kind load docker-image localhost/$img:latest --name {{KIND_CLUSTER}}; \
    done

# ── Helm ──────────────────────────────────────────────────

# render templates locally (debug)
helm-template:
    helm template {{RELEASE}} {{CHART}} \
        --set env.host=localhost \
        --set env.target=dev \
        --set images.registry=localhost/ \
        --set images.pullPolicy=IfNotPresent

# install or upgrade chart to current k8s context
helm-install:
    helm upgrade --install {{RELEASE}} {{CHART}} \
        --namespace {{NAMESPACE}} \
        --create-namespace \
        --set env.host=localhost \
        --set env.target=dev \
        --set images.registry=localhost/ \
        --set images.pullPolicy=IfNotPresent

# uninstall chart
helm-delete:
    helm uninstall {{RELEASE}} --namespace {{NAMESPACE}}

# list helm releases
helm-list:
    helm list --all-namespaces

# ── Full pipeline ─────────────────────────────────────────

# build + load + deploy to kind
kind-deploy: kind-load helm-install

# uninstall from remote
remote-delete:
    helm uninstall {{RELEASE}} --namespace {{REMOTE_NAMESPACE}}

# ── Remote (polarity.science) ────────────────────────────────

DOCKER_REGISTRY := "c0ntradicti0n/pianalytic"

# build all images tagged for remote
remote-build:
    DOCKER_BUILDKIT=1 docker build -t localhost/system-base:latest ./base
    TARGET=prod docker compose build web nginx
    DOCKER_BUILDKIT=1 bash -c ' \
        for svc in server service philo integrator linker classifier worker; do \
            docker build --build-context lib=./lib --build-context integrator=./integrator --build-context classifier=./classifier -f $svc/Dockerfile -t $svc:latest $svc/; \
        done; \
        docker build --build-context lib=./lib --build-context integrator=./integrator --build-context classifier=./classifier -f worker/Dockerfile -t queue:latest worker/'

# tag and push images to Docker Hub
remote-push:
    #!/usr/bin/env bash
    set -euo pipefail
    docker login -u c0ntradicti0n --password "$DOCKER_API_KEY"
    for svc in nginx server service philo integrator linker classifier web queue worker; do \
        tag="{{DOCKER_REGISTRY}}:system-$svc"; \
        docker tag $svc:latest $tag; \
        docker push $tag; \
    done

# install/upgrade helm chart on remote k3s
remote-deploy:
    helm upgrade --install {{RELEASE}} {{CHART}} \
        --namespace {{REMOTE_NAMESPACE}} \
        --create-namespace \
        --set env.host=polarity.science \
        --set env.target=prod \
        --set images.registry={{DOCKER_REGISTRY}}:system- \
        --set images.redis=redis:latest \
        --set images.pullPolicy=Always \
        --set ingress.enabled=true \
        --set ingress.host=polarity.science \
        --set ingress.className=nginx \
        --set ingress.annotations.'cert-manager\.io/cluster-issuer'=letsencrypt-prod \
        --set ingress.tls.secretName=polarity-science-tls
    kubectl rollout restart deployment -n {{REMOTE_NAMESPACE}} --all 2>/dev/null || true

# push images + deploy in one step
remote-deploy-full: remote-build remote-push remote-deploy remote-sync-data

# just re-deploy without rebuilding (images already pushed)
remote-upgrade: remote-deploy remote-sync-data

# ── Data sync (dev loop) ──────────────────────────────────

DIALECTICS_SRC := "../dialectics"

# sync dialectics data into running server pod (instant, no rebuild)
sync-data:
    #!/usr/bin/env bash
    set -e
    kubectl wait --for=condition=ready pod -n $NAMESPACE -l app=server --timeout=30s
    POD=$(kubectl get pod -n $NAMESPACE -l app=server -o name)
    tar -C "$SYSTEM" -cf - . | kubectl exec -n $NAMESPACE $POD -i -- tar -C /system -xf - 2>/dev/null || true
    echo "synced $SYSTEM to server:/system"

# sync dialectics data to remote cluster
remote-sync-data:
    #!/usr/bin/env bash
    set -e
    SRC="${SYSTEM:-$DIALECTICS_SRC}"
    kubectl wait --for=condition=ready pod -n {{REMOTE_NAMESPACE}} -l app=server --timeout=60s
    POD=$(kubectl get pod -n {{REMOTE_NAMESPACE}} -l app=server -o name)
    tar -C "$SRC" -cf - . | kubectl exec -n {{REMOTE_NAMESPACE}} $POD -i -- tar -C /system -xf - 2>/dev/null || true
    echo "synced $SRC to server:/system (namespace: {{REMOTE_NAMESPACE}})"

# rebuild + reload all images (code changes)
reload: kind-load
    kubectl delete pods -n {{NAMESPACE}} --all --force --grace-period=0 2>/dev/null

# rebuild code + sync data (fast dev loop)
redeploy: reload sync-data

# ── Utility ───────────────────────────────────────────────

# port-forward nginx to localhost
pf:
    kubectl port-forward --namespace {{NAMESPACE}} svc/nginx 8080:8080 8443:8443

# show pod status
pods:
    kubectl get pods --namespace {{NAMESPACE}} -o wide

# show all k8s resources for the release
resources:
    kubectl get all,pvc,configmap,secret,ingress --namespace {{NAMESPACE}}
