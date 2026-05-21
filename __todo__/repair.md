So we made that helm chart:

Hm, when starting now with compose up:

------                                                                                                                                                       
 > [integrator internal] load metadata for docker.io/library/lib:latest:
------
[+] up 0/8
 ⠙ Image service           Building                                                                                                                      1.8s
 ⠙ Image server            Building                                                                                                                      1.8s
 ⠙ Image worker:latest     Building                                                                                                                      1.8s
 ⠙ Image web:latest        Building                                                                                                                      1.8s
 ⠙ Image queue:latest      Building                                                                                                                      1.8s
 ⠙ Image integrator:latest Building                                                                                                                      1.8s
 ⠙ Image philo             Building                                                                                                                      1.8s
 ⠙ Image nginx:latest      Building                                                                                                                      1.8s
Dockerfile:10

--------------------

   8 |     

   9 |     COPY . .

  10 | >>> COPY --from=lib . /app/lib/

  11 |     RUN mkdir /.cache

  12 |     RUN chmod 777 /.cache

--------------------

target worker: failed to solve: lib: failed to resolve source metadata for docker.io/library/lib:latest: pull access denied, repository does not exist or may require authorization: server message: insufficient_scope: authorization failed

---

**Fix applied:** `COPY --from=lib . /app/lib/` in Dockerfiles references a named build context `lib`, but no local stage or context existed → Docker tried `docker.io/library/lib:latest`. Added `additional_contexts: - lib=./lib` to all 8 services in compose.yml (server, service, philo, integrator, linker, classifier, worker, queue). BuildKit now resolves `--from=lib` to local `./lib` directory instead of Docker Hub.
