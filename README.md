# mioti-mic-push-t
LeRobot PushT in colab.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/mioti-mic/mioti-mic-push-t/blob/main/notebooks/pusht_smoke_test.ipynb)


## Funciones para huggingface
``` bash
src/pusht/
  data/
    __init__.py
    hf_io.py        # login, create/load dataset, push_to_hub, upload_folder, revisions
    schema.py       # Features (datasets.Features), validaciones, normalización
    writers.py      # convertir rollouts -> tablas/arrow, sharding, parquet/jsonl
    rollouts.py     # generate rollouts (env loop), extracción info, determinismo seeds
    materialize.py  # steps/episodes -> DatasetDict, sharding, parquet, checksums
    metadata.py     # run_metadata.json, schema.json, pip_freeze, git_commit

```