from huggingface_hub import HfApi


def clean_hf(
    repo_id = "mioti-mic/mioti-mic-push-t",
    bad_prefixes = ("src/", "scripts/", "notebooks/", ".github/"),
    bad_suffixes = (".py", ".ipynb")
):
    api = HfApi()
    files = api.list_repo_files(repo_id = repo_id, repo_type = "dataset")

    to_delete = [
        file for file in files
        if file.startswith(bad_prefixes) or file.endswith(bad_suffixes)
    ]
    
    print("Archivos identificados para borrar:\n", to_delete)
    user_input = input(f"Escribe '{repo_id}' para borrar archivos:")

    if user_input == repo_id:
        for file in to_delete:
            api.delete_file(
                repo_id = repo_id,
                repo_type = "dataset",
                path_in_repo = file,
                commit_message = f"Remove non-dataset file: {file}"
            )
        print("Archivos borrados.")
    else:
        print("Cancelado.")


if __name__ == "__main__":
    clean_hf()
