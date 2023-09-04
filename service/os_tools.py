import os
import subprocess
from contextlib import contextmanager


@contextmanager
def git_auto_commit(repo_path, commit_message_prefix=""):
    # remember our original working directory
    original_path = os.getcwd()

    # change to repo path
    os.chdir(repo_path)

    try:
        # Check if the directory is a Git repo; if not, initialize it
        subprocess.check_call(["git", "rev-parse"], stderr=subprocess.STDOUT)

        subprocess.check_call(["git", "add", "-A"])

        # compute initial git status
        initial_status = (
            subprocess.check_output(["git", "status", "--porcelain", "."])
            .decode()
            .split("\n")
        )

        os.chdir(original_path)
        yield  # This is where your code runs
        os.chdir(repo_path)

        subprocess.check_call(["git", "add", "-A"])

        # compute final git status
        final_status = (
            subprocess.check_output(["git", "status", "--porcelain", "."])
            .decode()
            .split("\n")
        )

        # find changed files
        changed_files = set(final_status) - set(initial_status)
        if not changed_files:
            changed_files = final_status

        changed_files = [file.split()[-1] for file in changed_files if file]

        # add and commit changed files
        if changed_files:
            # 'git add -A' stages all changes
            commit_message = f"{commit_message_prefix}: {' '.join(changed_files)}"
            subprocess.check_call(["git", "commit", "-m", commit_message])
            print(f"Commit made with message: {commit_message}")

    except subprocess.CalledProcessError as e:
        print("Error occurred in git operation:", e)
        raise e

    finally:
        # change back to the original directory
        os.chdir(original_path)
