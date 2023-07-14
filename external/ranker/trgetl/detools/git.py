try:
    import git
except ImportError:
    git = None
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from ..filesystem import REPO_PATH


class Git:
    def __init__(self, path: Path = None, branch: str = "master"):
        if path is None:
            path = REPO_PATH
        self.repo = git.Repo(path)
        self.branch = branch

    @contextmanager
    def stash(self) -> Iterator:
        try:
            print(">>> git stash")
            stash = self.repo.git.stash()
            print(stash)
            yield
        finally:
            if stash != "No local changes to save":
                print(">>> git stash pop")
                print(self.repo.git.stash("pop"))

    @contextmanager
    def other_branch(self, name: str) -> Iterator:
        try:
            print(f">>> git branch {name}")
            print(self.repo.git.branch(name))
            self.checkout(name)
            yield
        finally:
            self.checkout()
            print(f">>> git branch -D {name}")
            print(self.repo.git.branch(name, D=True))

    def is_on_branch(self) -> bool:
        return self.branch == str(self.repo.active_branch)

    def checkout(self, branch: Optional[str] = None) -> None:
        if branch is None:
            branch = self.branch
        print(f">>> git checkout {branch}")
        with self.stash():
            print(self.repo.git.checkout(branch))

    def status(self) -> None:
        assert self.is_on_branch()
        print(">>> git status")
        print(self.repo.git.status())

    def log(self) -> None:
        assert self.is_on_branch()
        print(">>> git log")
        print(self.repo.git.log()[:10000])

    def diff(self) -> None:
        assert self.is_on_branch()
        print(">>> git diff")
        print(self.repo.git.diff())

    def pull(self) -> None:
        assert self.is_on_branch()
        with self.stash():
            print(">>> git fetch")
            print(self.repo.git.fetch())
            print(f">>> git rebase origin/{self.branch}")
            print(self.repo.git.rebase(f"origin/{self.branch}"))

    def push(self, branch: Optional[str] = None) -> None:
        assert self.is_on_branch()
        self.pull()

        if branch is None:
            print(f">>> git push origin {self.branch}")
            print(self.repo.git.push("origin", branch))
        else:
            assert " " not in branch
            with self.other_branch(branch):
                print(f">>> git push origin {branch}")
                print(self.repo.git.push("origin", branch))

    def commit(
        self,
        comment: str,
        push: bool = False,
        on_branch: bool = False,
        add: bool = False,
    ) -> None:
        assert self.is_on_branch()
        if add:
            self.add()
        print(f">>> git commit -m '{comment}'")
        print(self.repo.git.commit(m=comment))

        if push:
            branch = None
            if on_branch:
                if isinstance(on_branch, str):
                    branch = on_branch
                else:
                    branch = comment
            self.push(branch)

    def add(self, file: Optional[str] = None) -> None:
        assert self.is_on_branch()
        if file is not None:
            print(f">>> git add {file}")
            print(self.repo.git.add(file))

        else:
            diff = self.repo.index.diff(None)
            add_files = [d.a_path for d in diff if not d.deleted_file]
            add_files += self.repo.untracked_files
            rm_files = [d.a_path for d in diff if d.deleted_file]

            for add_file in add_files:
                self.add(add_file)
            for rm_file in rm_files:
                self.rm(rm_file)

    def rm(self, file: str) -> None:
        assert self.is_on_branch()
        print(f">>> git rm {file} -r")
        print(self.repo.git.rm(file, r=True))

    def reset(self, file: Optional[str] = None, hard: bool = False) -> None:
        if hard:
            print(f">>> git reset {file} --hard")
        else:
            print(f">>> git reset {file}")
        if file is None:
            print(self.repo.git.reset(hard=hard))
        else:
            print(self.repo.git.reset(file, hard=hard))
