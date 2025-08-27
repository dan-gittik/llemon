import pathlib

from llemon.sync.tool import Toolbox


class Directory(Toolbox):

    def __init__(self, path: str | pathlib.Path, readonly: bool = True) -> None:
        self.path = pathlib.Path(path).absolute()
        self.readonly = readonly
        super().__init__(str(self.path))
        self._init.update(
            path=str(self.path),
            readonly=self.readonly,
        )

    def __str__(self) -> str:
        return f"directory at {self.path}"

    def __repr__(self) -> str:
        return f"<{self}>"

    @property
    def tool_names(self) -> list[str]:
        if self.readonly:
            return ["read_files"]
        return ["read_files", "write_file", "delete_files"]

    def read_files_tool(self, paths: list[str]) -> dict[str, str]:
        contents: dict[str, str] = {}
        for path in paths:
            contents[path] = self._path(path).read_text()
        return contents

    def read_files_description(self) -> str:
        files = []
        for file in self.path.rglob("*"):
            files.append(f"- {file.relative_to(self.path)} ({file.stat().st_size}b)")
        return f"""
            Receives the paths of the files to read, and returns a dictionary mapping their paths to their contents.
            The available files are:
            {'\n'.join(files)}
        """

    def write_file_tool(self, path: str, content: str) -> None:
        """
        Receives a path of the file to write to and its content.
        """
        self._path(path).write_text(content)

    def delete_files_tool(self, paths: list[str]) -> None:
        """
        Receives the paths of the files to delete.
        """
        for path in paths:
            self._path(path).unlink()

    def render_file(self, path: str) -> str:
        return self._path(path).read_text()

    def _path(self, file: str) -> pathlib.Path:
        if ".." in file.split("/"):
            raise ValueError("using .. in the filename is not allowed")
        return self.path / file
