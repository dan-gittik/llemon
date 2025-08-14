import pathlib

from .toolbox import Toolbox


class Directory(Toolbox):

    def __init__(self, path: str | pathlib.Path, readonly: bool = True) -> None:
        self.path = pathlib.Path(path)
        self.readonly = readonly
    
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
            self._check_path(path)
            contents[path] = (self.path / path).read_text()
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
        self._check_path(path)
        (self.path / path).write_text(content)
    
    def delete_files_tool(self, paths: list[str]) -> None:
        """
        Receives the paths of the files to delete.
        """
        for path in paths:
            self._check_path(path)
            (self.path / path).unlink()
    
    def _check_path(self, file: str) -> None:
        if ".." in file.split("/"):
            raise ValueError("using .. in the filename is not allowed")