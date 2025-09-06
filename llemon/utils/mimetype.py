import mimetypes


def get_mimetype(name: str) -> str:
    mimetype, _ = mimetypes.guess_type(name)
    if not mimetype:
        raise ValueError(f"unknown mimetype for {name}")
    return mimetype


def get_extension(mimetype: str) -> str:
    extension = mimetypes.guess_extension(mimetype)
    if not extension:
        raise ValueError(f"unknown extension for {mimetype}")
    return extension
