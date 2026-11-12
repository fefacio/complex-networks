from collections.abc import Iterator
from pathlib import Path
from typing import Literal


def iter_files(
    directory: str | Path,
    pattern: str = "*",
    *,
    order: Literal["name", "size"] = "name",
) -> Iterator[Path]:
    """Itera sobre os arquivos que correspondem ao padrão no diretório.

    ``name`` ordena alfabeticamente pelo nome, sem distinguir maiúsculas.
    ``size`` ordena do menor para o maior em bytes no disco e usa o nome
    para desempatar.
    """
    paths = (path for path in Path(directory).glob(pattern) if path.is_file())

    def name_key(path: Path) -> tuple[str, str]:
        return path.name.casefold(), path.as_posix()

    if order == "name":
        return iter(sorted(paths, key=name_key))
    if order == "size":
        return iter(sorted(paths, key=lambda path: (path.stat().st_size, *name_key(path))))
    raise ValueError(f"Ordem inválida {order!r}. Use 'name' ou 'size'.")
