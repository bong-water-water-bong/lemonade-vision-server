from pathlib import Path


def remove_background(image_path: Path, out_path: Path) -> Path:
    try:
        from rembg import remove as rembg_remove

        with open(image_path, "rb") as f:
            data = f.read()
        result = rembg_remove(data)
        if not isinstance(result, bytes):
            raise TypeError("rembg returned a non-bytes result")
        with open(out_path, "wb") as f:
            f.write(result)
        return out_path
    except Exception:
        import shutil

        shutil.copy2(image_path, out_path)
        return out_path
