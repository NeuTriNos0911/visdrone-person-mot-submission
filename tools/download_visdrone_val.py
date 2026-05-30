from __future__ import annotations

import argparse
import html
from pathlib import Path
import re
from zipfile import ZipFile


FILE_ID = "1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu"
DEFAULT_NAME = "VisDrone2019-MOT-val.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download VisDrone2019 MOT validation set.")
    parser.add_argument("--out", default="data/raw", help="Directory to store the zip and extracted data.")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / DEFAULT_NAME
    if zip_path.exists() and zip_path.stat().st_size < 1024 * 1024:
        print(f"Removing incomplete download: {zip_path}")
        zip_path.unlink()
    if not zip_path.exists():
        download_google_drive_file(FILE_ID, zip_path)
    else:
        print(f"Already exists: {zip_path}")

    if not args.no_extract:
        extract_dir = out_dir / "VisDrone2019-MOT-val"
        if extract_dir.exists():
            print(f"Already extracted: {extract_dir}")
        else:
            with ZipFile(zip_path) as zf:
                zf.extractall(out_dir)
            print(f"Extracted to: {out_dir}")
    return 0


def download_google_drive_file(file_id: str, destination: Path) -> None:
    try:
        import requests
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError("Install requirements first: `pip install -r requirements.txt`.") from exc

    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True, timeout=30)
    token = _confirm_token(response)
    if token:
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True, timeout=30)
    elif _looks_like_drive_warning(response):
        action, params = _parse_drive_warning_form(response.text)
        response = session.get(action, params=params, stream=True, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type.lower():
        raise RuntimeError("Google Drive returned HTML instead of the ZIP. Open the Drive link manually.")

    total = int(response.headers.get("content-length", 0))
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with tmp_path.open("wb") as fh, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=destination.name,
    ) as progress:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                fh.write(chunk)
                progress.update(len(chunk))
    tmp_path.replace(destination)
    print(f"Downloaded: {destination}")


def _confirm_token(response) -> str | None:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _looks_like_drive_warning(response) -> bool:
    content_type = response.headers.get("content-type", "")
    return "text/html" in content_type.lower() and "download-form" in response.text


def _parse_drive_warning_form(page: str) -> tuple[str, dict[str, str]]:
    action_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', page)
    if not action_match:
        raise RuntimeError("Could not find Google Drive confirmation form.")
    action = html.unescape(action_match.group(1))
    params = {
        html.unescape(name): html.unescape(value)
        for name, value in re.findall(
            r'<input[^>]+type="hidden"[^>]+name="([^"]+)" value="([^"]*)"',
            page,
        )
    }
    if "id" not in params or "confirm" not in params:
        raise RuntimeError("Could not parse Google Drive confirmation parameters.")
    return action, params


if __name__ == "__main__":
    raise SystemExit(main())

