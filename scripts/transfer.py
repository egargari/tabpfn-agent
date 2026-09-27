"""Upload / download files for the TabPFN MCP handoff, restricted to Prior Labs storage.

The TabPFN MCP server hands out signed URLs and suggests running curl. This script does
the same transfer, but refuses anything outside two boundaries, so it can be
pre-approved instead of blanket-allowing curl:
  - the remote URL must be https://storage.googleapis.com/p2or-datasets-prod/...
  - the local file must be inside this project's reports/ folder

Usage:
  uv run python scripts/transfer.py upload <local_file> "<upload_command or signed URL>"
  uv run python scripts/transfer.py download "<prediction_uri>" <local_file>

For upload, pass the server's `upload_command` verbatim (in double quotes): the script
takes the signed URL and headers from it, so they always match what the server signed.
"""

import shlex
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

REPORTS = Path(__file__).resolve().parent.parent / "reports"
ALLOWED_HOST = "storage.googleapis.com"
ALLOWED_PREFIX = "/p2or-datasets-prod/"


def check_url(url: str) -> str:
    u = urlparse(url)
    if u.scheme != "https" or u.hostname != ALLOWED_HOST or not u.path.startswith(ALLOWED_PREFIX):
        sys.exit(f"refused: {u.scheme}://{u.hostname}{u.path[:40]}... is not Prior Labs storage")
    return url


def check_local(path: str) -> Path:
    p = Path(path).resolve()
    if not p.is_relative_to(REPORTS):
        sys.exit(f"refused: {path} is outside reports/")
    return p


def parse_upload(spec: str) -> tuple[str, dict[str, str]]:
    """Signed URL + headers from a curl upload_command, or a bare URL with no headers."""
    if not spec.lstrip().startswith("curl"):
        return spec, {}
    tokens = shlex.split(spec)
    headers, url = {}, None
    for i, tok in enumerate(tokens):
        if tok == "-H":
            name, _, value = tokens[i + 1].partition(":")
            headers[name.strip()] = value.strip()
        elif tok.startswith("https://"):
            url = tok
    if url is None:
        sys.exit("no https URL found in the upload command")
    headers.pop("Host", None)  # urllib sets it from the URL
    return url, headers


def send(request: urllib.request.Request) -> bytes:
    try:
        with urllib.request.urlopen(request) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")


def main() -> None:
    if len(sys.argv) != 4 or sys.argv[1] not in ("upload", "download"):
        sys.exit(__doc__)
    action = sys.argv[1]
    if action == "upload":
        local = check_local(sys.argv[2])
        url, headers = parse_upload(sys.argv[3])
        send(urllib.request.Request(check_url(url), data=local.read_bytes(), headers=headers, method="PUT"))
        print(f"uploaded {local.relative_to(REPORTS.parent)} ({local.stat().st_size:,} bytes)")
    else:
        url, local = check_url(sys.argv[2]), check_local(sys.argv[3])
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(send(urllib.request.Request(url)))
        print(f"downloaded {local.relative_to(REPORTS.parent)} ({local.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
