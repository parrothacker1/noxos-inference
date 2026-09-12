import os
import socket

DEFAULT_SOCKET_PATH = os.environ.get("CLAMD_SOCKET", "/var/run/clamav/clamd.ctl")
CHUNK_SIZE = 8192


class ClamdUnavailable(Exception):
    pass


def scan_bytes(data: bytes, socket_path: str = DEFAULT_SOCKET_PATH) -> tuple[str, str | None]:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(30)
        sock.connect(socket_path)
    except OSError as e:
        sock.close()
        raise ClamdUnavailable(f"cannot reach clamd at {socket_path}: {e}") from e

    try:
        sock.sendall(b"zINSTREAM\0")
        for offset in range(0, len(data), CHUNK_SIZE):
            chunk = data[offset:offset + CHUNK_SIZE]
            sock.sendall(len(chunk).to_bytes(4, "big") + chunk)
        sock.sendall((0).to_bytes(4, "big"))

        response = b""
        while True:
            part = sock.recv(4096)
            if not part:
                break
            response += part
    finally:
        sock.close()

    text = response.decode("utf-8", errors="replace").strip().rstrip("\0")
    if text.endswith("OK"):
        return "OK", None
    if "FOUND" in text:
        signature = text.rsplit(":", 1)[-1].replace("FOUND", "").strip()
        return "FOUND", signature
    return "ERROR", text
