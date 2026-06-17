"""
Genetic lock: optional AES-256-CTR encryption of ``neural_weights`` in ``.axb`` payloads.

Topology / ABI / IR remain readable; only the serialized neural weight blob is ciphertext.
Requires optional dependency: ``pip install -e ".[lock]"``.
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class BundleUnlockError(RuntimeError):
    """Cannot decrypt bundle (wrong key, missing secret, or incompatible environment)."""


class BundleLockError(ValueError):
    """Cannot apply lock in this environment (e.g. device lock without CUDA)."""


class LockMode(str, Enum):
    NONE = "none"
    DEVICE = "device"
    HOST = "host"
    ENV_SECRET = "env-secret"


def _require_crypto() -> None:
    try:
        import cryptography  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Bundle lock requires optional cryptography: pip install -e \".[lock]\""
        ) from e


def _fingerprint_key(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:32]


def _derive_key_device() -> bytes:
    if not torch.cuda.is_available():
        raise BundleLockError("device lock requires a CUDA GPU and CUDA runtime")
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    uid = getattr(props, "uuid", None)
    if uid is not None:
        seed = f"device|cuda|{uid}".encode("utf-8")
    else:
        cap = torch.cuda.get_device_capability(idx)
        seed = f"device|cuda|{props.name}|{idx}|{cap[0]}.{cap[1]}".encode("utf-8")
    return hashlib.sha256(seed).digest()


def _derive_key_host() -> bytes:
    node = platform.node() or platform.uname().node or "unknown-host"
    mac = uuid.getnode()
    seed = f"host|{node}|{mac:x}".encode("utf-8")
    return hashlib.sha256(seed).digest()


def _derive_key_env_secret() -> bytes:
    secret = os.environ.get("AXIOM_BUNDLE_SECRET", "").strip()
    if not secret:
        raise BundleUnlockError("AXIOM_BUNDLE_SECRET is not set (required for env-secret lock)")
    return hashlib.sha256(secret.encode("utf-8")).digest()


def derive_aes256_key(mode: str | LockMode) -> bytes:
    """Derive 32-byte AES key for the current environment."""
    m = mode.value if isinstance(mode, LockMode) else str(mode).lower().strip()
    if m == LockMode.NONE.value:
        raise BundleLockError("cannot derive key for lock mode none")
    if m == LockMode.DEVICE.value:
        return _derive_key_device()
    if m == LockMode.HOST.value:
        return _derive_key_host()
    if m == LockMode.ENV_SECRET.value:
        return _derive_key_env_secret()
    raise BundleLockError(f"unknown lock mode: {mode!r}")


def _derive_key_for_save(mode: str | LockMode) -> bytes:
    """Like :func:`derive_aes256_key` but use :class:`BundleLockError` for missing env at save time."""
    m = mode.value if isinstance(mode, LockMode) else str(mode).lower().strip()
    if m == LockMode.DEVICE.value:
        return _derive_key_device()
    if m == LockMode.HOST.value:
        return _derive_key_host()
    if m == LockMode.ENV_SECRET.value:
        secret = os.environ.get("AXIOM_BUNDLE_SECRET", "").strip()
        if not secret:
            raise BundleLockError("AXIOM_BUNDLE_SECRET must be set to create env-secret lock")
        return hashlib.sha256(secret.encode("utf-8")).digest()
    raise BundleLockError(f"unknown lock mode: {mode!r}")


def _neural_weights_to_bytes(nw: Optional[Dict[str, Any]]) -> bytes:
    buf = io.BytesIO()
    torch.save(nw, buf)
    return buf.getvalue()


def _bytes_to_neural_weights(raw: bytes) -> Optional[Dict[str, Any]]:
    buf = io.BytesIO(raw)
    try:
        obj = torch.load(buf, map_location="cpu", weights_only=True)
    except TypeError:
        buf.seek(0)
        obj = torch.load(buf, map_location="cpu")
    if obj is None:
        return None
    if isinstance(obj, dict) and not obj:
        return None
    return obj


def _aes_ctr_encrypt(key: bytes, plaintext: bytes) -> tuple[bytes, bytes]:
    _require_crypto()
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    if len(key) != 32:
        raise ValueError("AES-256 requires 32-byte key")
    nonce = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    enc = cipher.encryptor()
    ct = enc.update(plaintext) + enc.finalize()
    return ct, nonce


def _aes_ctr_decrypt(key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    _require_crypto()
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    cipher = Cipher(algorithms.AES(key), modes.CTR(nonce), backend=default_backend())
    dec = cipher.decryptor()
    return dec.update(ciphertext) + dec.finalize()


def apply_lock_to_payload(payload: Dict[str, Any], mode: str | LockMode) -> Dict[str, Any]:
    """Encrypt ``neural_weights`` in place; set ``lock`` metadata and clear plaintext weights."""
    m = mode.value if isinstance(mode, LockMode) else str(mode).lower().strip()
    if m == LockMode.NONE.value:
        return payload
    raw = _neural_weights_to_bytes(payload.get("neural_weights"))
    key = _derive_key_for_save(m)
    fp = _fingerprint_key(key)
    ct, nonce = _aes_ctr_encrypt(key, raw)
    payload["neural_weights"] = None
    payload["lock"] = {
        "encrypted": True,
        "lock_mode": m,
        "nonce_hex": nonce.hex(),
        "payload_len": len(raw),
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "ciphertext_sha256": hashlib.sha256(ct).hexdigest(),
        "key_fingerprint": fp,
        "ciphertext_hex": ct.hex(),
    }
    return payload


def unlock_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """If payload is locked, decrypt ``neural_weights``; otherwise return unchanged."""
    lock = payload.get("lock")
    if not lock or not lock.get("encrypted"):
        return payload
    _require_crypto()
    mode = lock.get("lock_mode")
    if not mode:
        raise BundleUnlockError("locked bundle missing lock_mode")
    key = derive_aes256_key(mode)
    fp = _fingerprint_key(key)
    if fp != lock.get("key_fingerprint"):
        raise BundleUnlockError(
            "bundle key fingerprint mismatch — wrong AXIOM_BUNDLE_SECRET, host, or GPU"
        )
    nonce = bytes.fromhex(lock["nonce_hex"])
    ct = bytes.fromhex(lock["ciphertext_hex"])
    expected_ct = lock.get("ciphertext_sha256")
    if expected_ct and hashlib.sha256(ct).hexdigest() != str(expected_ct):
        raise BundleUnlockError("locked bundle ciphertext tampered (hash mismatch)")
    raw = _aes_ctr_decrypt(key, nonce, ct)
    if len(raw) != int(lock.get("payload_len", -1)):
        raise BundleUnlockError("decrypted length mismatch (corrupt bundle)")
    expected_raw = lock.get("payload_sha256")
    if expected_raw and hashlib.sha256(raw).hexdigest() != str(expected_raw):
        raise BundleUnlockError("locked bundle plaintext tampered (hash mismatch)")
    payload["neural_weights"] = _bytes_to_neural_weights(raw)
    return payload


def _missing_bundle_msg(src: Path) -> str:
    abs_p = Path(src).expanduser().resolve()
    return (
        f"Input bundle not found: {src}\n"
        f"Resolved path: {abs_p}\n"
        "Repository `examples/*.axb` files are gitignored — generate a bundle first, "
        "e.g. `python examples/train_portfolio.py` (writes `examples/portfolio_trained.axb`)."
    )


def lock_bundle_file(src: Path, dst: Path, mode: str | LockMode) -> None:
    """Load an unlocked ``.axb``, apply lock, write ``dst``."""
    _require_crypto()
    if not src.is_file():
        raise FileNotFoundError(_missing_bundle_msg(src))
    from axiom.compiler.deserializer import _read_bundle_payload

    payload = _read_bundle_payload(src, trusted=True)
    if payload.get("lock", {}).get("encrypted"):
        raise ValueError("bundle is already locked")
    m = mode.value if isinstance(mode, LockMode) else str(mode).lower().strip()
    if m == LockMode.NONE.value:
        raise BundleLockError("use lock mode device, host, or env-secret for lock-bundle")
    apply_lock_to_payload(payload, m)
    dst = Path(dst)
    if str(dst.parent) not in ("", "."):
        dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    wsrc = Path(str(src) + ".weights.pt")
    wdst = Path(str(dst) + ".weights.pt")
    if wsrc.is_file():
        wdst.write_bytes(wsrc.read_bytes())
    elif wdst.is_file():
        wdst.unlink()
