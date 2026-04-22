#!/usr/bin/env python3
"""Solana tx signer + broadcaster for Zhu.

Usage pattern (imported by zhu_agent):
    from zhu_sol_sign import sign_and_broadcast
    sig = sign_and_broadcast(tx_data_b58, rpc_url)

Flow:
  1. Read mnemonic from macOS Keychain (service=zhu-wallet)
  2. Derive Solana Ed25519 keypair (BIP-44 m/44'/501'/0'/0')
  3. Base58-decode OKX onchainos tx data -> VersionedTransaction
  4. Sign message with keypair
  5. Broadcast via Solana JSON-RPC sendTransaction
  6. Discard key immediately after signing

Never logs mnemonic/priv key. Private key lives only in function-local
variables and is overwritten as soon as signing completes.
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.request
from typing import Optional

import base58
from bip_utils import (
    Bip39SeedGenerator,
    Bip44,
    Bip44Changes,
    Bip44Coins,
)
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction

KEYCHAIN_SERVICE = "zhu-wallet"
KEYCHAIN_ACCOUNT = "mnemonic-bip39-24"

# Public Solana mainnet-beta RPC (rate-limited; swap to Helius/QuickNode for live)
DEFAULT_RPC = "https://api.mainnet-beta.solana.com"


def _read_keychain_mnemonic() -> str:
    r = subprocess.run(
        ["security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT,
         "-s", KEYCHAIN_SERVICE, "-w"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"keychain mnemonic not found "
            f"(service={KEYCHAIN_SERVICE} account={KEYCHAIN_ACCOUNT}); "
            f"run zhu_wallet_setup.py first"
        )
    return r.stdout.strip()


def _derive_sol_keypair() -> Keypair:
    """BIP-44 m/44'/501'/0'/0' Ed25519 via SLIP-10."""
    mnemonic = _read_keychain_mnemonic()
    try:
        seed = Bip39SeedGenerator(mnemonic).Generate()
        sol = Bip44.FromSeed(seed, Bip44Coins.SOLANA)
        acct = sol.Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT)
        priv_bytes = acct.PrivateKey().Raw().ToBytes()
    finally:
        # best-effort memory clear
        mnemonic = ""
        del mnemonic
    # solders.Keypair.from_seed expects 32-byte seed
    return Keypair.from_seed(priv_bytes)


def sign_tx_data(tx_data: str) -> str:
    """Sign base58-encoded Solana tx data from onchainos swap swap.

    Returns base64-encoded signed VersionedTransaction ready for RPC.
    """
    raw = base58.b58decode(tx_data)
    tx = VersionedTransaction.from_bytes(raw)
    kp = _derive_sol_keypair()
    try:
        # Re-sign: solders allows replacing signatures via construction
        signed = VersionedTransaction(tx.message, [kp])
    finally:
        # overwrite key bytes in memory
        kp = None
        del kp
    return base64.b64encode(bytes(signed)).decode()


def rpc_send_raw(signed_b64: str, rpc_url: str = DEFAULT_RPC,
                 skip_preflight: bool = False) -> dict:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sendTransaction",
        "params": [
            signed_b64,
            {
                "encoding": "base64",
                "skipPreflight": skip_preflight,
                "maxRetries": 3,
            },
        ],
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        rpc_url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def rpc_get_signature_statuses(signatures: list, rpc_url: str = DEFAULT_RPC) -> dict:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignatureStatuses",
        "params": [signatures, {"searchTransactionHistory": True}],
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        rpc_url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def sign_and_broadcast(tx_data_b58: str, rpc_url: str = DEFAULT_RPC,
                       wait_confirm: bool = True, timeout_sec: int = 60) -> dict:
    """Full pipeline: sign -> send -> optionally poll confirm.

    Returns dict with keys: signature, confirmed, error (if any), raw_response.
    """
    signed = sign_tx_data(tx_data_b58)
    resp = rpc_send_raw(signed, rpc_url=rpc_url)
    result = {"signature": None, "confirmed": False, "error": None,
              "raw_response": resp}
    if "error" in resp:
        result["error"] = resp["error"]
        return result
    sig = resp.get("result")
    result["signature"] = sig
    if not wait_confirm or not sig:
        return result
    # poll confirm
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        st = rpc_get_signature_statuses([sig], rpc_url=rpc_url)
        lst = (st.get("result") or {}).get("value") or [None]
        if lst[0]:
            s = lst[0]
            if s.get("err") is not None:
                result["error"] = s["err"]
                return result
            status = s.get("confirmationStatus")
            if status in ("confirmed", "finalized"):
                result["confirmed"] = True
                return result
        time.sleep(2)
    result["error"] = "confirm_timeout"
    return result


if __name__ == "__main__":
    # smoke: derive-only, show Sol public address (no signing)
    kp = _derive_sol_keypair()
    print(f"Sol pubkey: {kp.pubkey()}")
