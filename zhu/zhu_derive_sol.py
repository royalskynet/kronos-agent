#!/usr/bin/env python3
"""Derive Solana private key from Keychain mnemonic to a temp file.

Writes base58-encoded 64-byte secret key (Phantom / order_sign compatible)
to a fresh tempfile with 0600 perms. Caller passes path via
`--private-key-file-sol` and bgw's order_make_sign_send.py reads + deletes.

Mnemonic never leaves memory outside of this derivation; the tempfile is
the only place the private key touches disk, and only briefly.
"""
import base64
import os
import subprocess
import sys
import tempfile

from bip_utils import (
    Bip39SeedGenerator,
    Bip44,
    Bip44Changes,
    Bip44Coins,
    Base58Encoder,
)

KEYCHAIN_SERVICE = "zhu-wallet"
KEYCHAIN_ACCOUNT = "mnemonic-bip39-24"


def read_keychain_mnemonic() -> str:
    r = subprocess.run(
        ["security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT,
         "-s", KEYCHAIN_SERVICE, "-w"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"ERROR: keychain entry not found "
              f"(service={KEYCHAIN_SERVICE} account={KEYCHAIN_ACCOUNT})",
              file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip()


def derive_sol_keypair_base58(mnemonic: str) -> str:
    """Return base58-encoded 64-byte Solana secret key (privkey32 + pubkey32)."""
    seed = Bip39SeedGenerator(mnemonic).Generate()
    sol = Bip44.FromSeed(seed, Bip44Coins.SOLANA)
    acct = sol.Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT)
    priv32 = acct.PrivateKey().Raw().ToBytes()
    pub32 = acct.PublicKey().RawCompressed().ToBytes()[1:]
    return Base58Encoder.Encode(priv32 + pub32)


def write_temp(secret_b58: str) -> str:
    fd, path = tempfile.mkstemp(prefix="zhu_sol_", suffix=".key")
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, secret_b58.encode())
    finally:
        os.close(fd)
    return path


def main():
    mnemonic = read_keychain_mnemonic()
    secret = derive_sol_keypair_base58(mnemonic)
    # discard mnemonic asap
    mnemonic = None
    del mnemonic
    path = write_temp(secret)
    secret = None
    del secret
    # print only the path to stdout
    print(path)


if __name__ == "__main__":
    main()
