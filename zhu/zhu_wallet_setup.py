#!/usr/bin/env python3
"""Zhu Wallet Setup — one-shot Solana wallet creation via BIP-39.

Generates 24-word mnemonic, derives Solana + EVM addresses, stores
mnemonic in macOS Keychain, copies mnemonic to clipboard for 60s so
user can back up to password manager / paper.

Safety rules (non-negotiable):
- Mnemonic never written to stdout, stderr, log file, env var, or chat.
- Mnemonic only hits: memory -> Keychain (encrypted) -> clipboard (60s).
- Derived private keys never persisted.

Usage:
    python3 zhu_wallet_setup.py
"""
import os
import subprocess
import sys
import time
from bip_utils import (
    Bip39MnemonicGenerator,
    Bip39WordsNum,
    Bip39SeedGenerator,
    Bip44,
    Bip44Coins,
    Bip44Changes,
    Bip32Slip10Ed25519,
)

KEYCHAIN_SERVICE = "zhu-wallet"
KEYCHAIN_ACCOUNT = "mnemonic-bip39-24"


def keychain_exists():
    r = subprocess.run(
        ["security", "find-generic-password", "-a", KEYCHAIN_ACCOUNT, "-s", KEYCHAIN_SERVICE],
        capture_output=True,
    )
    return r.returncode == 0


def keychain_write(mnemonic: str):
    subprocess.run(
        [
            "security",
            "add-generic-password",
            "-a",
            KEYCHAIN_ACCOUNT,
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
            mnemonic,
            "-U",
        ],
        check=True,
        capture_output=True,
    )


def derive_solana_address(mnemonic: str) -> str:
    seed = Bip39SeedGenerator(mnemonic).Generate()
    sol = Bip44.FromSeed(seed, Bip44Coins.SOLANA)
    acct = sol.Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT)
    return acct.PublicKey().ToAddress()


def derive_evm_address(mnemonic: str) -> str:
    seed = Bip39SeedGenerator(mnemonic).Generate()
    evm = Bip44.FromSeed(seed, Bip44Coins.ETHEREUM)
    addr_obj = evm.Purpose().Coin().Account(0).Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
    return addr_obj.PublicKey().ToAddress()


def pbcopy(text: str):
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(text.encode())


def pbcopy_clear_after(seconds: int):
    """Fork a background process to clear clipboard after N seconds."""
    subprocess.Popen(
        ["bash", "-c", f"sleep {seconds}; echo -n '' | pbcopy"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main():
    if keychain_exists():
        print("Keychain entry already exists:")
        print(f"  service={KEYCHAIN_SERVICE} account={KEYCHAIN_ACCOUNT}")
        print("Refusing to overwrite. If you want fresh wallet:")
        print("  security delete-generic-password "
              f"-a {KEYCHAIN_ACCOUNT} -s {KEYCHAIN_SERVICE}")
        sys.exit(1)

    print("Generating 24-word BIP-39 mnemonic...")
    mnemonic = str(Bip39MnemonicGenerator().FromWordsNumber(Bip39WordsNum.WORDS_NUM_24))

    sol_addr = derive_solana_address(mnemonic)
    evm_addr = derive_evm_address(mnemonic)

    print("Writing mnemonic to macOS Keychain...")
    keychain_write(mnemonic)

    print("Copying mnemonic to clipboard (will auto-clear in 60s)...")
    pbcopy(mnemonic)
    pbcopy_clear_after(60)

    # Discard mnemonic from memory
    mnemonic = None
    del mnemonic

    print()
    print("=" * 60)
    print("WALLET CREATED")
    print("=" * 60)
    print(f"Solana address: {sol_addr}")
    print(f"EVM address:    {evm_addr}")
    print()
    print("ACTIONS REQUIRED NOW (within 60s before clipboard clears):")
    print("  1. Paste clipboard into 1Password / Bitwarden / paper backup")
    print("  2. Do NOT paste into chat / code / email")
    print()
    print("Mnemonic now lives in:")
    print(f"  - macOS Keychain (service={KEYCHAIN_SERVICE})")
    print(f"  - your password manager / paper (after step 1 above)")
    print()
    print(f"Fund this Solana address to start: {sol_addr}")
    print()
    print("To retrieve mnemonic later:")
    print(f"  security find-generic-password -a {KEYCHAIN_ACCOUNT} -s {KEYCHAIN_SERVICE} -w")


if __name__ == "__main__":
    main()
