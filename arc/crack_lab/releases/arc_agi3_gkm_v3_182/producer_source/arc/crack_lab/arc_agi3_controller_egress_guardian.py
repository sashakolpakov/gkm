#!/usr/bin/env python3
"""Default-deny transparent TLS egress guardian for the model controller.

The model controller joins this container's network namespace as a non-root
UID.  Netfilter redirects that UID's TCP/443 traffic to this guardian, which
parses the TLS ClientHello and relays only an exact OpenAI/ChatGPT SNI
allowlist.  The guardian itself resolves the admitted name, rejects every
non-global address, and connects upstream as root; all other non-root egress
remains denied by the filter table.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import select
import signal
import socket
import ssl
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Sequence


POLICY_NAME = "openai_https_only"
POLICY_SCHEMA = 1
CONTROLLER_UID = 65532
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 19443
MAX_CLIENT_HELLO_BYTES = 128 * 1024
READINESS_PATH = Path("/run/arc-agi3-egress/readiness.json")
ALLOWED_SNI = (
    "api.openai.com",
    "auth.openai.com",
    "chatgpt.com",
)
DENIED_PROBE_SNI = "example.com"


class EgressGuardianError(RuntimeError):
    """The egress policy could not be installed or proved."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def policy_document() -> dict[str, object]:
    return {
        "schema": POLICY_SCHEMA,
        "kind": "arc_agi3_controller_egress_policy",
        "policy": POLICY_NAME,
        "controller_uid": CONTROLLER_UID,
        "allowed_sni": list(ALLOWED_SNI),
        "allowed_transport": "tcp/443-via-transparent-sni-relay",
        "resolver_egress":
            "udp+tcp/53-to-runtime-resolv-conf-ipv4-only",
        "literal_ip_destinations": "deny",
        "non_global_resolutions": "deny",
        "ipv6_controller_egress": "deny",
        "other_controller_egress": "deny",
    }


def policy_sha256() -> str:
    return hashlib.sha256(canonical_json(policy_document())).hexdigest()


def _run_exact(argv: tuple[str, ...]) -> str:
    completed = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
        env={
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        },
    )
    if completed.returncode != 0:
        raise EgressGuardianError(
            f"policy command failed: {argv[0]} {completed.stderr[:512]!r}"
        )
    return completed.stdout.decode("utf-8", "strict")


def _run_exact_input(argv: tuple[str, ...], payload: str) -> None:
    completed = subprocess.run(
        argv,
        input=payload.encode("ascii"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=20,
        env={
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        },
    )
    if completed.returncode != 0 or completed.stdout:
        raise EgressGuardianError(
            f"policy restore failed: {argv[0]} "
            f"{completed.stderr[:512]!r}"
        )


def _resolver_ipv4s(
    path: Path = Path("/etc/resolv.conf"),
) -> tuple[str, ...]:
    try:
        raw = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise EgressGuardianError(
            "resolver configuration cannot be read"
        ) from exc
    if len(raw) > 16 * 1024:
        raise EgressGuardianError(
            "resolver configuration exceeds its bound"
        )
    result: list[str] = []
    for line in raw.splitlines():
        fields = line.split("#", 1)[0].split(";", 1)[0].split()
        if not fields or fields[0] != "nameserver":
            continue
        if len(fields) != 2:
            raise EgressGuardianError(
                "resolver configuration is malformed"
            )
        try:
            address = ipaddress.ip_address(fields[1])
        except ValueError as exc:
            raise EgressGuardianError(
                "resolver address is malformed"
            ) from exc
        if address.version == 4 and str(address) not in result:
            result.append(str(address))
    if not result or len(result) > 4:
        raise EgressGuardianError(
            "resolver configuration lacks a bounded IPv4 set"
        )
    return tuple(result)


def install_default_deny(
    resolver_ipv4: tuple[str, ...] | None = None,
) -> tuple[str, str]:
    resolvers = (
        _resolver_ipv4s()
        if resolver_ipv4 is None
        else resolver_ipv4
    )
    try:
        valid_resolvers = (
            bool(resolvers)
            and len(resolvers) <= 4
            and len(set(resolvers)) == len(resolvers)
            and all(
                ipaddress.ip_address(address).version == 4
                and str(ipaddress.ip_address(address)) == address
                for address in resolvers
            )
        )
    except (TypeError, ValueError):
        valid_resolvers = False
    if not valid_resolvers:
        raise EgressGuardianError("resolver allowlist is malformed")
    resolver_rules = "\n".join(
        f"-A OUTPUT -m owner --uid-owner {uid} -p {protocol} "
        f"-d {address} --dport 53 -j ACCEPT"
        for uid in (0, CONTROLLER_UID)
        for protocol in ("udp", "tcp")
        for address in resolvers
    )
    ipv4_policy = f"""*filter
:INPUT DROP [0:0]
:FORWARD DROP [0:0]
:OUTPUT DROP [0:0]
-A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
-A INPUT -i lo -j ACCEPT
-A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
-A OUTPUT -o lo -m owner --uid-owner 0 -j ACCEPT
{resolver_rules}
-A OUTPUT -m owner --uid-owner 0 -p tcp --dport 443 -j ACCEPT
-A OUTPUT -m owner --uid-owner {CONTROLLER_UID} -p tcp -d {LISTEN_HOST} --dport {LISTEN_PORT} -j ACCEPT
COMMIT
*nat
:PREROUTING ACCEPT [0:0]
:INPUT ACCEPT [0:0]
:OUTPUT ACCEPT [0:0]
:POSTROUTING ACCEPT [0:0]
-A OUTPUT -m owner --uid-owner {CONTROLLER_UID} -p tcp --dport 443 -j REDIRECT --to-ports {LISTEN_PORT}
COMMIT
"""
    ipv6_policy = """*filter
:INPUT DROP [0:0]
:FORWARD DROP [0:0]
:OUTPUT DROP [0:0]
-A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
-A INPUT -i lo -j ACCEPT
-A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
-A OUTPUT -o lo -m owner --uid-owner 0 -j ACCEPT
COMMIT
"""
    _run_exact_input(
        ("/usr/sbin/iptables-restore", "--wait", "10"),
        ipv4_policy,
    )
    _run_exact_input(
        ("/usr/sbin/ip6tables-restore", "--wait", "10"),
        ipv6_policy,
    )
    ipv4 = _run_exact(("/usr/sbin/iptables-save",))
    ipv6 = _run_exact(("/usr/sbin/ip6tables-save",))
    return (
        hashlib.sha256(ipv4.encode("utf-8")).hexdigest(),
        hashlib.sha256(ipv6.encode("utf-8")).hexdigest(),
    )


def _tls_sni(buffer: bytes) -> str | None:
    if len(buffer) < 5 or buffer[0] != 22:
        return None
    record_length = int.from_bytes(buffer[3:5], "big")
    if record_length + 5 > len(buffer):
        return None
    body = memoryview(buffer)[5 : 5 + record_length]
    if len(body) < 4 or body[0] != 1:
        return None
    hello_length = int.from_bytes(body[1:4], "big")
    if hello_length + 4 > len(body):
        return None
    offset = 4 + 2 + 32
    if offset >= len(body):
        return None
    session_length = body[offset]
    offset += 1 + session_length
    if offset + 2 > len(body):
        return None
    cipher_length = int.from_bytes(body[offset : offset + 2], "big")
    offset += 2 + cipher_length
    if offset >= len(body):
        return None
    compression_length = body[offset]
    offset += 1 + compression_length
    if offset + 2 > len(body):
        return None
    extensions_length = int.from_bytes(body[offset : offset + 2], "big")
    offset += 2
    end = offset + extensions_length
    if end > len(body):
        return None
    while offset + 4 <= end:
        extension_type = int.from_bytes(body[offset : offset + 2], "big")
        extension_length = int.from_bytes(
            body[offset + 2 : offset + 4], "big"
        )
        offset += 4
        extension_end = offset + extension_length
        if extension_end > end:
            return None
        if extension_type == 0:
            extension = body[offset:extension_end]
            if len(extension) < 5:
                return None
            names_length = int.from_bytes(extension[0:2], "big")
            cursor = 2
            names_end = min(len(extension), cursor + names_length)
            while cursor + 3 <= names_end:
                name_type = extension[cursor]
                name_length = int.from_bytes(
                    extension[cursor + 1 : cursor + 3], "big"
                )
                cursor += 3
                if cursor + name_length > names_end:
                    return None
                if name_type == 0:
                    try:
                        name = bytes(
                            extension[cursor : cursor + name_length]
                        ).decode("ascii")
                    except UnicodeDecodeError:
                        return None
                    canonical = name.rstrip(".").lower()
                    if (
                        not canonical
                        or len(canonical) > 253
                        or canonical != name
                    ):
                        return None
                    return canonical
                cursor += name_length
            return None
        offset = extension_end
    return None


def _read_client_hello(client: socket.socket) -> bytes:
    client.settimeout(10)
    data = bytearray()
    while len(data) < MAX_CLIENT_HELLO_BYTES:
        block = client.recv(min(16384, MAX_CLIENT_HELLO_BYTES - len(data)))
        if not block:
            break
        data.extend(block)
        if len(data) >= 5:
            required = 5 + int.from_bytes(data[3:5], "big")
            if len(data) >= required:
                break
    return bytes(data)


def _global_ipv4(host: str) -> tuple[str, ...]:
    try:
        rows = socket.getaddrinfo(
            host,
            443,
            family=socket.AF_INET,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        raise EgressGuardianError("allowed SNI cannot be resolved") from exc
    result: list[str] = []
    for row in rows:
        address = row[4][0]
        parsed = ipaddress.ip_address(address)
        if not parsed.is_global or parsed.version != 4:
            raise EgressGuardianError(
                "allowed SNI resolved to a non-global address"
            )
        if address not in result:
            result.append(address)
    if not result:
        raise EgressGuardianError("allowed SNI has no global IPv4 address")
    return tuple(result)


def _relay(first: socket.socket, second: socket.socket) -> None:
    sockets = (first, second)
    while True:
        readable, _, _ = select.select(sockets, (), (), 60)
        if not readable:
            return
        for source in readable:
            target = second if source is first else first
            block = source.recv(65536)
            if not block:
                return
            target.sendall(block)


def _serve_client(client: socket.socket) -> None:
    upstream: socket.socket | None = None
    try:
        hello = _read_client_hello(client)
        host = _tls_sni(hello)
        if host not in ALLOWED_SNI:
            return
        last_error: OSError | None = None
        for address in _global_ipv4(host):
            candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            candidate.settimeout(10)
            try:
                candidate.connect((address, 443))
                upstream = candidate
                break
            except OSError as exc:
                last_error = exc
                candidate.close()
        if upstream is None:
            raise EgressGuardianError(
                f"allowed upstream connection failed: {last_error!r}"
            )
        upstream.sendall(hello)
        _relay(client, upstream)
    except (OSError, EgressGuardianError):
        return
    finally:
        client.close()
        if upstream is not None:
            upstream.close()


def _write_readiness(value: dict[str, object]) -> None:
    READINESS_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = canonical_json(value) + b"\n"
    descriptor = os.open(
        READINESS_PATH,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _probe_tls(host: str, *, should_succeed: bool) -> bool:
    context = ssl.create_default_context()
    try:
        with socket.create_connection((host, 443), timeout=10) as raw:
            with context.wrap_socket(raw, server_hostname=host) as secured:
                secured.do_handshake()
        succeeded = True
    except (OSError, ssl.SSLError):
        succeeded = False
    return succeeded is should_succeed


def run_probe(nonce: str) -> int:
    checks = {
        "allowed_openai_tls": _probe_tls(
            ALLOWED_SNI[0], should_succeed=True
        ),
        "denied_unallowlisted_sni": _probe_tls(
            DENIED_PROBE_SNI, should_succeed=False
        ),
    }
    for label, host, port in (
        ("denied_loopback", "127.0.0.1", 80),
        ("denied_metadata", "169.254.169.254", 80),
    ):
        try:
            connection = socket.create_connection((host, port), timeout=1)
        except OSError:
            checks[label] = True
        else:
            checks[label] = False
            connection.close()
    body = {
        "schema": 1,
        "kind": "arc_agi3_controller_egress_live_probe",
        "policy": POLICY_NAME,
        "policy_sha256": policy_sha256(),
        "nonce": nonce,
        "uid": os.getuid(),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    print(canonical_json(body).decode("ascii"), flush=True)
    return 0 if body["status"] == "PASS" else 1


def run_guardian(args: argparse.Namespace) -> int:
    expected_policy = policy_sha256()
    if (
        args.policy != POLICY_NAME
        or args.policy_sha256 != expected_policy
        or os.getuid() != 0
    ):
        raise EgressGuardianError("egress guardian launch binding differs")
    for value in (
        args.campaign_id,
        args.generation_id,
        args.attempt_id,
    ):
        parsed = uuid.UUID(value)
        if str(parsed) != value:
            raise EgressGuardianError("egress guardian identity is malformed")
    if (
        len(args.readiness_nonce) != 64
        or any(character not in "0123456789abcdef"
               for character in args.readiness_nonce)
    ):
        raise EgressGuardianError("egress readiness nonce is malformed")
    resolver_ipv4 = _resolver_ipv4s()
    ipv4_sha256, ipv6_sha256 = install_default_deny(
        resolver_ipv4
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((LISTEN_HOST, LISTEN_PORT))
    listener.listen(64)
    listener.settimeout(1)
    readiness = {
        "schema": 1,
        "kind": "arc_agi3_controller_egress_readiness",
        "status": "READY",
        "campaign_id": args.campaign_id,
        "generation_id": args.generation_id,
        "attempt_id": args.attempt_id,
        "policy": POLICY_NAME,
        "policy_sha256": expected_policy,
        "readiness_nonce": args.readiness_nonce,
        "guardian_pid": os.getpid(),
        "controller_uid": CONTROLLER_UID,
        "listen": f"{LISTEN_HOST}:{LISTEN_PORT}",
        "iptables_rules_sha256": ipv4_sha256,
        "ip6tables_rules_sha256": ipv6_sha256,
        "allowed_sni": list(ALLOWED_SNI),
        "resolver_ipv4": list(resolver_ipv4),
        "default_deny_installed": True,
    }
    _write_readiness(readiness)
    stop = threading.Event()
    for selected in (signal.SIGTERM, signal.SIGINT):
        signal.signal(selected, lambda *_args: stop.set())
    while not stop.is_set():
        try:
            client, _address = listener.accept()
        except socket.timeout:
            continue
        thread = threading.Thread(
            target=_serve_client, args=(client,), daemon=True
        )
        thread.start()
    listener.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--print-policy-sha256", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--policy", default=POLICY_NAME)
    parser.add_argument("--policy-sha256")
    parser.add_argument("--campaign-id")
    parser.add_argument("--generation-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--readiness-nonce")
    args = parser.parse_args(argv)
    if args.print_policy_sha256:
        print(policy_sha256())
        return 0
    if args.probe:
        if not isinstance(args.readiness_nonce, str):
            raise EgressGuardianError("probe nonce is required")
        return run_probe(args.readiness_nonce)
    required = (
        args.policy_sha256,
        args.campaign_id,
        args.generation_id,
        args.attempt_id,
        args.readiness_nonce,
    )
    if any(not isinstance(item, str) or not item for item in required):
        raise EgressGuardianError("egress guardian binding is incomplete")
    return run_guardian(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EgressGuardianError as error:
        print(f"egress guardian failed: {error}", file=sys.stderr)
        raise SystemExit(70)
