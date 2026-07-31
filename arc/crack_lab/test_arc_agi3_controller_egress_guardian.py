from __future__ import annotations

import hashlib
import socket
import ssl

import pytest

import arc_agi3_controller_egress_guardian as G


def _client_hello(host: str) -> bytes:
    incoming = ssl.MemoryBIO()
    outgoing = ssl.MemoryBIO()
    context = ssl.create_default_context()
    connection = context.wrap_bio(
        incoming, outgoing, server_side=False, server_hostname=host
    )
    with pytest.raises(ssl.SSLWantReadError):
        connection.do_handshake()
    return outgoing.read()


def test_egress_policy_and_client_hello_sni_are_exact():
    policy = G.policy_document()
    assert policy["policy"] == "openai_https_only"
    assert policy["allowed_sni"] == list(G.ALLOWED_SNI)
    assert G.policy_sha256() == hashlib.sha256(
        G.canonical_json(policy)
    ).hexdigest()
    assert G._tls_sni(_client_hello("api.openai.com")) == (
        "api.openai.com"
    )
    assert G._tls_sni(_client_hello("example.com")) == "example.com"
    assert G._tls_sni(b"GET / HTTP/1.1\r\n\r\n") is None


def test_egress_resolution_rejects_one_non_global_answer(monkeypatch):
    monkeypatch.setattr(
        G.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("203.0.113.9", 443),
            )
        ],
    )
    with pytest.raises(G.EgressGuardianError, match="non-global"):
        G._global_ipv4("api.openai.com")


def test_runtime_resolver_allowlist_is_exact_and_ipv4_only(tmp_path):
    resolv = tmp_path / "resolv.conf"
    resolv.write_text(
        "nameserver 127.0.0.11\n"
        "nameserver 192.0.2.53 # exact runtime resolver\n"
        "nameserver ::1\n",
        encoding="ascii",
    )
    assert G._resolver_ipv4s(resolv) == (
        "127.0.0.11",
        "192.0.2.53",
    )
    resolv.write_text("nameserver not-an-ip\n", encoding="ascii")
    with pytest.raises(G.EgressGuardianError, match="malformed"):
        G._resolver_ipv4s(resolv)


def test_default_deny_is_installed_by_two_atomic_restores(monkeypatch):
    restores: list[tuple[tuple[str, ...], str]] = []
    saves: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        G,
        "_run_exact_input",
        lambda argv, payload: restores.append((argv, payload)),
    )

    def save(argv):
        saves.append(argv)
        return "frozen rules\n"

    monkeypatch.setattr(G, "_run_exact", save)
    first, second = G.install_default_deny(("127.0.0.11",))

    assert [item[0][0] for item in restores] == [
        "/usr/sbin/iptables-restore",
        "/usr/sbin/ip6tables-restore",
    ]
    assert saves == [
        ("/usr/sbin/iptables-save",),
        ("/usr/sbin/ip6tables-save",),
    ]
    ipv4 = restores[0][1]
    ipv6 = restores[1][1]
    assert ":OUTPUT DROP [0:0]" in ipv4
    assert "--uid-owner 65532" in ipv4
    assert "-d 127.0.0.11 --dport 53" in ipv4
    assert "--dport 443 -j REDIRECT --to-ports 19443" in ipv4
    assert "--uid-owner 65532" not in ipv6
    expected = hashlib.sha256(b"frozen rules\n").hexdigest()
    assert (first, second) == (expected, expected)


def test_live_probe_fails_closed_when_allow_or_deny_check_is_wrong(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        G,
        "_probe_tls",
        lambda host, *, should_succeed: host == G.DENIED_PROBE_SNI,
    )
    monkeypatch.setattr(
        G.socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("denied")
        ),
    )
    assert G.run_probe("a" * 64) == 1
    assert '"status":"FAIL"' in capsys.readouterr().out
