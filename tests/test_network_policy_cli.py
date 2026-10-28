#!/usr/bin/env python3
"""Exercise the opt-in network policy through the actual headless MCP process."""

import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading


class ProbeServer:
    """A listener that answers one HTTP request, so a permitted fetch ends."""

    def __init__(self):
        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(4)
        self.socket.settimeout(0.3)
        self.host, self.port = self.socket.getsockname()
        self.hits = 0
        self.log = []
        self._stop = False
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop:
            try:
                connection, _ = self.socket.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with connection:
                try:
                    connection.settimeout(2.0)
                    data = connection.recv(4096)
                    if not data:
                        # A bare connectivity probe, not a request.
                        continue
                    self.log.append(data[:120])
                    self.hits += 1
                    connection.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                        b"Content-Type: text/plain\r\nConnection: close\r\n\r\nok")
                except OSError:
                    pass

    def close(self):
        self._stop = True
        try:
            self.socket.close()
        except OSError:
            pass


def run_filo(binary, project, env, url, extra_args):
    """Run one headless MCP session and return the parsed responses."""
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                    "clientInfo": {"name": "network-policy-test", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "fetch_url", "arguments": {"url": url}}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "run_terminal_command", "arguments": {
             "command": "curl -s --max-time 2 -X PUT --data-binary probe " + url}}},
    ]
    process = subprocess.run(
        [binary, "--mcp", "stdio", "--headless", "--sandbox", "workspace-write"]
        + extra_args,
        input="".join(json.dumps(value) + "\n" for value in requests),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=project, env=env, timeout=30)
    if process.returncode:
        if ("Requested sandbox backend" in process.stderr
                or ("Sandbox readiness check failed" in process.stderr
                    and "Operation not permitted" in process.stderr)):
            return 77, None
        raise AssertionError("Filo failed to start: " + process.stderr[-1000:])
    responses = [json.loads(line) for line in process.stdout.splitlines()
                 if line.startswith("{")]
    return 0, responses


def open_listener():
    try:
        return ProbeServer()
    except PermissionError:
        return None


def main():
    binary = str(pathlib.Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="filo-network-policy-") as tmp:
        root = pathlib.Path(tmp)
        project = root / "project"
        project.mkdir()
        (root / "config").mkdir()
        env = os.environ.copy()
        env.update(HOME=tmp, XDG_CONFIG_HOME=str(root / "config"),
                   XDG_DATA_HOME=str(root / "data"))

        listener = open_listener()
        if listener is None:
            print("SKIP: local socket binding is unavailable")
            return 77
        host, port = listener.host, listener.port
        # Prove the listener is reachable before asserting that requests fail.
        with socket.create_connection((host, port), timeout=2):
            pass
        url = "http://127.0.0.1:{}/probe".format(port)

        # A policy that asks about everything: the headless process has nobody
        # to ask, so the request is denied and the shell cannot reach the
        # network at all.
        code, responses = run_filo(binary, project, env, url, ["--policy", "network=ask"])
        if code == 77:
            listener.close()
            print("SKIP: native sandbox unavailable in this environment")
            return 77
        fetch = next((item for item in responses if item.get("id") == 2), None)
        shell = next((item for item in responses if item.get("id") == 3), None)
        assert fetch and "approval" in json.dumps(fetch), fetch
        assert shell and shell.get("result", {}).get("structuredContent", {}).get("exit_code", 0) != 0, shell
        try:
            if listener.hits != 0:
                raise AssertionError(
                    "A protected request reached the listener: " + repr(listener.log))
        finally:
            listener.close()

        # A policy file that names the destination authorizes the fetch. The
        # listener never answers, so the call fails after the connection was
        # made: the point is that the request was not blocked.
        listener2 = open_listener()
        if listener2 is None:
            print("SKIP: local socket binding is unavailable")
            return 77
        host2, port2 = listener2.host, listener2.port
        policy = root / "policy.json"
        policy.write_text(json.dumps({
            "version": 1,
            "default": "deny",
            "rules": [{
                "id": "probe",
                "effect": "allow",
                "principal": "web_tool:fetch_url",
                "operation": "read",
                "host": "127.0.0.1",
                "port": port2,
            }],
        }))
        allowed_url = "http://127.0.0.1:{}/probe".format(port2)
        try:
            code, responses = run_filo(binary, project, env, allowed_url,
                                       ["--policy", "network=" + str(policy)])
            if code == 77:
                print("SKIP: native sandbox unavailable in this environment")
                return 77
            fetch = next((item for item in responses if item.get("id") == 2), None)
            assert fetch is not None, fetch
            assert "Network policy" not in json.dumps(fetch), fetch
            assert listener2.hits > 0, "the authorized fetch never reached the listener"
        finally:
            listener2.close()

        # A policy file that cannot be parsed is a startup failure, never a
        # silently unprotected session.
        broken = root / "broken.json"
        broken.write_text("{ not json")
        result = subprocess.run(
            [binary, "--mcp", "stdio", "--headless", "--sandbox", "workspace-write",
             "--policy", "network=" + str(broken)],
            input="", text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=project, env=env, timeout=30)
        assert result.returncode != 0, "a broken policy file was accepted"
        assert "could not be configured" in result.stderr, result.stderr[-500:]
        assert "not valid JSON" in result.stderr, result.stderr[-500:]

        print("Headless fetch and shell upload were blocked; a named destination was allowed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
