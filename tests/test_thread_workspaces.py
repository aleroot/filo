"""Exercise thread workspace isolation through the real TUI (POSIX, Python 3).

Run: python3 tests/test_thread_workspaces.py /path/to/filo
Uses temporary config/data and a dummy provider; no model requests are made.
"""

import fcntl
import json
import os
import pathlib
import pty
import re
import select
import signal
import struct
import sys
import tempfile
import termios
import time

binary = str(pathlib.Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix='filo-thread-smoke-') as tmp:
    root = pathlib.Path(tmp).resolve()
    first = root / 'mlx-llm'
    second = root / 'filo'
    for path in (first, second, first / 'nested', root / 'config' / 'filo'):
        path.mkdir(parents=True, exist_ok=True)
    (root / 'config' / 'filo' / 'config.json').write_text(json.dumps({
        'default_provider': 'openai', 'default_model_selection': 'manual',
        'providers': {'openai': {'type': 'openai', 'auth_type': 'api_key', 'api_key': 'test-only', 'model': 'gpt-4o-mini', 'base_url': 'http://127.0.0.1:1/v1'}},
    }))
    env = {'PATH': os.environ['PATH'], 'TERM': 'xterm-256color', 'LANG': 'en_US.UTF-8',
           'XDG_CONFIG_HOME': str(root / 'config'), 'XDG_DATA_HOME': str(root / 'data'), 'XDG_CACHE_HOME': str(root / 'cache')}
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(first)
        os.execve(binary, [binary, '--sandbox', 'off'], env)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', 45, 140, 0, 0))
    output = bytearray()
    def pump(duration=.1):
        if select.select([fd], [], [], duration)[0]:
            try:
                data = os.read(fd, 65536)
            except OSError:
                return
            output.extend(data)
            if b'\x1b[6n' in data:
                os.write(fd, b'\x1b[1;1R')
    def wait_for(check, label, timeout=12):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pump()
            result = check()
            if result:
                return result
        raise AssertionError(label + '\n' + re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', output.decode(errors='replace'))[-5000:])
    def command(text):
        print('>', text, flush=True)
        os.write(fd, text.encode())
        # Let the autocomplete render before Enter, as in a terminal.
        end = time.monotonic() + .15
        while time.monotonic() < end:
            pump(.02)
        os.write(fd, b'\r')
    def sessions():
        result = []
        for path in (root / 'data' / 'filo' / 'sessions').glob('*.json'):
            try:
                result.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return result
    def named(name):
        return next((s for s in sessions() if s.get('name') == name), None)
    try:
        wait_for(lambda: b'BUILD' in output or b'gpt-4o-mini' in output, 'startup')
        command('/rename first')
        one = wait_for(lambda: named('first'), 'first session saved')
        command('/new')
        wait_for(lambda: b'New thread started' in output, 'new thread')
        command('/workspace change ' + str(second))
        wait_for(lambda: b'Switched the working directory' in output, 'workspace changed')
        command('/rename second')
        two = wait_for(lambda: named('second'), 'second session saved')
        assert pathlib.Path(two['working_dir']) == second, two
        command('/resume ' + one['session_id'])
        command('!pwd > first-pwd.txt')
        wait_for(lambda: (first / 'first-pwd.txt').exists(), 'shell uses first workspace')
        assert (first / 'first-pwd.txt').read_text().strip() == str(first)
        wait_for(lambda: named('first') and named('first').get('messages'), 'shell history saved')
        assert pathlib.Path(named('first')['working_dir']) == first
        command('/init --with-prompt')
        wait_for(lambda: (first / 'FILO.md').exists(), 'init uses first workspace')
        assert not (second / 'FILO.md').exists()
        command('/new')
        command('/rename third')
        three = wait_for(lambda: named('third'), 'third session saved')
        assert pathlib.Path(three['working_dir']) == first, three
        command('/resume ' + two['session_id'])
        command('!pwd > second-pwd.txt')
        wait_for(lambda: (second / 'second-pwd.txt').exists(), 'shell uses second workspace')
        assert (second / 'second-pwd.txt').read_text().strip() == str(second)
        wait_for(lambda: named('second') and named('second').get('messages'), 'second shell saved')
        command('/resume ' + one['session_id'])
        command('!sleep 1 && pwd > background-pwd.txt')
        command('/resume ' + two['session_id'])
        wait_for(lambda: (first / 'background-pwd.txt').exists(), 'background shell workspace')
        wait_for(lambda: 'background-pwd.txt' in json.dumps(named('first').get('messages')),
                 'background thread saved while another tab is selected')
        assert pathlib.Path(named('first')['working_dir']) == first
        command('/resume ' + one['session_id'])
        command('/workspace change nested')
        command('/rename nested')
        nested = wait_for(lambda: named('nested'), 'relative workspace saved')
        assert pathlib.Path(nested['working_dir']) == first / 'nested', nested
        assert pathlib.Path(named('second')['working_dir']) == second
        assert pathlib.Path(named('third')['working_dir']) == first
        command('/quit')
        def exited_status():
            child, status = os.waitpid(pid, os.WNOHANG)
            return (child, status) if child else None
        exited = wait_for(exited_status, 'clean exit')
        assert exited[1] == 0, exited
        pid = None
        print('PASS: thread creation, switching, relative workspace changes, shell cwd, and saved workspace isolation', flush=True)
    finally:
        if pid is not None:
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except ProcessLookupError:
                pass
        os.close(fd)
