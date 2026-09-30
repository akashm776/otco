"""Stream child output to console and a separate, closed-on-exit Drive log."""
from collections import deque
import os
from pathlib import Path
import subprocess


def run_logged(cmd, cwd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=80)
    # Never overwrite an existing diagnostic transcript.
    with log_path.open('x', encoding='utf-8', buffering=1) as log:
        env = {**os.environ, 'OTCO_POLICY_LOG_PATH': str(log_path)}
        with subprocess.Popen(cmd, cwd=cwd, env=env, text=True, encoding='utf-8', errors='replace',
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1) as child:
            try:
                for line in child.stdout:
                    log.write(line)
                    tail.append(line)
                    print(line, end='', flush=True)
                code = child.wait()
                log.write(f'\nCHILD_EXIT_CODE: {code}\n')
                log.flush()
                os.fsync(log.fileno())
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                raise
    if code:
        print('\nFAILED SUBPROCESS TAIL:\n'+''.join(tail), flush=True)
        print('FULL_SUBPROCESS_LOG:', log_path, flush=True)
        raise subprocess.CalledProcessError(code, cmd, output=''.join(tail))
    return code
