"""Fail-closed Linux code evaluator: dropped UID, Landlock, seccomp, limits.

Never executes a candidate unless all isolation layers are installed. This
protects host resources; it is not an adversarial anti-cheating certificate.
"""
import argparse
import ctypes
import ctypes.util
import errno
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import uuid


def prepare_stdlib(destination='/tmp/dp-paper-safe-stdlib-20260917'):
    target = Path(destination)
    if not str(target).startswith('/tmp/dp-paper-safe-stdlib-'):
        raise ValueError('Unexpected runtime destination')
    marker = target / 'runtime.json'
    version = {'python': sys.version, 'stdlib': sysconfig.get_path('stdlib')}
    if marker.exists():
        if json.loads(marker.read_text()) != version or target.stat().st_uid != os.getuid():
            raise RuntimeError('Public runtime identity mismatch')
        return target / 'stdlib'
    target.mkdir(mode=0o755, exist_ok=False)
    shutil.copytree(version['stdlib'], target / 'stdlib',
        ignore=shutil.ignore_patterns('site-packages', 'test', 'tests', '__pycache__',
            'tkinter', 'idlelib', 'turtledemo', 'ensurepip'))
    marker.write_text(json.dumps(version))
    return target / 'stdlib'


def checked_libcall(result, label):
    if result < 0:
        raise RuntimeError(f'{label}: errno={ctypes.get_errno()}')
    return result


def isolate(directory, stdlib):
    if sys.platform != 'linux' or os.getuid() != 0 or os.uname().machine != 'x86_64':
        raise RuntimeError('Unsupported platform; refusing candidate execution')
    libc = ctypes.CDLL(None, use_errno=True)
    sec = ctypes.CDLL(ctypes.util.find_library('seccomp') or '', use_errno=True)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_syscall_resolve_name.restype = ctypes.c_int
    sec.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sec.seccomp_rule_add.restype = ctypes.c_int
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_load.restype = ctypes.c_int
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    abi = checked_libcall(libc.syscall(444, 0, 0, 1), 'Landlock ABI')
    if abi < 1:
        raise RuntimeError('Landlock unavailable')
    # ABI1 filesystem bits0..12. Additional supported protections are added
    # when available, never assumed on the currently verified ABI1 hosts.
    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= 1 << 13
    if abi >= 3:
        handled |= 1 << 14
    class Rules(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]
    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    rules = Rules(handled)
    fd = checked_libcall(libc.syscall(444, ctypes.byref(rules), ctypes.sizeof(rules), 0), 'Landlock create')
    readonly = (1 << 2) | (1 << 3)
    writable = readonly | (1 << 1) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8)
    if abi >= 3:
        writable |= 1 << 14
    try:
        for path, rights in [(str(stdlib), readonly), ('/lib', readonly), ('/lib64', readonly),
                ('/usr/lib', readonly), (str(directory), writable),
                ('/etc/ld.so.cache', 1 << 2), ('/dev/null', (1 << 2) | (1 << 1)),
                ('/dev/urandom', 1 << 2)]:
            if not os.path.exists(path):
                continue
            path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = Beneath(rights, path_fd)
                checked_libcall(libc.syscall(445, fd, 1, ctypes.byref(rule), 0), 'Landlock add')
            finally:
                os.close(path_fd)
        checked_libcall(libc.prctl(38, 1, 0, 0, 0), 'no_new_privs')
        checked_libcall(libc.prctl(8, 0, 0, 0, 0), 'disable keepcaps')
        os.setgroups([])
        os.setresgid(65534, 65534, 65534)
        os.setresuid(65534, 65534, 65534)
        if os.geteuid() == 0 or os.getresuid() != (65534, 65534, 65534):
            raise RuntimeError('Privilege drop failed')
        checked_libcall(libc.syscall(446, fd, 0), 'Landlock restrict')
    finally:
        os.close(fd)
    context = sec.seccomp_init(0x7FFF0000)  # ALLOW except denied operations.
    if not context:
        raise RuntimeError('seccomp init failed')
    deny = 0x00050000 | errno.EPERM
    names = ('socket', 'socketpair', 'connect', 'bind', 'listen', 'accept', 'accept4',
        'sendto', 'sendmsg', 'recvfrom', 'recvmsg', 'execve', 'execveat', 'clone', 'clone3',
        'fork', 'vfork', 'ptrace', 'process_vm_readv', 'process_vm_writev', 'kill', 'tkill',
        'tgkill', 'pidfd_send_signal', 'pidfd_getfd', 'mount', 'umount2', 'pivot_root',
        'chroot', 'setns', 'unshare', 'bpf', 'perf_event_open', 'io_uring_setup',
        'io_uring_enter', 'io_uring_register', 'keyctl', 'add_key', 'request_key',
        'open_by_handle_at', 'name_to_handle_at', 'kexec_load', 'reboot', 'swapon',
        'swapoff', 'init_module', 'finit_module', 'delete_module', 'userfaultfd', 'ioctl',
        'setuid', 'setgid', 'setresuid', 'setresgid', 'setreuid', 'setregid', 'setgroups',
        'truncate', 'ftruncate', 'openat2', 'chmod', 'fchmod', 'fchmodat', 'chown',
        'fchown', 'lchown', 'fchownat')
    try:
        for name in names:
            number = sec.seccomp_syscall_resolve_name(name.encode())
            if number < 0:
                if name in ('socket', 'execve', 'clone', 'fork', 'ptrace', 'kill'):
                    raise RuntimeError('Required seccomp syscall not resolved')
                continue
            if sec.seccomp_rule_add(context, deny, number, 0) != 0:
                raise RuntimeError('seccomp rule installation failed')
        # Landlock ABI1 does not mediate all truncation operations. Block
        # O_TRUNC even with O_RDONLY, plus truncate/ftruncate/openat2 above.
        class ArgCmp(ctypes.Structure):
            _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_int),
                        ('datum_a', ctypes.c_uint64), ('datum_b', ctypes.c_uint64)]
        sec.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ArgCmp)]
        sec.seccomp_rule_add_array.restype = ctypes.c_int
        for syscall, argument in [('open', 1), ('openat', 2)]:
            number = sec.seccomp_syscall_resolve_name(syscall.encode())
            if number < 0: raise RuntimeError('Required open syscall not resolved')
            comparison = ArgCmp(argument, 7, os.O_TRUNC, os.O_TRUNC)
            if sec.seccomp_rule_add_array(context, deny, number, 1, ctypes.byref(comparison)) != 0:
                raise RuntimeError('Truncation protection rule installation failed')
        if sec.seccomp_load(context) != 0:
            raise RuntimeError('seccomp enforcement failed')
    finally:
        sec.seccomp_release(context)
    return abi


def worker(directory, stdlib, marker):
    program = (directory / 'candidate.py').read_text()
    write_bytes = os.write
    os.chdir(directory)
    sys.path[:] = [str(stdlib), str(stdlib / 'lib-dynload')]
    sys.dont_write_bytecode = True
    os.environ.clear()
    os.environ.update(PATH='', HOME='/nonexistent', LANG='C', LC_ALL='C')
    resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NPROC, (8, 8))
    try:
        abi = isolate(directory, stdlib)
    except Exception as error:
        write_bytes(2, f'SANDBOX_UNAVAILABLE:{type(error).__name__}:{error}\n'.encode())
        return 72
    try:
        exec(compile(program, 'candidate.py', 'exec'), {'__name__': '__main__'})
    except BaseException as error:
        write_bytes(2, f'CANDIDATE_FAILED:{type(error).__name__}\n'.encode())
        return 1
    write_bytes(1, ('\n' + marker + ':PASSED:ABI' + str(abi) + '\n').encode())
    return 0


def run_program(program, stdlib, timeout=10):
    if sys.platform != 'linux' or os.getuid() != 0:
        return {'passed': None, 'status': 'sandbox_unavailable'}
    marker = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='dp-paper-eval-') as name:
        directory = Path(name)
        source = directory / 'candidate.py'
        source.write_text(program)
        source.chmod(0o400)
        os.chown(source, 65534, 65534)
        os.chown(directory, 65534, 65534)
        directory.chmod(0o700)
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                result = subprocess.run([sys.executable, '-I', str(Path(__file__).resolve()),
                    '--worker', str(directory), '--stdlib', str(stdlib), '--marker', marker],
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, timeout=timeout,
                    close_fds=True, start_new_session=True,
                    env={'PATH': '', 'LANG': 'C', 'LC_ALL': 'C', 'PYTHONHASHSEED': '0'})
            except subprocess.TimeoutExpired:
                return {'passed': False, 'status': 'timeout'}
            stdout.seek(0); stderr.seek(0)
            output = stdout.read(8192).decode(errors='replace')
            stdout.seek(0, os.SEEK_END)
            stdout.seek(max(0, stdout.tell()-8192))
            output += stdout.read(8192).decode(errors='replace')
            error = stderr.read(8192).decode(errors='replace')
        if result.returncode == 72 or 'SANDBOX_UNAVAILABLE:' in error:
            return {'passed': None, 'status': 'sandbox_unavailable', 'error': error}
        return {'passed': result.returncode == 0 and marker + ':PASSED:ABI' in output,
            'status': 'passed' if result.returncode == 0 and marker + ':PASSED:ABI' in output else 'failed',
            'returncode': result.returncode, 'error': error}


def selftest(stdlib):
    with tempfile.TemporaryDirectory(prefix='dp-paper-sentinel-') as name:
        sentinel = Path(name) / 'host-private-marker'
        sentinel.write_text('non-secret synthetic isolation sentinel')
        sentinel.chmod(0o666)
        Path(name).chmod(0o755)
        program = f'''import os, socket, math, typing, collections
assert os.getuid() == 65534
assert math.sqrt(4) == 2
for action in [lambda: open({str(sentinel)!r}).read(), lambda: os.open({str(sentinel)!r}, os.O_RDONLY | os.O_TRUNC), lambda: os.truncate({str(sentinel)!r}, 0), lambda: socket.socket(), lambda: os.fork(), lambda: os.kill(1, 0)]:
    try:
        action()
    except (PermissionError, OSError):
        pass
    else:
        raise AssertionError('isolation failure')
'''
        result = run_program(program, stdlib)
        result['sentinel_not_modified'] = sentinel.read_text() == 'non-secret synthetic isolation sentinel'
    assert_fail = run_program('assert False', stdlib)
    fake_success = run_program('import os; print("PASS"); os._exit(0)', stdlib)
    return {'available': result['passed'] is True and result['sentinel_not_modified'] and assert_fail['passed'] is False and fake_success['passed'] is False,
        'isolation_probe': result, 'assertion_failure_probe': assert_fail, 'early_exit_probe': fake_success,
        'policy': 'UID65534 + no_new_privs + Landlock FS allowlist + seccomp no network/fork/exec/signalling + resource limits',
        'not_a_security_proof': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--stdlib', type=Path)
    parser.add_argument('--marker')
    args = parser.parse_args()
    if args.worker:
        sys.exit(worker(args.worker, args.stdlib, args.marker))
    runtime = prepare_stdlib()
    print(json.dumps(selftest(runtime), indent=2))
