# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""A Python safety net for accidental device and foreign rig endpoint access.

hermetic.py arms this for tests not marked `hardware`; refusals are recorded in
HERMETIC_WORK/trips and fail the test. It checks the routes below on POSIX and
Windows, with platform-specific wrappers. It is not a sandbox against deliberate
bypass.

Catches:
- Python open/mkdir audit events check real paths for unowned serial devices,
  other /dev entries outside TEST_DEVICES and the verified tty0tty exception,
  and the operator's state directory, and refuse any write under the captures
  directory the example station names in the operator's home. POSIX os.open
  also checks relative paths against dir_fd. Test-created PTYs and verified fixture COM ports are allowed.
- Audited Popen, exec and posix_spawn launches check supported CAT/media tool
  device arguments, including child cwd and the JSON exec trampoline. Opaque
  shells/scripts/wrappers require explicit admission, except exactly POSIX
  [bash/sh, '-n', <absolute path to an existing regular file>] syntax checks
  whose real path passes the device check, without startup-file environment settings;
  os.system is refused unless its exact command string is admitted.
  On POSIX, `strace -- argv` with only the coexist observer's options is
  checked as argv launched directly, and its trace file as a write. A script
  standing in for a supported tool runs only when its real path is admitted.
  Python children must keep this interpreter, site startup, guard path and test
  identity; sitecustomize reinstalls the guard, including for accepted Python
  shebang scripts. An explicitly admitted child of this same interpreter with
  the full admitted argv and devices may start without the guard's startup
  path, and with exactly [-S, ...] site-free; that child and its descendants
  do not inherit the guard.
- The default sounddevice module trips on use. A scripted sounddevice.py may
  precede it only when its real file is inside this test's HERMETIC_WORK;
  device checks and protection of the startup modules still apply.
- Windows pyserial Serial.open and kernel32 CreateFileW/CreateFileA through
  patched ctypes.WinDLL, windll and cached hfmodem APIs check COM ownership,
  including attribute and indexed function lookup.
- Audited socket connections to 4532, 8300, 8301 and 8515 reject non-loopback
  hosts and foreign loopback listeners. Matching live test endpoint manifests
  authorize access; successful guarded binds track socket lifetime, and an
  owned rigctld registers its endpoint for its own lifetime (POSIX: the exec
  carries the manifest lock; Windows: a manifest keyed to the created process).
  An absent listener is allowed.

Does not cover:
- Deliberate bypass of Python hooks/wrappers, saved unwrapped callables, or raw
  syscalls through native extensions and other native code.
- Descriptors passed in from outside, processes started outside guarded spawn
  routes, or native descendants of admitted programs and trusted native tools.
- Races after path/endpoint checks, including a foreign listener appearing
  after the probe or before an owned rigctld has bound its registered endpoint.
- `os.mkdir(dir_fd=...)` with a relative path.
- UDP connections, which are not checked against foreign listeners.
"""
import glob
import ipaddress
import json
import ntpath
import os
import re
import shutil
import shlex
import socket
import stat
import subprocess
import sys
import threading
import weakref

TEST_DEVICES = re.compile(r"/dev/(null|zero|urandom|ptmx|std(in|out|err)|fd/\d+)")
SERIAL = re.compile(r"/dev/(?:tty[^\s\"',;]*|cu\.[^\s\"',;]*|pts/[^\s\"',;]+)|(?:\\\\[.?]\\)?COM\d+:?", re.I)
PORTS = {4532, 8300, 8301, 8515}
RIGCTLD = 4532
LOOPBACK = ("127.0.0.1", "localhost", "::1")
CLAIM = re.compile(r"/(private/)?tmp/hfmodem-rig-[^/]*\.claim$")
if os.name == "posix":
    import pwd
    HOME = pwd.getpwuid(os.getuid()).pw_dir
    STATE = os.path.realpath(os.path.join(HOME, ".local/state/hfmodem"))
else:
    HOME = os.path.expanduser("~")
    STATE = os.path.realpath(os.path.join(os.environ.get("LOCALAPPDATA") or os.path.join(HOME, "AppData", "Local"),
                                          "hfmodem", "state"))
CAPTURES = os.path.realpath(os.path.join(HOME, "radio", "captures"))
WRITES = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND
armed = False
_installed = False
_tty0tty = {}
_devices = {}
_admitted = []
_ports = []
_socket_ports = {}
DEVICE_KEYS = ("CAT_PORT", "PTT_PORT")
_pty_fds = {}
_serial_installed = False
_owner_started = None
_locks = {}
_device_lock = threading.RLock()
TTY0TTY_PAIR = ("/dev/tnt0", "/dev/tnt1")
TTY0TTY_RECEIPT = "/var/lib/hfmodem-ci/provisioning.json"
TTY0TTY_COMMIT = "e6334d0cbd5a7956571cd6c01181e3992732af8e"


def _text(path):
    with open(path) as fh:
        return fh.read()


def _trusted(path, kind):
    info = os.lstat(path)
    return kind(info.st_mode) and info.st_uid == 0 and not info.st_mode & 0o022


def _tty0tty_node(path, major):
    info = os.lstat(path)
    minor = TTY0TTY_PAIR.index(path)
    if (not stat.S_ISCHR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.getuid() or info.st_nlink != 1
            or (os.major(info.st_rdev), os.minor(info.st_rdev)) != (major, minor)
            or _text(f"/sys/class/tty/tnt{minor}/dev").strip() != f"{major}:{minor}"
            or os.path.realpath(f"/sys/dev/char/{major}:{minor}")
            != f"/sys/devices/virtual/tty/tnt{minor}"):
        raise ValueError("not the provisioned tty0tty node")
    return (info.st_dev, info.st_ino, info.st_rdev, info.st_uid, info.st_mode,
            info.st_ctime_ns)


def _validated_tty0tty():
    """A software-pair exception only in the owned Linux coexist execution.

    The root-managed receipt, kernel module and device namespace are trusted;
    participant environments or a matching device name alone grant nothing.
    No device is opened while validating this provenance.
    """
    if os.environ.get("HFMODEM_COEXIST_LINE_PROOF") != "1":
        return {}
    if sys.platform != "linux" or os.environ.get("HFMODEM_QUALIFICATION_PHASE") != "coexist":
        return {}
    try:
        system = os.uname()
        if system.nodename.split(".")[0] not in ("hfmodem-ci", "lima-hfmodem-ci"):
            return {}
        if not all(_trusted(path, stat.S_ISDIR) for path in
                   ("/", "/dev", "/var", "/var/lib", "/var/lib/hfmodem-ci")):
            return {}
        if not _trusted(TTY0TTY_RECEIPT, stat.S_ISREG):
            return {}
        receipt = json.loads(_text(TTY0TTY_RECEIPT))
        module = receipt["tty0tty"]
        if (receipt["schema_version"] != 1 or receipt["vm"] != "hfmodem-ci"
                or receipt["kernel"] != system.release
                or module["upstream"] != "https://github.com/freemed/tty0tty.git"
                or module["commit"] != TTY0TTY_COMMIT
                or module["pair"] != list(TTY0TTY_PAIR)
                or module["owner_uid"] != os.getuid()
                or not re.fullmatch(r"[0-9A-F]+", module["srcversion"])
                or _text("/sys/module/tty0tty/srcversion").strip() != module["srcversion"]):
            return {}
        drivers = [row.split() for row in _text("/proc/tty/drivers").splitlines()
                   if row.split()[:1] == ["tty0tty"]]
        if (len(drivers) != 1 or len(drivers[0]) != 5
                or drivers[0][1] != "/dev/tnt" or drivers[0][3:] != ["0-7", "serial"]):
            return {}
        major = int(drivers[0][2])
        if major <= 0:
            return {}
        return dict(nodes={path: _tty0tty_node(path, major) for path in TTY0TTY_PAIR},
                    major=major, srcversion=module["srcversion"])
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _tty0tty_open(event, original, path):
    if os.environ.get("HFMODEM_COEXIST_LINE_PROOF") != "1":
        return False
    if (event != "open" or original != path or path not in _tty0tty.get("nodes", {})
            or os.environ.get("HFMODEM_QUALIFICATION_PHASE") != "coexist"):
        return False
    try:
        # /dev is root-managed, so participants cannot replace an entry between
        # this audit check and open. Concurrent privileged namespace/module
        # changes are outside this Python tripwire's trust boundary.
        return (_trusted("/dev", stat.S_ISDIR)
                and _text("/sys/module/tty0tty/srcversion").strip() == _tty0tty["srcversion"]
                and _tty0tty_node(path, _tty0tty["major"]) == _tty0tty["nodes"][path])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _record(name, line):
    work = os.environ.get("HERMETIC_WORK")
    if work:
        with open(os.path.join(work, name), "a") as fh:
            fh.write(line + "\n")


def trip(what):
    _record("trips", f"python {what}")
    print(f"!! HERMETIC TRIP: python {what}", file=sys.stderr)
    raise PermissionError(f"HERMETIC TRIP: {what}")


def _fd_path(fd):
    import fcntl
    if hasattr(fcntl, "F_GETPATH"):
        return os.fsdecode(fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024)).rstrip(b"\0"))
    return os.readlink(f"/proc/self/fd/{fd}")


def _device_identity(path):
    # Not ctime: a terminal's moves whenever a client writes it by path. A
    # number reused after our pair closes is not ours: closing drops it from
    # the manifest.
    info = os.stat(path)
    return [info.st_dev, info.st_ino, info.st_rdev]


def _device_file():
    work = os.environ.get('HERMETIC_WORK')
    return os.path.join(work, f'devices-{os.getpid()}.json') if work else None


def _save_devices():
    global _owner_started
    path = _device_file()
    if path:
        if os.name == 'nt':
            if _owner_started is None:
                from hfmodem.core import claim
                _owner_started = claim.process_started(os.getpid())
        elif path not in _locks:
            # Held for the life of this process, or of the program it execs:
            # a reader that can share the lock knows the owner has ended.
            # Asking ps instead would fork from inside the audit hook, and a
            # fork briefly holds every descriptor its parent does.
            import fcntl
            fd = os.open(path[:-len('.json')] + '.lock', os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            _locks[path] = fd
        _write_manifest(path, dict(pid=os.getpid(), started=_owner_started, devices=_devices,
                                   admitted=_admitted, ports=_ports))


def _write_manifest(path, owner):
    temporary = path + '.tmp'
    with open(temporary, 'w') as stream:
        json.dump(owner, stream)
    os.replace(temporary, path)


def _alive(manifest, owner):
    if os.name == 'nt':
        # A retained handle keeps an exited process's identity: require it running.
        from hfmodem.core import winclaim
        started, exited = winclaim.process_info(owner['pid'])
        return started == owner['started'] and not exited
    import fcntl
    try:
        fd = os.open(manifest[:-len('.json')] + '.lock', os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def own_pty(master, slave):
    path = os.path.realpath(os.ttyname(slave))
    with _device_lock:
        _devices[path] = _device_identity(path)
        _pty_fds[master] = _pty_fds[slave] = path
        _save_devices()


def closed_pty(fd):
    with _device_lock:
        path = _pty_fds.pop(fd, None)
        if path and path not in _pty_fds.values():
            _devices.pop(path, None)
            _save_devices()


def own_com(port, target):
    if not any(word in target.casefold() for word in ('com0com', 'cnc')):
        raise ValueError(f'{port} is not a software serial device: {target}')
    com = re.fullmatch(r'(?:\\\\[.?]\\)?(COM\d+):?', port, re.I)
    if not com:
        raise ValueError(f'{port} is not a COM port')
    with _device_lock:
        _devices[com[1].upper()] = target
        _save_devices()


def admit(argv, environment, *, exact=False):
    """Let this test's launch boundary start an opaque program or site-free probe.

    Admits any launch whose argv begins with `argv` (a launcher re-executing
    itself adds its own arguments), or with `exact` only `argv` itself, and
    carries the devices `environment` names, each a device this test owns, now
    and again when it starts. A Python -S probe always requires the complete
    argv to match exactly.
    """
    words = _command_words(argv)
    if not words:
        raise ValueError("admit names a program")
    _admit(dict(argv=words, exact=exact), environment)


def admit_system(command, environment):
    """Let this test pass exactly `command` to os.system, carrying the devices
    `environment` names; every other command string is still refused."""
    if not command:
        raise ValueError("admit_system names a command")
    _admit(dict(system=os.fsdecode(command)), environment)


def _admit(entry, environment):
    devices = {key: environment.get(key) or "" for key in DEVICE_KEYS}
    for endpoint in filter(None, devices.values()):
        check_serial(endpoint)
    entry['devices'] = devices
    with _device_lock:
        if entry not in _admitted:
            _admitted.append(entry)
        # A test may have moved HERMETIC_WORK since an earlier admission.
        _save_devices()


def _manifests():
    """The device manifests of live owners in this test's work directory."""
    work = os.environ.get('HERMETIC_WORK')
    for manifest in glob.glob(os.path.join(work, 'devices-*.json')) if work else ():
        try:
            with open(manifest) as stream:
                owner = json.load(stream)
            if _alive(manifest, owner):
                yield owner
        except (OSError, TypeError, ValueError, KeyError, RuntimeError):
            pass


def _admitted_launch(words, environment, cwd=None, *, exact=False):
    def matches(entry):
        argv = entry.get('argv')
        return argv is not None and argv == (words if exact or entry.get('exact') else words[:len(argv)])
    return _granted(matches, environment, cwd)


def _admitted_system(command):
    command = os.fsdecode(command)
    return _granted(lambda entry: entry.get('system') == command, None)


def _granted(matches, environment, cwd=None):
    env = os.environ if environment is None else environment
    devices = {key: env.get(key) or "" for key in DEVICE_KEYS}
    def grants(admitted):
        return any(matches(entry) and entry['devices'] == devices for entry in admitted)
    if not (grants(_admitted) or any(grants(owner.get('admitted', ())) for owner in _manifests())):
        return False
    for endpoint in filter(None, devices.values()):
        check_serial(endpoint, cwd)
    return True


def check_serial(original, cwd=None):
    original = os.fsdecode(original)
    # A native CAT program can open a symlink just as Python can.
    com = re.fullmatch(r'(?:\\\\[.?]\\)?(COM\d+):?', original, re.I)
    if os.name == 'nt' and re.match(r'\\\\[.?]\\pipe\\', original, re.I):
        # No serial device; and resolving it would open the pipe, taking the
        # instance its own client is about to connect to.
        return
    if not com:
        if cwd is not None:
            original = os.path.join(os.fsdecode(cwd), original)
        original = os.path.realpath(original)
    if not SERIAL.fullmatch(original):
        return
    path = com[1].upper() if com else os.path.realpath(original)
    for owner in _manifests():
        try:
            devices = owner['devices']
            if path not in devices:
                continue
            if com:
                from hfmodem.core import winclaim
                if winclaim.serial_target(path) == devices[path]:
                    return
            elif _device_identity(path) == devices[path]:
                return
        except (OSError, TypeError, ValueError, KeyError, RuntimeError):
            pass
    trip(f'opened {original}: serial device is not owned by this test')


def _command_words(argv):
    if isinstance(argv, (str, bytes)):
        if os.name == 'nt':
            # Popen has already applied list2cmdline at its Windows audit event.
            # Use the OS parser; shlex does not implement Windows quoting.
            import ctypes as C
            from ctypes import wintypes as W
            shell = C.WinDLL('shell32', use_last_error=True)
            parse = shell.CommandLineToArgvW
            parse.argtypes, parse.restype = [W.LPCWSTR, C.POINTER(C.c_int)], C.POINTER(W.LPWSTR)
            free = C.WinDLL('kernel32', use_last_error=True).LocalFree
            free.argtypes, free.restype = [C.c_void_p], C.c_void_p
            count = C.c_int()
            result = parse(os.fsdecode(argv), C.byref(count))
            if not result:
                raise C.WinError(C.get_last_error())
            try:
                return [result[index] for index in range(count.value)]
            finally:
                free(result)
        return shlex.split(os.fsdecode(argv), posix=False)
    return [os.fsdecode(word) for word in argv]


def _program(executable, words):
    program = os.fsdecode(executable) if executable is not None else (words[0] if words else '')
    return ntpath.basename(program.strip('"\'')).lower().removesuffix('.exe')


def _opaque(program):
    return not (program in ('rigctl', 'rigctld', 'ffmpeg', 'ps', 'lsof', 'pgrep')
                or program.startswith(('python', 'pypy')))


def _shell_syntax_check(program, words, environment):
    env = os.environ if environment is None else environment
    return (os.name == 'posix' and program in ('bash', 'sh') and len(words) == 3
            and _program(None, words) == program
            and words[1] == '-n' and os.path.isabs(words[2]) and os.path.isfile(words[2])
            and not any(env.get(key) for key in ('BASH_ENV', 'ENV')))


def check_command(argv, executable=None, cwd=None, environment=None):
    words = _command_words(argv)
    program = _program(executable, words)
    native = program in ('rigctl', 'rigctld', 'ffmpeg')
    python = program.startswith(('python', 'pypy'))
    syntax = _shell_syntax_check(program, words, environment)
    if _opaque(program) and not syntax:
        trip(f'opaque native launch refused: {program}; use a protected Python child')
    if syntax:
        guard('open', (os.path.realpath(words[2]), 'r', 0))
    for word in words[1:]:
        # The native trampoline carries argv in JSON; check it before it can
        # exec a non-Python program outside the audit hook.
        if python and word.startswith('{'):
            try:
                payload = json.loads(word)
            except ValueError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get('argv'), list):
                inner = _command_words(payload['argv'])
                # The trampoline execs this argv; admission grants it as it would directly.
                # Its exec audits the PATH-resolved program, which only an absolute
                # argv[0] matches, so a bare name is refused here rather than there.
                if not (inner and os.path.isabs(inner[0]) and _opaque(_program(None, inner))
                        and _admitted_launch(inner, environment, cwd)):
                    check_command(inner, cwd=cwd, environment=environment)
        elif native:
            # Includes temporary aliases, --rig-file=/path and -rCOM41.
            # Inspect the endpoint, not COM substrings in ordinary filenames.
            endpoint = word.rsplit('=', 1)[-1]
            attached = re.fullmatch(r'-[rRpP](.+)', endpoint)
            check_serial(attached[1] if attached else endpoint, cwd)


STRACE_FLAGS = {'-f', '-ttt', '-yy', '--seccomp-bpf'}
STRACE_VALUES = {'-s': re.compile(r'\d+'), '-e': re.compile(r'trace=[%\w,]+'), '-o': re.compile(r'[^|!].*')}


def _traced(words, cwd):
    """The argv a `strace [options] -- argv` launch runs, its trace file checked.

    Only the options the Linux coexist observer passes are accepted: others can
    set the traced program's environment or user, attach to another process,
    inject syscall results or pipe the trace into a shell.
    """
    index = 1
    while index < len(words) and words[index] != '--':
        word = words[index]
        if word in STRACE_FLAGS:
            index += 1
        elif word in STRACE_VALUES and index + 1 < len(words) and STRACE_VALUES[word].fullmatch(words[index + 1]):
            if word == '-o':
                guard('open', (os.path.join(cwd, words[index + 1]), 'w', os.O_WRONLY | os.O_CREAT))
            index += 2
        else:
            trip(f'strace option refused: {word}')
    if len(words) < index + 2:
        trip('strace without a command after -- refused')
    return words[index + 1:]


def check_launch(executable, argv, environment, cwd=None):
    """Validate the actual audit boundary, including Python startup inheritance.

    Accept only this interpreter for Python children. Merely naming a binary
    'python' is not evidence that it loads our sitecustomize. A traced launch
    is checked as the command strace runs, as if launched directly.
    """
    cwd = os.getcwd() if cwd is None else os.path.realpath(os.fsdecode(cwd))
    traced = None
    # Windows Popen audits the command line it built, so admission reads that.
    if os.name == 'nt' or not isinstance(argv, (str, bytes)):
        words = _command_words(argv)
        if (_opaque(_program(executable, words))
                and (executable is None or os.fsdecode(executable) in words[:1])
                and _admitted_launch(words, environment, cwd)):
            return
        if os.name == 'posix' and _program(executable, words) == 'strace':
            traced = _traced(words, cwd)
    if traced is None:
        check_command(argv, executable, cwd, environment)
    if isinstance(argv, (str, bytes)) and os.name != 'nt':
        # Python command-line strings have platform-specific quoting rules;
        # require an argv sequence rather than invent another shell parser.
        trip('command-line string refused; use an argv sequence')
    words = _command_words(argv)
    if executable is None:
        executable = words[0] if words else ''
    env = os.environ if environment is None else environment
    executable = os.fsdecode(executable)
    search = os.pathsep.join(os.path.join(cwd, entry)
                             for entry in env.get('PATH', os.defpath).split(os.pathsep))
    resolved = shutil.which(os.path.join(cwd, executable) if os.path.dirname(executable) else executable,
                            path=search)
    program = ntpath.basename(os.fsdecode(executable)).lower().removesuffix('.exe')
    # A venv's base interpreter is the same one: Windows twins start it
    # directly so the stop channel owns the real process.
    interpreters = {os.path.realpath(sys.executable),
                    os.path.realpath(getattr(sys, '_base_executable', sys.executable))}
    if not (program.startswith('python') or program.startswith('pypy')):
        if resolved:
            actual = ntpath.basename(os.path.realpath(resolved)).lower().removesuffix('.exe')
            try:
                with open(resolved, 'rb') as stream:
                    line = stream.readline(4096)
            except OSError:
                line = b''  # nothing there to execute
            # An interpreter line naming exactly this interpreter, with no
            # options, starts a Python child; anything else is opaque.
            ours = line.startswith(b'#!') and os.path.realpath(os.fsdecode(line[2:].strip())) in interpreters
            syntax = _shell_syntax_check(program, words, env)
            # A test's own stand-in for a supported tool is admitted by its real
            # path, consulted only for a launch otherwise refused: reading the
            # manifests takes their locks, which a launch the guard accepts must not.
            if ((actual != program and not (syntax and actual in ('bash', 'sh', 'dash'))
                    or line.startswith(b'#!') and (syntax or not ours))
                    and not (not _opaque(program) and line.startswith(b'#!')
                             and _admitted_launch([os.path.realpath(resolved), *words[1:]], environment, cwd))):
                trip(f'opaque native launch refused: {executable}')
            if ours:
                _check_python_environment(env)
        if traced is not None:
            check_launch(None, traced, environment, cwd)
        return
    if (not os.path.isabs(os.fsdecode(executable)) or not resolved
            or os.path.realpath(resolved) not in interpreters):
        trip('unprotected Python interpreter refused')
    admitted = _admitted_launch(words, environment, cwd, exact=True)
    if words[1:2] == ['-S'] and admitted:
        return
    for word in words[1:]:
        if word in ('-c', '-m', '--', '-') or not word.startswith('-'):
            break
        # Admit only simple startup switches we need and understand. Options
        # taking values (-X/-W) must not hide an isolation flag after that value.
        if not re.fullmatch(r'-[uB]+', word):
            trip(f'unprotected Python startup refused: {word}')
    # An exactly admitted child may start without the guard's startup path.
    if not admitted:
        _check_python_environment(env)


def _fixture_sounddevice(entry, env):
    work = env.get('HERMETIC_WORK')
    path = os.path.join(entry, 'sounddevice.py')
    return (bool(work) and os.path.isfile(path)
            and os.path.normcase(os.path.realpath(path)).startswith(
                os.path.normcase(os.path.realpath(work)) + os.sep))


def _check_python_environment(env):
    # Launchers prepend package directories; only this test's scripted audio
    # may shadow a module supplied by this startup.
    startup = os.path.dirname(os.path.abspath(__file__))
    modules = [name for name in os.listdir(startup) if name.endswith('.py')]
    for entry in env.get('PYTHONPATH', '').split(os.pathsep):
        # A child may chdir before loading sitecustomize. Every entry through
        # ours must name the same directory there, including potential shadows.
        if not os.path.isabs(entry):
            trip('unprotected Python environment refused: relative startup path')
        entry = os.path.abspath(entry)
        if entry == startup:
            break
        shadows = [name for name in modules if os.path.exists(os.path.join(entry, name))
                   or os.path.isdir(os.path.join(entry, name[:-3]))]
        if shadows and (shadows != ['sounddevice.py'] or not _fixture_sounddevice(entry, env)
                        or os.path.isdir(os.path.join(entry, 'sounddevice'))):
            break
    else:
        entry = None
    if (entry != startup
            or any(not env.get(key) or env.get(key) != os.environ.get(key)
                   for key in ('HERMETIC_WORK', 'HERMETIC_TAG'))):
        trip('unprotected Python environment refused')


def _serve(endpoint):
    with _device_lock:
        _ports.append(endpoint)
        _save_devices()


def _closed_socket(reference):
    with _device_lock:
        endpoint = _socket_ports.pop(reference, None)
        if endpoint is not None:
            _ports.remove(endpoint)
            _save_devices()


# rigctld's own getopt_long table (Hamlib tests/rigctld.c).
RIGCTLD_SHORT = 'm:r:p:d:P:D:s:S:c:T:t:C:W:w:x:lLuovhVZRA:b'
RIGCTLD_LONG = {
    'model': 'm', 'rig-file': 'r', 'ptt-file': 'p', 'dcd-file': 'd', 'ptt-type': 'P',
    'dcd-type': 'D', 'serial-speed': 's', 'separator': 'S', 'civaddr': 'c', 'listen-addr': 'T',
    'port': 't', 'set-conf': 'C', 'list': 'l', 'show-conf': 'L', 'dump-caps': 'u', 'vfo': 'o',
    'verbose': 'v', 'help': 'h', 'version': 'V', 'twiddle_timeout': 'W', 'twiddle_rit': 'w',
    'uplink': 'x', 'debug-time-stamps': 'Z', 'password': 'A', 'rigctld-idle': 'R', 'bind-all': 'b'}
# Help, version, list, dump-caps and show-conf print and exit; -b binds every interface.
RIGCTLD_NOT_OWNED = set('hVluLb')


def _rigctld_options(words, environment=None):
    """The options rigctld reads from argv, in order; None for any form it would not accept."""
    env = os.environ if environment is None else environment
    # getopt_long permutes operands past options unless POSIXLY_CORRECT is set.
    posix = any((os.fsdecode(key).upper() if os.name == 'nt' else os.fsdecode(key)) == 'POSIXLY_CORRECT'
                for key in env)
    takes = {letter: RIGCTLD_SHORT[at + 1:at + 2] == ':'
             for at, letter in enumerate(RIGCTLD_SHORT) if letter != ':'}
    options, index = [], 1
    while index < len(words):
        word = words[index]
        index += 1
        if word == '--':
            break
        if word.startswith('--'):
            name, equals, value = word[2:].partition('=')
            names = [name] if name in RIGCTLD_LONG else [n for n in RIGCTLD_LONG if n.startswith(name)]
            if len(names) != 1:
                return None
            letter = RIGCTLD_LONG[names[0]]
            if takes[letter] and not equals:
                if index == len(words):
                    return None
                value, index = words[index], index + 1
            elif equals and not takes[letter]:
                return None
            options.append((letter, value if takes[letter] else None))
        elif word.startswith('-') and word != '-':
            for at, letter in enumerate(word[1:], 2):
                if letter not in takes:
                    return None
                if takes[letter]:
                    value = word[at:]
                    if not value:
                        if index == len(words):
                            return None
                        value, index = words[index], index + 1
                    options.append((letter, value))
                    break
                options.append((letter, None))
        elif posix:
            break
    return options


def _owned_rigctld(executable, argv, environment=None):
    """The endpoint a launch serves when it is this test's own rigctld."""
    words = _command_words(argv)
    options = _rigctld_options(words, environment) if _program(executable, words) == 'rigctld' else None
    if options is None or any(letter in RIGCTLD_NOT_OWNED for letter, _ in options):
        return None
    # Like rigctld, the last value given wins.
    values = dict(options)
    if values.get('t', str(RIGCTLD)) != str(RIGCTLD):
        return None
    try:
        address = ipaddress.ip_address(values.get('T', '127.0.0.1'))
    except ValueError:
        return None
    family = socket.AF_INET if address.version == 4 else socket.AF_INET6
    return [socket.SOCK_STREAM, family, str(address), RIGCTLD, False]


def _created(pid, handle, endpoint):
    # Windows has no exec, and the process that creates rigctld (the native
    # supervisor) exits once it is running. The endpoint lives as long as the
    # rigctld process itself, so its manifest is keyed to that process.
    work = os.environ.get('HERMETIC_WORK')
    if work:
        from hfmodem.core import winclaim
        started, _ = winclaim.process_info(pid, handle=handle)
        _write_manifest(os.path.join(work, f'devices-{pid}.json'),
                        dict(pid=pid, started=started, devices={}, admitted=[], ports=[endpoint]))


def _end_created(winapi, process, thread):
    """End a created child whose handles nothing else holds.

    Every step runs. Returns whether it is gone and the first interruption.
    """
    steps = (lambda: winapi.TerminateProcess(process, 1),
             lambda: winapi.WaitForSingleObject(process, 5000) == 0,
             lambda: winapi.CloseHandle(thread),
             lambda: winapi.CloseHandle(process))
    results, interruption = [], None
    for step in steps:
        try:
            results.append(step())
        except OSError:
            results.append(None)
        except BaseException as exc:
            results.append(None)
            interruption = interruption or exc
    ended = results[1] is True
    return ended, interruption


def _endpoint(sock, address):
    host, port = address[:2]
    if host == 'localhost':
        host = '::1' if sock.family == socket.AF_INET6 else '127.0.0.1'
    try:
        host = ipaddress.ip_address(host)
    except ValueError:
        return None
    if (sock.family, host.version) not in ((socket.AF_INET, 4), (socket.AF_INET6, 6)):
        return None
    dual = False
    if sock.family == socket.AF_INET6:
        try:
            dual = not sock.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY)
        except (AttributeError, OSError):
            pass
    return [sock.type, sock.family, str(host), port, dual]


def _addresses(endpoint):
    _, family, host, _, dual = endpoint
    address = ipaddress.ip_address(host)
    yield family, address
    if family == socket.AF_INET6 and dual:
        if address.is_unspecified:
            yield socket.AF_INET, ipaddress.ip_address('0.0.0.0')
        elif address.ipv4_mapped:
            yield socket.AF_INET, address.ipv4_mapped


def _served(sock, address):
    target = _endpoint(sock, address)
    if target is None:
        return False
    for owner in _manifests():
        for endpoint in owner.get('ports', ()):
            if endpoint[0] != target[0] or endpoint[3] != target[3]:
                continue
            if any(family == wanted_family and (host == wanted or host.is_unspecified and wanted.is_loopback)
                   for family, host in _addresses(endpoint)
                   for wanted_family, wanted in _addresses(target)):
                return True
    return False


_probing = threading.local()


def _listened(host, port):
    """Whether anything accepts on the port. The probe sends nothing."""
    import socket
    _probing.active = True
    try:
        socket.create_connection((host, port), timeout=1).close()
        return True
    except OSError:
        return False
    finally:
        _probing.active = False


def guard(event, args):
    if not armed:
        return
    if event in ("open", "os.mkdir") and isinstance(args[0], (str, bytes, os.PathLike)):
        original = os.fsdecode(args[0])
        path = os.path.realpath(original)
        check_serial(original if re.fullmatch(r'(?:\\\\[.?]\\)?COM\d+:?', original, re.I) else path)
        if (path.startswith("/dev/") and not TEST_DEVICES.fullmatch(path)
                and not SERIAL.fullmatch(path)
                and not _tty0tty_open(event, original, path)):
            trip(f"opened {path}")
        if path == STATE or path.startswith(STATE + os.sep):
            trip(f"reached the operator's state directory at {path}")
        if ((event == "os.mkdir" or (args[2] or 0) & WRITES)
                and (path == CAPTURES or path.startswith(CAPTURES + os.sep))):
            trip(f"wrote into the operator's captures at {path}")
    elif event in ('subprocess.Popen', 'os.exec', 'os.posix_spawn'):
        check_launch(args[0], args[1], args[3] if event == 'subprocess.Popen' else args[2],
                     args[2] if event == 'subprocess.Popen' else None)
        endpoint = _owned_rigctld(args[0], args[1], args[2]) if event == 'os.exec' else None
        if endpoint is not None:
            # This process becomes a rigctld of this test's own devices; the
            # lock it carries across exec says while.
            _serve(endpoint)
            for fd in _locks.values():
                os.set_inheritable(fd, True)
    elif event == 'os.system':
        if not _admitted_system(args[0]):
            trip('opaque native launch refused: os.system')
    elif event == "socket.connect" and isinstance(args[1], tuple) and args[1][1:2] and args[1][1] in PORTS:
        host, port = args[1][:2]
        if getattr(_probing, 'active', False):
            return
        if host not in LOOPBACK or not _served(args[0], args[1]) and _listened(host, port):
            trip(f"connected to {host}:{port}")
    elif event == "fcntl.flock":
        try:
            path = _fd_path(args[0])
        except OSError:
            return
        if CLAIM.search(path):
            _record("claims", path)


def arm():
    global armed, _installed, _tty0tty
    _tty0tty = _validated_tty0tty()
    if not _installed:
        sys.addaudithook(guard)
        _installed = True
    armed = True


class SerialOpen:
    def __init__(self, function):
        object.__setattr__(self, 'function', function)

    def __getattr__(self, name):
        return getattr(self.function, name)

    def __setattr__(self, name, value):
        setattr(self.function, name, value)

    def __call__(self, path, *args):
        value = getattr(path, 'value', path)
        if armed and isinstance(value, (str, bytes, os.PathLike)):
            check_serial(value)
        return self.function(path, *args)


def install_serial_guards(patch=setattr):
    global _serial_installed
    if patch is setattr and _serial_installed:
        return
    if patch is setattr:
        _serial_installed = True
    original_open = os.open
    def open_file(path, flags, mode=0o777, *, dir_fd=None):
        if (armed and os.name == 'posix' and dir_fd is not None
                and isinstance(path, (str, bytes, os.PathLike))):
            relative = os.fsdecode(path)
            if not os.path.isabs(relative):
                guard('open', (os.path.join(_fd_path(dir_fd), relative), None, flags))
        return original_open(path, flags, mode, dir_fd=dir_fd)
    patch(os, 'open', open_file)
    import socket
    socket_bind, socket_close, socket_detach = socket.socket.bind, socket.socket.close, socket.socket.detach
    def bind_socket(self, address):
        with _device_lock:
            result = socket_bind(self, address)
            if armed:
                bound = self.getsockname()
                if isinstance(bound, tuple) and bound[1:2] and bound[1] in PORTS:
                    reference = weakref.ref(self, _closed_socket)
                    endpoint = _endpoint(self, bound)
                    if endpoint is not None and reference not in _socket_ports:
                        _socket_ports[reference] = endpoint
                        _serve(endpoint)
            return result
    def close_socket(self):
        with _device_lock:
            try:
                return socket_close(self)
            finally:
                _closed_socket(weakref.ref(self))
    def detach_socket(self):
        with _device_lock:
            result = socket_detach(self)
            # A raw descriptor no longer has a socket lifetime we can track.
            _closed_socket(weakref.ref(self))
            return result
    patch(socket.socket, 'bind', bind_socket)
    patch(socket.socket, 'close', close_socket)
    patch(socket.socket, 'detach', detach_socket)
    if hasattr(os, 'openpty'):
        openpty, close = os.openpty, os.close
        def owned_pty():
            master, slave = openpty()
            if armed:
                own_pty(master, slave)
            return master, slave
        def close_pty(fd):
            result = close(fd)
            if armed:
                closed_pty(fd)
            return result
        patch(os, 'openpty', owned_pty)
        patch(os, 'close', close_pty)
    if os.name == 'nt':
        # subprocess's own binding: the one Popen creates processes through.
        winapi = getattr(subprocess, '_winapi', None)
        if winapi is not None:
            create_process = winapi.CreateProcess
            def create_owned(executable, command, *args):
                endpoint = _owned_rigctld(executable, command, args[4]) if armed else None
                result = create_process(executable, command, *args)
                if endpoint is not None:
                    process, thread = result[:2]
                    try:
                        _created(result[2], int(process), endpoint)
                    except BaseException as exc:
                        # Popen never sees these handles, so nothing else ends the child.
                        ended, interruption = _end_created(winapi, process, thread)
                        error = interruption or exc
                        if not ended:
                            raise RuntimeError(f'rigctld pid {result[2]}: termination unresolved') from error
                        raise error
                return result
            patch(winapi, 'CreateProcess', create_owned)
        import serial
        original = serial.Serial.open
        def open_serial(self):
            if armed and self.port is not None:
                check_serial(self.port)
            return original(self)
        patch(serial.Serial, 'open', open_serial)
        import ctypes as C
        indexed = set()
        def protect(api):
            for entry in ('CreateFileW', 'CreateFileA'):
                function = getattr(api, entry, None)
                if function is not None and not isinstance(function, SerialOpen):
                    patch(api, entry, SerialOpen(function))
            kind = type(api)
            if kind not in indexed and hasattr(kind, '__getitem__'):
                getitem = kind.__getitem__
                def checked_item(self, entry):
                    function = getitem(self, entry)
                    name = os.fsdecode(entry).split('\0', 1)[0] if isinstance(entry, (str, bytes)) else entry
                    if (isinstance(name, int) or name in ('CreateFileW', 'CreateFileA')) and not isinstance(function, SerialOpen):
                        return SerialOpen(function)
                    return function
                # Special methods are looked up on the class. Patching only
                # this DLL instance would leave api['CreateFileW'] unguarded.
                patch(kind, '__getitem__', checked_item)
                indexed.add(kind)
            return api
        windll = C.WinDLL
        def checked_dll(name, *args, **kwargs):
            api = windll(name, *args, **kwargs)
            if ntpath.basename(os.fsdecode(name)).lower().removesuffix('.dll') == 'kernel32':
                protect(api)
            return api
        patch(C, 'WinDLL', checked_dll)
        # LibraryLoader keeps its own DLL constructor and caches loaded DLLs.
        # Updating WinDLL alone changes neither path.
        loader = getattr(C, 'windll', None)
        if loader is not None:
            patch(loader, '_dlltype', checked_dll)
            for name, api in list(vars(loader).items()):
                if ntpath.basename(name).lower().removesuffix('.dll') == 'kernel32':
                    protect(api)
        for name in ('hfmodem.core.winclaim', 'hfmodem.core.winproc'):
            api = getattr(sys.modules.get(name), '_api_cache', None)
            if api is not None:
                protect(api)


def disarm():
    global armed, _tty0tty, _owner_started
    armed = False
    _tty0tty = {}
    _devices.clear()
    _admitted.clear()
    _ports.clear()
    _socket_ports.clear()
    _pty_fds.clear()
    _owner_started = None
    for fd in _locks.values():
        os.close(fd)
    _locks.clear()


def _held(path):
    lsof = shutil.which("lsof", path=os.environ.get("HERMETIC_PATH") or os.environ.get("PATH"))
    return bool(lsof and subprocess.run([lsof, "-t", path], capture_output=True, text=True).stdout.split())


def sweep(work, tag):
    """Clear the claims this run took: every recorded one named for `tag`, and,
    for a claim a shell or another non-Python child took, every one the tag
    names. A recorded claim without the tag is a TRIP and stays put."""
    try:
        recorded = {os.path.realpath(p) for p in open(os.path.join(work, "claims")).read().split()}
    except FileNotFoundError:
        recorded = set()
    named = {os.path.realpath(p) for p in glob.glob(f"/tmp/hfmodem-rig-*{tag}*.claim")}
    for path in sorted(recorded | named):
        if tag not in os.path.basename(path):
            _record("trips", f"claim {path} is not this run's -- left in place")
            continue
        if f"no-cat.{tag}" in path:
            _record("trips", f"claim {path} taken by a leg that holds one already")
        if os.path.exists(path) and not _held(path):
            os.unlink(path)


if __name__ == "__main__" and sys.argv[1:2] == ["sweep"]:
    sweep(os.environ["HERMETIC_WORK"], os.environ["HERMETIC_TAG"])
