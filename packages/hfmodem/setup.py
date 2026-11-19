# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Include the reviewed operator tools in wheels and source distributions."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tomllib

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def operator_source():
    staged = HERE / '_operator_source'
    if staged.is_dir():
        return staged, [p.relative_to(staged).as_posix() for p in staged.rglob('*') if p.is_file()]
    inventory = REPO / 'publish/operator-tools.toml'
    if not inventory.is_file():
        raise RuntimeError('Build from the complete source distribution or release checkout.')
    names = tomllib.loads(inventory.read_text())['operators']['paths']
    return REPO, [name for name in names if not Path(name).name.startswith('test_')
                  and not name.endswith('.md')]


def copy_operator(destination):
    source, names = operator_source()
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'tools':
            raise RuntimeError(f'Invalid operator resource: {name}')
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)


class OperatorBuild(build_py):
    def run(self):
        super().run()
        resources = Path(self.build_lib) / 'hfmodem/_operator'
        if resources.exists():
            shutil.rmtree(resources)
        copy_operator(resources)

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [str(p) for p in
            (Path(self.build_lib) / 'hfmodem/_operator').rglob('*') if p.is_file()]


class OperatorSdist(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        copy_operator(Path(base_dir) / '_operator_source')


# Release wheels are py3-none-any (PyPI refuses bare linux tags), so a pure build
# outranks HFMODEM_NATIVE; source installs and editable checkouts still build the
# P4 LONG kernel when cargo is present.
pure = os.environ.get('HFMODEM_PURE') == '1'

native_options = {}
if os.environ.get('HFMODEM_NATIVE') == '1' and not pure:
    from setuptools_rust import Binding, RustExtension

    # setuptools-rust is not a declared build requirement, so default builds need
    # none; an opt-in build brings its own. From packages/hfmodem, in place:
    #   HFMODEM_NATIVE=1 python setup.py build_rust --inplace --release
    # or a wheel: HFMODEM_NATIVE=1 pip wheel --no-build-isolation --no-deps .
    native_options = {
        'rust_extensions': [RustExtension('hfmodem.core._dsp',
                                         path='native/Cargo.toml', binding=Binding.PyO3,
                                         args=['--locked'])],
        'options': {'bdist_wheel': {'py_limited_api': 'cp313'}},
        'zip_safe': False,
    }


class NativeBuild(build_ext):
    def get_ext_filename(self, fullname):
        # get_ext_fullpath passes only the last component of the module name.
        if fullname.rpartition('.')[2] != '_p4long_native':
            return super().get_ext_filename(fullname)
        target = os.environ.get('CARGO_BUILD_TARGET', '')
        platform = ('win32' if 'windows' in target else 'darwin' if 'apple' in target
                    else 'linux') if target else sys.platform
        suffix = '.dll' if platform == 'win32' else '.dylib' if platform == 'darwin' else '.so'
        return os.path.join(*fullname.split('.')) + suffix

    def build_extension(self, ext):
        # LONG replies need the native kernel to meet the slot; measured Python exceeds the budget.
        cargo = shutil.which('cargo')
        if cargo is None:
            self.warn('cargo unavailable; installing P4 LONG Python fallback')
            return
        crate = HERE / 'native/p4long'
        target = os.environ.get('CARGO_BUILD_TARGET', '')
        # ARMv7 cross-builds need the armv7-unknown-linux-gnueabihf target and linker.
        environment = os.environ.copy()
        environment.pop('RUSTFLAGS', None)
        environment.pop('CARGO_ENCODED_RUSTFLAGS', None)
        try:
            subprocess.run([cargo, 'build', '--offline', '--release', '--locked',
                            '--manifest-path', str(crate / 'Cargo.toml'),
                            '--target-dir', str(crate / 'target')], check=True, env=environment)
            if self.editable_mode:
                # An editable checkout loads the kernel from native/p4long/target; a copy
                # beside the sources would shadow later cargo builds.
                return
            destination = Path(self.get_ext_fullpath(ext.name))
            suffix = destination.suffix
            source = crate / 'target' / target / 'release' / (
                ('hfmodem_p4long' if suffix == '.dll' else 'libhfmodem_p4long') + suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        except (OSError, subprocess.CalledProcessError) as error:
            self.warn(f'P4 LONG native build unavailable; installing Python fallback: {error}')


p4long = [] if pure else [
    Extension('hfmodem.shrike._p4long_native', ['native/p4long/src/lib.rs'], optional=True)]

setup(cmdclass={'build_py': OperatorBuild, 'sdist': OperatorSdist, 'build_ext': NativeBuild},
      ext_modules=p4long, **native_options)
