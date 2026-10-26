# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
"""Include the reviewed operator tools in wheels and source distributions."""
from pathlib import Path
import shutil
import tomllib

from setuptools import setup
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


setup(cmdclass={'build_py': OperatorBuild, 'sdist': OperatorSdist})
