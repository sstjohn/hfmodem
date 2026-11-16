# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)
import ast
from dataclasses import FrozenInstanceError, fields, is_dataclass
from enum import Enum
import inspect
import sys
import types
from typing import Literal, Union, get_args, get_origin, get_type_hints

import pytest

from hfmodem.contact import chat_state as s


def sample(annotation):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        return None if type(None) in args else sample(args[0])
    if origin is Literal:
        return args[0]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return ()
        return tuple(sample(arg) for arg in args)
    if annotation is str:
        return 'value'
    if annotation is int:
        return 1
    if annotation is float:
        return 1.25
    if annotation is bool:
        return False
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return next(iter(annotation))
    if is_dataclass(annotation):
        hints = get_type_hints(annotation)
        return annotation(**{f.name: sample(hints[f.name]) for f in fields(annotation)})
    raise AssertionError(annotation)


@pytest.mark.parametrize('record_type', tuple(s.RECORD_TYPES.values()), ids=lambda cls: cls.__name__)
def test_all_contract_records_frozen_slotted_and_strictly_round_trip(record_type):
    record = sample(record_type)
    assert s.decode(s.encode(record)) == record
    assert not hasattr(record, '__dict__')
    with pytest.raises(FrozenInstanceError):
        setattr(record, fields(record)[0].name, None)
    wire = s.to_wire(record)
    wire.pop(fields(record)[0].name)
    with pytest.raises(ValueError, match='missing or extra'):
        s.from_wire(wire)


def test_actions_are_a_closed_typed_union_and_effects_are_complete():
    assert {cls for cls, _ in s.ACTIONS.values()} == set(get_args(s.Action))
    for kind, (cls, effect) in s.ACTIONS.items():
        action = sample(cls)
        assert action.kind == kind
        assert s.effect(action) == effect
    assert set(get_args(s.OperationalAction)) == {
        cls for kind, (cls, effect) in s.ACTIONS.items() if effect is not s.Effect.R}
    assert not any(e is s.Effect.T for _, e in s.ACTIONS.values())


def test_contract_uses_standard_library_only_and_no_textual_backend_imports():
    tree = ast.parse(inspect.getsource(s))
    imported = {n.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for n in node.names}
    imported |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert imported <= sys.stdlib_module_names
    from hfmodem.contact import chat_controller, chat_session, chat_fake, varac
    for module in (chat_controller, chat_session, chat_fake, varac):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(n.name.split('.')[0] != 'textual' for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or '').split('.')[0] != 'textual'


def test_serialization_rejects_bool_integer_confusion_mutable_objects_and_nonfinite_values():
    wire = s.to_wire(s.ChatQSY(2))
    wire['slot'] = True
    with pytest.raises(ValueError):
        s.from_wire(wire)
    wire = s.to_wire(s.ChatSend('hello'))
    wire['kind'] = 'chat.exit'
    with pytest.raises(ValueError):
        s.from_wire(wire)
    with pytest.raises(ValueError):
        s.encode({'mutable': []})
    with pytest.raises(ValueError):
        s.encode(float('nan'))


def test_action_metadata_has_one_registry_and_no_unused_read_page():
    assert set(s.ACTIONS) == {sample(cls).kind for cls in get_args(s.Action)}
    for kind, (cls, effect) in s.ACTIONS.items():
        assert sample(cls).kind == kind
        assert s.effect(sample(cls)) is effect
    with pytest.raises(ValueError, match='unknown record'):
        s.from_wire({'_type': 'ReadPage'})
