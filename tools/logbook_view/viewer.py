"""Read-only discovery of derived replay artifacts; no decoder imports."""
from pathlib import Path
import json
from urllib.parse import quote


def artifacts(root, arm):
    if not arm or Path(arm).name != arm or arm in ('.', '..'):
        return []
    parent = Path(root).resolve()
    results = []
    for path in (parent / arm).glob('*/manifest.json'):
        if path.parent.name.startswith('.') or not path.resolve().is_relative_to(parent):
            continue
        try:
            manifest = json.loads(path.read_text())
            if manifest.get('schema') != 1 or manifest.get('arm') != arm:
                continue
            results.append({'revision': path.parent.name, 'created_utc': manifest['created_utc'],
                            'source_sha256': manifest['decoder']['source_sha256'],
                            'base': '/viewer-data/' + quote(arm) + '/' + quote(path.parent.name) + '/',
                            'manifest': manifest})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(results, key=lambda r: r['created_utc'], reverse=True)


def data_path(root, rel):
    root = Path(root).resolve()
    parts = Path(rel).parts
    if len(parts) != 3 or any(p.startswith('.') for p in parts):
        return None
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    # Serve only the files in the viewer's artifact format.
    if path.name not in ('manifest.json', 'events.json') and not (
        path.name.startswith('symbols-') and path.suffix == '.json' or
        path.name.startswith('waterfall-') and path.suffix == '.bin'
    ):
        return None
    return path
