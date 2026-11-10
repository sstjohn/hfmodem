# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Saul St John (W9SSJ)

"""Serve a built site locally, including byte ranges for recording seeking.

    python site/tools/build.py
    python site/tools/preview.py --directory _site --port 8767
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import re


class RecordingHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        # This server previews files edited in place; cached modules can mix revisions.
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def send_head(self):
        self.remaining = None
        header = self.headers.get('Range')
        path = Path(self.translate_path(self.path))
        if not header or not path.is_file():
            return super().send_head()
        source = path.open('rb')
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
        if match and any(match.groups()):
            first, last = match.groups()
            start = int(first) if first else max(0, size - int(last))
            end = min(size - 1, int(last)) if first and last else size - 1
        else:
            start, end = size, -1
        if not 0 <= start <= end < size:
            source.close()
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        source.seek(start)
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(self.remaining))
        self.end_headers()
        return source

    def copyfile(self, source, output):
        if self.remaining is None:
            return super().copyfile(source, output)
        while self.remaining:
            chunk = source.read(min(65536, self.remaining))
            if not chunk:
                break
            output.write(chunk)
            self.remaining -= len(chunk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('_site'))
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    handler = partial(RecordingHandler, directory=str(args.directory.resolve()))
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    print(f'Site preview: http://localhost:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
