"""Loopback dashboard; run with python -m streaming.app.dashboard."""
import argparse
import importlib.util
import json
import os
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .transfers import Transfers, MAX_BYTES
from .locking import RuntimeLock

WEB = Path(__file__).resolve().parents[1] / 'web' / 'index.html'


def server(root, port=8765):
    runtime_lock = RuntimeLock(Path(root) / "dashboard.lock")
    runtime_lock.__enter__()
    transfers = Transfers(root)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, body, mime='application/json'):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'nonce-" + token + "'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self, mutation=False):
            host = '127.0.0.1:' + str(self.server.server_port)
            if self.headers.get('Host') != host:
                self.respond(403, {'error': 'Use the local dashboard address.'})
                return False
            if mutation and (self.headers.get('Origin') != 'http://' + host or
                             not secrets.compare_digest(self.headers.get('X-Session-Token', ''), token)):
                self.respond(403, {'error': 'Refresh the local dashboard session.'})
                return False
            return True

        def do_GET(self):
            if not self.allowed():
                return
            path = urlparse(self.path).path
            if path == '/':
                self.respond(200, WEB.read_text(encoding='utf-8').replace('__TOKEN__', token).encode(), 'text/html; charset=utf-8')
            elif path == '/api/status':
                self.respond(200, {'jobs': transfers.history(),
                                  'kafka_client': importlib.util.find_spec('confluent_kafka') is not None,
                                  'secret_ready': len(os.environ.get('FCL_TRANSFER_SECRET', '')) >= 32})
            elif path.startswith('/api/download/'):
                key = path.rsplit('/', 1)[-1]
                job = next((j for j in transfers.history() if j['id'] == key and j['status'] == 'Verified'), None)
                file = transfers.root / (key + '.weights') if job else None
                if file and file.is_file():
                    self.respond(200, file.read_bytes(), 'application/octet-stream')
                else:
                    self.respond(404, {'error': 'Verified file not found.'})
            else:
                self.respond(404, {'error': 'Not found.'})

        def do_POST(self):
            if not self.allowed(mutation=True):
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                limit = MAX_BYTES if self.path == '/api/upload' else 8192
                if not 0 < length <= limit:
                    raise ValueError('Request is empty or too large.')
                self.connection.settimeout(30)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError('Incomplete request.')
                if self.path == '/api/upload':
                    result = transfers.upload(body)
                else:
                    data = json.loads(body)
                    if not isinstance(data, dict):
                        raise ValueError('Expected a JSON object.')
                    if self.path == '/api/start':
                        result = {'id': transfers.start(data)}
                    elif self.path == '/api/cancel':
                        transfers.cancel(data.get('id'))
                        result = {'requested': True}
                    else:
                        self.respond(404, {'error': 'Not found.'})
                        return
                self.respond(200, result)
            except (ValueError, KeyError) as exc:
                self.respond(400, {'error': str(exc)})
            except Exception:
                self.respond(500, {'error': 'Local operation failed. Check storage and restart if needed.'})

    class Server(ThreadingHTTPServer):
        def server_close(self):
            super().server_close()
            runtime_lock.__exit__()

    try:
        httpd = Server(('127.0.0.1', port), Handler)
    except BaseException:
        runtime_lock.__exit__()
        raise
    httpd.transfers = transfers
    return httpd


def main():
    parser = argparse.ArgumentParser(description='Model-weight file dashboard (no training)')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--state-dir', type=Path, default=Path(__file__).resolve().parents[1] / '.local' / 'transfers')
    args = parser.parse_args()
    app = server(args.state_dir, args.port)
    print(f'Weight dashboard: http://127.0.0.1:{app.server_port}', flush=True)
    try:
        app.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.transfers.cancelled.set()
        app.server_close()


if __name__ == '__main__':
    main()
