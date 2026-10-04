"""Refresh the runtime-proxy token of a colab-cli session (CLI 0.6.0 never renews it,
so file access dies ~1 h after `colab new`). Usage: python refresh_token.py <session>"""
import sys
from colab_cli.common import State

name = sys.argv[1]
state = State()
s = state.store.get(name)
if s is None:
    sys.exit(f"no local session {name!r}")
for a in state.client.list_assignments():
    if a.endpoint == s.endpoint:
        changed = a.runtime_proxy_info.token != s.token
        s.token = a.runtime_proxy_info.token
        s.url = a.runtime_proxy_info.url
        state.store.add(s)
        print(f"refreshed {name}: token {'changed' if changed else 'unchanged'}, "
              f"expires in {a.runtime_proxy_info.token_expires_in_seconds}s")
        break
else:
    sys.exit(f"endpoint {s.endpoint} not among server assignments")
