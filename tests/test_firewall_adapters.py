import json
from types import SimpleNamespace

from src.firewall_adapters import MockFirewallAdapter, UfwFirewallAdapter, NftablesFirewallAdapter


class FakeRunner:
    def __init__(self, returncode=0, stdout=""):
        self.returncode = returncode
        self.stdout = stdout
        self.calls = []

    def __call__(self, args, capture_output=True, text=True, **kwargs):
        self.calls.append(args)
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout)


def test_mock_adapter_block_unblock():
    adapter = MockFirewallAdapter()
    assert adapter.block("1.2.3.4", 60) is True
    assert adapter.unblock("1.2.3.4") is True


def test_ufw_adapter_calls_expected_commands():
    runner = FakeRunner(returncode=0)
    adapter = UfwFirewallAdapter(run_cmd=runner)
    assert adapter.block("1.2.3.4", 60) is True
    assert runner.calls[-1][:3] == ["ufw", "deny", "from"]


def test_nftables_unblock_uses_handle_deletes():
    list_out = json.dumps({
        "nftables": [
            {
                "rule": {
                    "handle": 9,
                    "expr": [
                        {
                            "match": {
                                "left": {"payload": {"protocol": "ip", "field": "saddr"}},
                                "right": "1.2.3.4",
                            }
                        },
                        {"drop": None},
                    ],
                }
            }
        ]
    })

    class Runner:
        def __init__(self):
            self.calls = []

        def __call__(self, args, capture_output=True, text=True, **kwargs):
            self.calls.append(args)
            if args[:3] == ["nft", "-j", "list"]:
                return SimpleNamespace(returncode=0, stdout=list_out)
            return SimpleNamespace(returncode=0, stdout="")

    runner = Runner()
    adapter = NftablesFirewallAdapter(run_cmd=runner)
    assert adapter.unblock("1.2.3.4") is True
