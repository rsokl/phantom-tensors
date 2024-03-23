import importlib.metadata

# Skip collection of tests that require additional dependencies
collect_ignore_glob = []

OPTIONAL_TEST_DEPENDENCIES = (
    "numpy",
    "torch",
    "beartype",
    "phantom-types",
)

_installed = {dist.metadata["Name"] for dist in importlib.metadata.distributions()}

if any(_module_name not in _installed for _module_name in OPTIONAL_TEST_DEPENDENCIES):
    collect_ignore_glob.append("*third_party.py")
