import pkg_resources

# Skip collection of tests that require additional dependencies
collect_ignore_glob = []

OPTIONAL_TEST_DEPENDENCIES = (
    "numpy",
    "torch",
    "beartype",
)

_installed = {pkg.key for pkg in pkg_resources.working_set}

if any(_module_name not in _installed for _module_name in OPTIONAL_TEST_DEPENDENCIES):
    print("HERE")
    collect_ignore_glob.append("*third_party.py")
