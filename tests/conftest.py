def pytest_addoption(parser):
    parser.addoption("--onlyspin", action="store_true",
                     help="run only for SU(2)-adapted tensors")
    parser.addoption("--nospin", action="store_true",
                     help="run only for U(1)-adapted tensors")


def pytest_generate_tests(metafunc):
    if "kind" in metafunc.fixturenames:
        if metafunc.config.getoption("onlyspin"):
            kinds = ['SU(2)']
        elif metafunc.config.getoption("nospin"):
            kinds = ['U(1)']
        else:
            kinds = ['U(1)', 'SU(2)']
        metafunc.parametrize("kind", kinds)
