import axiom.compiler.parser as parser


def pytest_configure() -> None:
    parser.reset_parser()
