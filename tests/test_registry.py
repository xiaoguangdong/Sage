from sage.core.registry import Registry


def test_registry_register_and_get():
    reg = Registry("test")

    @reg.register("demo")
    class Demo:
        value = 1

    assert reg.get("demo") is Demo
    assert "demo" in list(reg.list())

