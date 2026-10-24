from railhead import server as srv

from rh17m import sched

ORIG = getattr(srv.OpenTTDServer.command, "__wrapped__", srv.OpenTTDServer.command)


def _cmd():
    s = srv.OpenTTDServer.__new__(srv.OpenTTDServer)
    s.binary, s.game_port, s.workdir, s.config = "openttd", 1, srv.Path("/w"), srv.ServerConfig()
    return srv.OpenTTDServer.command(s)


def test_install_prefixes_server_once(monkeypatch):
    monkeypatch.setattr(srv.OpenTTDServer, "command", ORIG)
    monkeypatch.delenv("RH17M_APP_POLICY", raising=False)
    sched.install()
    sched.install()
    cmd = _cmd()
    assert cmd[:3] == ["taskpolicy", "-a", "openttd"]


def test_install_can_be_disabled(monkeypatch):
    monkeypatch.setattr(srv.OpenTTDServer, "command", ORIG)
    monkeypatch.setenv("RH17M_APP_POLICY", "0")
    sched.install()
    assert _cmd()[0] == "openttd"
