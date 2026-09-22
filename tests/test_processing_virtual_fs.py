"""Tests for the virtual-filesystem sampled reader (``processing/virtual_fs.py``).

Three layers, each isolated at a real boundary rather than a mock of the module
itself:

1. ``VirtualFsSource.resolve`` / ``plan`` — pure path and arithmetic logic.  A
   library symlink into the mount must map to the HTTP URL, and the bitrate
   rule must keep the sequential pass for cheap files.
2. ``ChunkProxy`` — exercised against a real local HTTP upstream (a tiny
   ``http.server`` serving a random blob with Range support and a request
   counter).  This is where the cost model lives: one upstream request per
   aligned chunk, cache hits for the header/index re-reads, and no fetch of
   chunks the client never asked for.
3. ``extract`` — with the runner replaced by a fake that writes the output
   file, asserting the per-sample argv shape and the ``img-%06d.jpg`` naming
   the caller's rename step depends on.
"""

from __future__ import annotations

import http.client
import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from media_preview_generator.processing import virtual_fs
from media_preview_generator.processing.virtual_fs import MIB, ChunkProxy, VirtualFsSource

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeConfig:
    thumbnail_interval: int = 10
    virtual_fs_enabled: bool = True
    virtual_fs_mode: str = "auto"
    virtual_fs_sources: list = field(default_factory=list)
    virtual_fs_request_mb: int = 4
    virtual_fs_auto_margin: float = 1.5


class _CountingUpstream(ThreadingHTTPServer):
    """Serves one blob at any path with Range support and counts requests."""

    daemon_threads = True

    def __init__(self, blob: bytes):
        super().__init__(("127.0.0.1", 0), _UpstreamHandler)
        self.blob = blob
        self.requests: list[tuple[str, str | None]] = []
        self.lock = threading.Lock()
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/file.mkv"


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def _record(self):
        with self.server.lock:
            self.server.requests.append((self.command, self.headers.get("Range")))

    def do_HEAD(self):  # noqa: N802
        self._record()
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.server.blob)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        self._record()
        blob = self.server.blob
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            lo, _, hi = rng[6:].partition("-")
            start = int(lo)
            end = int(hi) if hi else len(blob) - 1
            data = blob[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(blob)}")
        else:
            data = blob
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _fetch(port: int, path: str, rng: str | None, read: int | None = None):
    """Issue one request to the proxy; optionally read only ``read`` bytes then drop the connection."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Range": rng} if rng else {}
    conn.request("GET", path, headers=headers)
    resp = conn.getresponse()
    body = resp.read(read) if read is not None else resp.read()
    conn.close()
    return resp, body


# ---------------------------------------------------------------------------
# 1. resolution + planning
# ---------------------------------------------------------------------------


def test_resolve_follows_library_symlink_into_the_mount(tmp_path):
    mount = tmp_path / "mount"
    (mount / ".ids" / "a").mkdir(parents=True)
    target = mount / ".ids" / "a" / "1234"
    target.write_bytes(b"x")
    library = tmp_path / "library" / "Show (2020)"
    library.mkdir(parents=True)
    link = library / "Show - S01E01 [x264].mkv"
    link.symlink_to(target)

    src = VirtualFsSource(local_root=str(mount), url="http://dav.local:8080/")

    assert src.resolve(str(link)) == "http://dav.local:8080/.ids/a/1234"
    assert src.resolve(str(tmp_path / "elsewhere.mkv")) is None


def test_plan_is_off_unless_enabled_and_on_a_source(tmp_path):
    f = tmp_path / "v.mkv"
    f.write_bytes(b"0" * 1024)
    # "sampled" mode so this test only exercises the enabled/source gates, not
    # the bitrate rule (a 1 KiB file would always lose to the sequential pass).
    cfg = FakeConfig(
        virtual_fs_mode="sampled",
        virtual_fs_sources=[{"local_root": str(tmp_path), "url": "http://h"}],
    )

    assert virtual_fs.plan(str(f), cfg, 3000.0) is not None
    cfg.virtual_fs_enabled = False
    assert virtual_fs.plan(str(f), cfg, 3000.0) is None
    cfg.virtual_fs_enabled = True
    cfg.virtual_fs_sources = []
    assert virtual_fs.plan(str(f), cfg, 3000.0) is None


def test_plan_auto_keeps_sequential_for_low_bitrate_files(tmp_path):
    # 367 MB over 44 min ≈ 1.1 Mbps: 265 samples x 4 MiB (~1.1 GB) would cost
    # three times the file. Auto must say no; forced mode must still say yes.
    f = tmp_path / "sd.avi"
    f.write_bytes(b"0")
    cfg = FakeConfig(virtual_fs_sources=[{"local_root": str(tmp_path), "url": "http://h"}])
    with patch("os.path.getsize", return_value=367 * 1_000_000):
        assert virtual_fs.plan(str(f), cfg, 2642.0) is None
        cfg.virtual_fs_mode = "sampled"
        forced = virtual_fs.plan(str(f), cfg, 2642.0)
    assert forced is not None
    assert len(forced.sample_times) == 265


def test_plan_auto_samples_high_bitrate_files(tmp_path):
    # 7.12 GB over 3183 s (the measured Borgias episode): 319 x 4 MiB ≈ 1.3 GB
    # against 7.1 GB sequential — sampling wins with room to spare.
    f = tmp_path / "hd.mkv"
    f.write_bytes(b"0")
    cfg = FakeConfig(virtual_fs_sources=[{"local_root": str(tmp_path), "url": "http://h"}])
    with patch("os.path.getsize", return_value=7_119_104_627):
        p = virtual_fs.plan(str(f), cfg, 3183.168)
    assert p is not None
    assert len(p.sample_times) == 319
    assert p.sample_times[0] == 0 and p.sample_times[-1] == 3180
    assert p.estimated_sampled_bytes == 319 * 4 * MIB
    assert p.request_bytes == 4 * MIB


def test_plan_honours_sequential_mode(tmp_path):
    f = tmp_path / "hd.mkv"
    f.write_bytes(b"0")
    cfg = FakeConfig(
        virtual_fs_mode="sequential",
        virtual_fs_sources=[{"local_root": str(tmp_path), "url": "http://h"}],
    )
    with patch("os.path.getsize", return_value=7_000_000_000):
        assert virtual_fs.plan(str(f), cfg, 3000.0) is None


# ---------------------------------------------------------------------------
# 2. chunk proxy against a real upstream
# ---------------------------------------------------------------------------


@pytest.fixture
def upstream():
    blob = os.urandom(10 * MIB + 12345)
    server = _CountingUpstream(blob)
    yield server
    server.shutdown()


@pytest.fixture
def proxy():
    p = ChunkProxy(request_bytes=1 * MIB, cache_chunks=8)
    yield p
    p.shutdown()


def test_proxy_serves_exact_ranges_from_aligned_chunks(upstream, proxy):
    local = proxy.add(upstream.url, None)
    port = proxy.server_address[1]
    path = local[len(proxy.base_url) :]

    resp, body = _fetch(port, path, "bytes=1000-2023")
    assert resp.status == 206
    assert resp.getheader("Content-Range") == f"bytes 1000-2023/{len(upstream.blob)}"
    assert body == upstream.blob[1000:2024]

    # Size lookup + exactly one 1 MiB chunk: the request unit is the chunk,
    # not the 1 KiB the client wanted.
    gets = [r for r in upstream.requests if r[0] == "GET"]
    assert len(gets) == 1
    assert gets[0][1] == f"bytes=0-{MIB - 1}"


def test_proxy_caches_chunks_so_header_rereads_cost_nothing(upstream, proxy):
    local = proxy.add(upstream.url, None)
    port = proxy.server_address[1]
    path = local[len(proxy.base_url) :]

    _fetch(port, path, "bytes=0-4095")
    before = len([r for r in upstream.requests if r[0] == "GET"])
    # ffmpeg re-reads the container header on every -ss; the second read must
    # be served from the cache.
    _, body = _fetch(port, path, "bytes=100-4195")
    after = len([r for r in upstream.requests if r[0] == "GET"])

    assert body == upstream.blob[100:4196]
    assert after == before
    assert proxy.snapshot()[2] >= 1  # cache hits


def test_proxy_spans_chunk_boundaries_and_reports_bytes(upstream, proxy):
    local = proxy.add(upstream.url, None)
    port = proxy.server_address[1]
    path = local[len(proxy.base_url) :]

    start, end = MIB - 10, 2 * MIB + 9  # touches chunks 0, 1 and 2
    _, body = _fetch(port, path, f"bytes={start}-{end}")
    assert body == upstream.blob[start : end + 1]
    requests, fetched, _ = proxy.snapshot()
    assert requests == 1 + 3  # HEAD for size + three chunks
    assert fetched == 3 * MIB


def test_proxy_open_ended_range_stops_when_client_leaves(upstream, proxy):
    local = proxy.add(upstream.url, None)
    port = proxy.server_address[1]
    path = local[len(proxy.base_url) :]

    # Read a little from an open-ended range, then hang up — the way ffmpeg
    # does once it has the frame it seeked to.
    resp, body = _fetch(port, path, f"bytes={3 * MIB}-", read=2048)
    assert resp.status == 206
    assert body == upstream.blob[3 * MIB : 3 * MIB + 2048]

    # Give the handler a moment to notice the closed socket, then check that
    # it did not walk to the end of the 10 MiB file.
    import time

    time.sleep(0.3)
    gets = [r for r in upstream.requests if r[0] == "GET"]
    assert len(gets) <= 3  # at most the current chunk plus one in flight


def test_proxy_head_reports_full_size(upstream, proxy):
    local = proxy.add(upstream.url, None)
    conn = http.client.HTTPConnection("127.0.0.1", proxy.server_address[1], timeout=10)
    conn.request("HEAD", local[len(proxy.base_url) :])
    resp = conn.getresponse()
    resp.read()
    conn.close()
    assert resp.status == 200
    assert int(resp.getheader("Content-Length")) == len(upstream.blob)


def test_proxy_sends_basic_auth_upstream(upstream, proxy):
    seen: list[str | None] = []
    original = _UpstreamHandler.do_GET

    def spy(self):
        seen.append(self.headers.get("Authorization"))
        original(self)

    with patch.object(_UpstreamHandler, "do_GET", spy):
        local = proxy.add(upstream.url, VirtualFsSource("/m", "http://x", "admin", "s3cret").auth_header())
        _fetch(proxy.server_address[1], local[len(proxy.base_url) :], "bytes=0-10")
    assert seen and seen[0] == "Basic YWRtaW46czNjcmV0"


# ---------------------------------------------------------------------------
# 3. extract: per-sample runner calls and output naming
# ---------------------------------------------------------------------------


def _plan_for(tmp_path, duration_s=95.0):
    f = tmp_path / "v.mkv"
    f.write_bytes(b"0")
    src = VirtualFsSource(local_root=str(tmp_path), url="http://h:1")
    return virtual_fs.SamplePlan(
        video_file=str(f),
        source=src,
        source_url=src.resolve(str(f)),
        duration_s=duration_s,
        interval_s=10,
        request_bytes=4 * MIB,
        size_bytes=5_000_000_000,
    )


def test_extract_runs_one_seek_per_interval_and_names_frames(tmp_path):
    p = _plan_for(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        with open(kwargs["output_override"], "wb") as fh:
            fh.write(b"jpg")
        return 0, 0.1, 0.0, []

    with patch.object(virtual_fs, "get_proxy") as gp:
        gp.return_value.add.return_value = "http://127.0.0.1:1/t1"
        gp.return_value.snapshot.return_value = (0, 0, 0)
        rc, seconds, speed, stderr = virtual_fs.extract(p, str(out), fake_runner)

    assert rc == 0
    assert [c["pre_input_args"][2] for c in calls] == [str(t) for t in range(0, 90 + 1, 10)]
    assert all(c["input_override"] == "http://127.0.0.1:1/t1" for c in calls)
    assert all(c["use_skip"] is False and c["simple_run"] is True for c in calls)
    assert all(c["post_input_args"][:2] == ["-frames:v", "1"] for c in calls)
    assert sorted(os.listdir(out)) == [f"img-{i:06d}.jpg" for i in range(1, 11)]
    assert speed > 0


def test_extract_reports_failure_when_too_many_samples_fail(tmp_path):
    p = _plan_for(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    def flaky_runner(**kwargs):
        # 4 of 10 fail: well past the 5 % tolerance.
        if kwargs["pre_input_args"][2] in {"10", "20", "30", "40"}:
            return 1, 0.1, 0.0, ["error"]
        with open(kwargs["output_override"], "wb") as fh:
            fh.write(b"jpg")
        return 0, 0.1, 0.0, []

    with patch.object(virtual_fs, "get_proxy") as gp:
        gp.return_value.add.return_value = "http://127.0.0.1:1/t1"
        gp.return_value.snapshot.return_value = (0, 0, 0)
        rc, *_ = virtual_fs.extract(p, str(out), flaky_runner)
    assert rc != 0


def test_extract_raises_when_cancelled(tmp_path):
    p = _plan_for(tmp_path)
    with patch.object(virtual_fs, "get_proxy") as gp:
        gp.return_value.add.return_value = "http://127.0.0.1:1/t1"
        gp.return_value.snapshot.return_value = (0, 0, 0)
        with pytest.raises(virtual_fs.SamplingCancelled):
            virtual_fs.extract(p, str(tmp_path), lambda **kw: (0, 0, 0, []), cancel_check=lambda: True)
