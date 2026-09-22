"""Sampled reads for media on a virtual filesystem (rclone / InfiniDysk / Decypharr / zurg).

The normal path decodes a file front to back with ``-skip_frame nokey``.  On a
local disk that is the cheapest thing to do: one process, sequential I/O, and
the bytes are free.  On a FUSE mount backed by Usenet or a debrid service the
same pass costs the *whole file* in remote traffic and runs at mount speed —
measured on one host: a 7.1 GB 1080p episode took ~11 minutes and pulled 7.1 GB
from the provider to make 319 thumbnails whose keyframes total well under 1 GB.

Seeking straight to each keyframe cuts the client-side reads to a few MB per
thumbnail, but only if the *requests* are shaped for the server.  These backends
serve every HTTP range request by prefetching at least one batch of articles
(a few MiB) regardless of how small the range is, so the cost unit is the
request, not the byte.  ffmpeg's http demuxer re-reads the container header and
index on every ``-ss``, and an open-ended ``bytes=N-`` lets the server read
ahead without limit.  Measured against InfiniDysk: naive open-ended seeks cost
125–180 MB *per thumbnail*; bounded but small (128 KiB) chunks still paid the
per-request floor a dozen times per seek.

So this module puts a tiny local proxy between ffmpeg and the virtual-fs
source that (a) turns every request into aligned, fixed-size upstream chunks
sized to the server's floor (default 4 MiB), one request per chunk, and (b)
caches chunks so the header and index are fetched once per file, not once per
seek.  Each thumbnail then costs ~one upstream request.

Whether to sample at all is a bitrate question, not a filesystem one.  A
thumbnail costs one request-floor of bytes; a sequential pass costs the file.
Sampling wins when ``bytes-per-interval`` exceeds the floor — roughly above
3–5 Mbps at a 10 s interval.  Below that (SD material) the file is small and
the sequential pass is cheaper *and* faster, so ``auto`` keeps it.  Local files
never come here: on a real disk a per-frame seek loop measured ~10x slower than
one sequential pass, because process spawn dominates when the bytes are free.
"""

from __future__ import annotations

import base64
import os
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from loguru import logger

MIB = 1 << 20
UPSTREAM_TIMEOUT_S = 60
# 32 chunks at the default 4 MiB is 128 MiB: enough to keep the container
# header and index of a handful of concurrently processed files resident while
# cluster chunks (used once) age out.
DEFAULT_CACHE_CHUNKS = 32
# A sampled run is rejected (so the caller falls back to the sequential pass)
# when more than this fraction of thumbnails failed to extract.
MAX_FAILURE_FRACTION = 0.05
# Samples in flight at once by default; each is one short ffmpeg mostly
# waiting on the source, so this overlaps that wait rather than adding load.
DEFAULT_CONCURRENCY = 4
MAX_CONCURRENCY = 16
# Log proxy request/byte counters this often during a sampled run.
STATS_EVERY = 25
# Socket buffer on both ends of the proxy connection.  Bounds how far ahead of
# ffmpeg's reads the proxy can get (and therefore how many chunks it fetches
# that ffmpeg never consumes) — see ``_ChunkHandler.setup``.  512 KiB keeps
# that to well under one chunk; 64 KiB was measured to stall loopback
# throughput (a 4 MiB chunk took seconds to hand over).
SOCKET_BUFFER_BYTES = 512 * 1024


class SamplingCancelled(Exception):
    """Raised when ``cancel_check`` fires between samples."""


@dataclass(frozen=True)
class VirtualFsSource:
    """One virtual-fs mount and the HTTP/WebDAV root that mirrors it.

    ``local_root`` is where the mount exposes the tree (``/mnt/debrid/infinidysk``);
    ``url`` serves the same tree over HTTP (``http://host:8080``).  Library files
    are usually symlinks into the mount, so resolution follows them first.
    """

    local_root: str
    url: str
    user: str = ""
    password: str = ""

    def resolve(self, video_file: str) -> str | None:
        real = os.path.realpath(video_file)
        root = os.path.realpath(self.local_root).rstrip("/") + "/"
        if not real.startswith(root):
            return None
        rel = real[len(root) :]
        return self.url.rstrip("/") + "/" + urllib.parse.quote(rel)

    def auth_header(self) -> str | None:
        if not self.user and not self.password:
            return None
        token = base64.b64encode(f"{self.user}:{self.password}".encode()).decode("ascii")
        return f"Basic {token}"


def sources_from_config(config) -> list[VirtualFsSource]:
    out: list[VirtualFsSource] = []
    for entry in getattr(config, "virtual_fs_sources", None) or []:
        root = str(entry.get("local_root") or "").strip()
        url = str(entry.get("url") or "").strip()
        if root and url:
            out.append(
                VirtualFsSource(
                    local_root=root,
                    url=url,
                    user=str(entry.get("user") or ""),
                    password=str(entry.get("password") or ""),
                )
            )
    return out


@dataclass(frozen=True)
class SamplePlan:
    video_file: str
    source: VirtualFsSource
    source_url: str
    duration_s: float
    interval_s: int
    request_bytes: int
    size_bytes: int
    concurrency: int = DEFAULT_CONCURRENCY

    @property
    def sample_times(self) -> list[int]:
        # Matches what ``fps=1/interval:round=up`` yields on the sequential path:
        # one frame at every multiple of the interval inside the runtime.
        times = list(range(0, int(self.duration_s), self.interval_s))
        return times or [0]

    @property
    def estimated_sampled_bytes(self) -> int:
        return len(self.sample_times) * self.request_bytes

    @property
    def estimated_sequential_bytes(self) -> int:
        return self.size_bytes


def plan(video_file: str, config, duration_s: float | None) -> SamplePlan | None:
    """Decide whether ``video_file`` should be read by sampling, and how.

    Returns ``None`` when the file is not on a configured virtual-fs source, when
    the mode forces the sequential pass, or when ``auto`` finds sequential
    cheaper (low-bitrate material).
    """
    if not getattr(config, "virtual_fs_enabled", False):
        return None
    mode = str(getattr(config, "virtual_fs_mode", "auto") or "auto").lower()
    if mode == "sequential":
        return None
    sources = sources_from_config(config)
    if not sources or not duration_s or duration_s <= 0:
        return None

    source_url = None
    source = None
    for candidate in sources:
        source_url = candidate.resolve(video_file)
        if source_url:
            source = candidate
            break
    if source is None or source_url is None:
        return None

    try:
        size = os.path.getsize(video_file)
    except OSError:
        return None
    if size <= 0:
        return None

    request_bytes = max(1, int(getattr(config, "virtual_fs_request_mb", 4) or 4)) * MIB
    concurrency = int(getattr(config, "virtual_fs_concurrency", DEFAULT_CONCURRENCY) or DEFAULT_CONCURRENCY)
    result = SamplePlan(
        video_file=video_file,
        source=source,
        source_url=source_url,
        duration_s=float(duration_s),
        interval_s=max(1, int(config.thumbnail_interval)),
        request_bytes=request_bytes,
        size_bytes=size,
        concurrency=max(1, min(MAX_CONCURRENCY, concurrency)),
    )

    if mode == "auto":
        margin = float(getattr(config, "virtual_fs_auto_margin", 1.5) or 1.5)
        if result.estimated_sampled_bytes * margin >= result.estimated_sequential_bytes:
            logger.debug(
                "virtual-fs: sequential pass is cheaper for '{}' ({} MiB file vs ~{} MiB sampled); keeping it",
                os.path.basename(video_file),
                result.estimated_sequential_bytes // MIB,
                result.estimated_sampled_bytes // MIB,
            )
            return None
    return result


class _ChunkHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        # ffmpeg asks for open-ended ranges (``bytes=N-``) and reads only what
        # it needs before seeking again.  With default loopback buffers the
        # kernel would happily absorb megabytes we write ahead of its reads,
        # and every one of those bytes is an upstream chunk fetched for
        # nothing (measured: 3–7 requests per thumbnail instead of 1–2).  A
        # small send buffer here, plus a small receive buffer on ffmpeg's side
        # (``SOCKET_BUFFER_BYTES`` in :func:`extract`), makes writes block
        # within the current chunk, so the next chunk is fetched only once
        # ffmpeg is actually reading it.
        super().setup()
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_BUFFER_BYTES)

    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler signature
        logger.trace("virtual-fs proxy: " + format, *args)

    def _target(self):
        token = self.path.lstrip("/").split("/", 1)[0].split("?", 1)[0]
        return self.server.targets.get(token)

    def send_response(self, code, message=None):
        # One request per connection, and say so.  ffmpeg >= 8 "soft-seeks"
        # by sending the next Range request on a connection it believes is
        # still open; if the server closed it silently the seek fails, the
        # demuxer is left at EOF, and it gives up on the index — every -ss
        # then degrades to a linear scan of the file (measured: 43 requests /
        # 169 MiB for one thumbnail instead of one).  Advertising the close
        # makes it open a fresh connection per seek, which is what we want.
        super().send_response(code, message)
        self.send_header("Connection", "close")
        self.close_connection = True

    def do_HEAD(self):  # noqa: N802 - http.server naming
        target = self._target()
        if target is None:
            self.send_error(404)
            return
        try:
            size = self.server.size_of(target)
        except urllib.error.URLError as exc:
            self.send_error(502, str(exc.reason))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.end_headers()

    def do_GET(self):  # noqa: N802 - http.server naming
        target = self._target()
        if target is None:
            self.send_error(404)
            return
        try:
            size = self.server.size_of(target)
        except urllib.error.URLError as exc:
            self.send_error(502, str(exc.reason))
            return

        start, end = 0, size - 1
        range_header = self.headers.get("Range", "")
        ranged = range_header.startswith("bytes=")
        if ranged:
            spec = range_header[len("bytes=") :].split(",", 1)[0]
            lo, _, hi = spec.partition("-")
            if lo:
                start = int(lo)
                if hi:
                    end = min(int(hi), size - 1)
            elif hi:  # suffix range: last N bytes
                start = max(0, size - int(hi))
        if start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        self.send_response(206 if ranged else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if ranged:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        chunk = self.server.request_bytes
        pos = start
        try:
            while pos <= end:
                idx = pos // chunk
                data = self.server.chunk(target, idx, size)
                lo = pos - idx * chunk
                hi = min(len(data), end - idx * chunk + 1)
                self.wfile.write(data[lo:hi])
                self.wfile.flush()
                pos = idx * chunk + hi
        except (BrokenPipeError, ConnectionResetError):
            # ffmpeg closes as soon as it has its frame; the next chunk was
            # never fetched, which is the whole point.
            return
        except urllib.error.URLError as exc:
            logger.debug("virtual-fs proxy: upstream failed mid-range for {}: {}", target.url, exc)
            return


@dataclass(frozen=True)
class _Target:
    url: str
    auth: str | None


class ChunkProxy(ThreadingHTTPServer):
    """Local HTTP front for one or more virtual-fs files.

    Every upstream request is exactly one aligned chunk of ``request_bytes``,
    and chunks are cached LRU across requests so a file's header and index cost
    one request for the whole run rather than one per seek.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, request_bytes: int = 4 * MIB, cache_chunks: int = DEFAULT_CACHE_CHUNKS):
        super().__init__(("127.0.0.1", 0), _ChunkHandler)
        self.request_bytes = max(64 * 1024, int(request_bytes))
        self.cache_chunks = max(2, int(cache_chunks))
        self.targets: dict[str, _Target] = {}
        self._sizes: dict[str, int] = {}
        self._cache: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._lock = threading.Lock()
        self.upstream_requests = 0
        self.upstream_bytes = 0
        self.cache_hits = 0
        self._thread = threading.Thread(target=self.serve_forever, name="virtual-fs-proxy", daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def add(self, url: str, auth: str | None) -> str:
        """Register an upstream file; returns the local URL ffmpeg should open."""
        token = f"t{len(self.targets) + 1}_{abs(hash(url)) % 10_000_000}"
        with self._lock:
            self.targets[token] = _Target(url=url, auth=auth)
        return f"{self.base_url}/{token}"

    def snapshot(self) -> tuple[int, int, int]:
        with self._lock:
            return self.upstream_requests, self.upstream_bytes, self.cache_hits

    # -- upstream --------------------------------------------------------

    def _request(self, target: _Target, method: str, range_header: str | None):
        req = urllib.request.Request(target.url, method=method)
        if target.auth:
            req.add_header("Authorization", target.auth)
        if range_header:
            req.add_header("Range", range_header)
        req.add_header("User-Agent", "media-preview-generator virtual-fs")
        return urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_S)  # noqa: S310 - operator-configured URL

    def size_of(self, target: _Target) -> int:
        with self._lock:
            cached = self._sizes.get(target.url)
        if cached is not None:
            return cached
        # Some WebDAV servers answer HEAD without a length; a 1-byte range
        # request always yields the total in Content-Range.
        size: int | None = None
        try:
            with self._request(target, "HEAD", None) as resp:
                length = resp.headers.get("Content-Length")
                size = int(length) if length else None
        except urllib.error.HTTPError as exc:
            if exc.code not in (405, 501):
                raise
        if size is None:
            with self._request(target, "GET", "bytes=0-0") as resp:
                content_range = resp.headers.get("Content-Range", "")
                size = int(content_range.rsplit("/", 1)[-1])
                resp.read()
        with self._lock:
            self.upstream_requests += 1
            self._sizes[target.url] = size
        return size

    def chunk(self, target: _Target, idx: int, size: int) -> bytes:
        key = (target.url, idx)
        with self._lock:
            data = self._cache.get(key)
            if data is not None:
                self._cache.move_to_end(key)
                self.cache_hits += 1
                return data
        start = idx * self.request_bytes
        end = min(start + self.request_bytes, size) - 1
        with self._request(target, "GET", f"bytes={start}-{end}") as resp:
            data = resp.read()
        with self._lock:
            self.upstream_requests += 1
            self.upstream_bytes += len(data)
            self._cache[key] = data
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_chunks:
                self._cache.popitem(last=False)
        return data


_proxy: ChunkProxy | None = None
_proxy_lock = threading.Lock()


def get_proxy(request_bytes: int) -> ChunkProxy:
    """One proxy per process; its chunk size is fixed by the first caller."""
    global _proxy
    with _proxy_lock:
        if _proxy is None or _proxy.request_bytes != request_bytes:
            if _proxy is not None:
                _proxy.shutdown()
            _proxy = ChunkProxy(request_bytes=request_bytes)
        return _proxy


def extract(
    sample_plan: SamplePlan,
    output_folder: str,
    run_ffmpeg,
    progress_callback=None,
    cancel_check=None,
    pause_check=None,
) -> tuple[int, float, float, list[str]]:
    """Extract one thumbnail per interval by seeking through the chunk proxy.

    ``run_ffmpeg`` is the per-file runner from :mod:`ffmpeg_runner`; each call
    keeps that file's hardware-acceleration and filter decisions and only swaps
    the input for the proxy URL with a ``-ss`` seek.  Frames are written as
    ``img-%06d.jpg`` so the caller's rename step needs no change.

    Returns ``(returncode, seconds, speed, stderr_lines)`` like the runner does.
    A non-zero code means too many samples failed; the caller's existing retry
    cascade then runs the sequential pass.
    """
    proxy = get_proxy(sample_plan.request_bytes)
    local_url = proxy.add(sample_plan.source_url, sample_plan.source.auth_header())
    req0, bytes0, hits0 = proxy.snapshot()

    times = sample_plan.sample_times
    concurrency = max(1, sample_plan.concurrency)
    started = time.time()
    failures = 0
    stderr_tail: list[str] = []
    logger.info(
        "virtual-fs sampled read for {}: {} thumbnails, ~{} MiB in {} MiB requests vs {} MiB sequential, {} in flight",
        os.path.basename(sample_plan.video_file),
        len(times),
        sample_plan.estimated_sampled_bytes // MIB,
        sample_plan.request_bytes // MIB,
        sample_plan.estimated_sequential_bytes // MIB,
        concurrency,
    )
    if progress_callback:
        progress_callback(0, 0, sample_plan.duration_s, "0.0x", media_file=sample_plan.video_file)

    def one(i: int, t: int) -> tuple[int, int, list[str]]:
        out = os.path.join(output_folder, f"img-{i + 1:06d}.jpg")
        t_start = time.time()
        req_before, bytes_before, _ = proxy.snapshot()
        rc, _, _, lines = run_ffmpeg(
            use_skip=False,
            input_override=local_url,
            pre_input_args=[
                "-noaccurate_seek",
                "-ss",
                str(t),
                "-seekable",
                "1",
                "-recv_buffer_size",
                str(SOCKET_BUFFER_BYTES),
            ],
            # ``-fps_mode passthrough``: without it the output's frame sync
            # drops every decoded frame timestamped before the seek target,
            # so ffmpeg quietly decodes from the keyframe all the way to ``t``
            # (measured: 80–120 frames and 8–13 MB per thumbnail).  Passing
            # frames through emits the keyframe the seek landed on — the same
            # frame the sequential ``-skip_frame nokey`` pass would use — and
            # the process ends after ~4 decoded frames and one chunk.
            post_input_args=["-frames:v", "1", "-update", "1", "-fps_mode", "passthrough"],
            output_override=out,
            simple_run=True,
        )
        ok = rc == 0 and os.path.exists(out)
        req_after, bytes_after, _ = proxy.snapshot()
        # Exact only when one sample is in flight; still a useful ceiling otherwise.
        logger.debug(
            "virtual-fs sample t={}s: rc={} {:.1f}s, {} upstream requests / {} MiB while it ran",
            t,
            rc,
            time.time() - t_start,
            req_after - req_before,
            (bytes_after - bytes_before) // MIB,
        )
        return t, (0 if ok else 1), (lines[-3:] if not ok else [])

    # Each sample is a short process that spends most of its life waiting on
    # the source (a 4 MiB request against a Usenet-backed server takes
    # seconds), so a few in flight overlap that wait without contending for
    # anything local.  Submission stays in order and bounded so cancel/pause
    # are honoured within one batch.
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="virtual-fs-sample") as pool:
        pending: set = set()
        it = iter(enumerate(times))
        exhausted = False
        while not exhausted or pending:
            while not exhausted and len(pending) < concurrency:
                if cancel_check and cancel_check():
                    for f in pending:
                        f.cancel()
                    raise SamplingCancelled(sample_plan.video_file)
                while pause_check and pause_check():
                    time.sleep(0.5)
                try:
                    i, t = next(it)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(pool.submit(one, i, t))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                t, failed, lines = f.result()
                done += 1
                if failed:
                    failures += 1
                    stderr_tail.extend(lines)
                    del stderr_tail[:-30]
                elapsed = time.time() - started
                if progress_callback:
                    speed = (done * sample_plan.interval_s) / elapsed if elapsed > 0 else 0.0
                    remaining = (len(times) - done) * (elapsed / done) if done else None
                    progress_callback(
                        100.0 * done / len(times),
                        t,
                        sample_plan.duration_s,
                        f"{speed:.1f}x",
                        remaining,
                    )
                if done % STATS_EVERY == 0 and done < len(times):
                    req_now, bytes_now, hits_now = proxy.snapshot()
                    logger.info(
                        "virtual-fs sampled read for {}: {}/{} thumbnails, {:.1f} s/thumbnail, "
                        "{} upstream requests, {} MiB fetched, {} cache hits",
                        os.path.basename(sample_plan.video_file),
                        done,
                        len(times),
                        elapsed / done,
                        req_now - req0,
                        (bytes_now - bytes0) // MIB,
                        hits_now - hits0,
                    )

    seconds = time.time() - started
    speed_val = sample_plan.duration_s / seconds if seconds > 0 else 0.0
    req1, bytes1, hits1 = proxy.snapshot()
    produced = len(times) - failures
    logger.info(
        "virtual-fs sampled read done for {}: {}/{} thumbnails in {:.1f}s ({:.1f}x), "
        "{} upstream requests, {} MiB fetched, {} cache hits",
        os.path.basename(sample_plan.video_file),
        produced,
        len(times),
        seconds,
        speed_val,
        req1 - req0,
        (bytes1 - bytes0) // MIB,
        hits1 - hits0,
    )
    too_many_failures = failures > max(1, int(len(times) * MAX_FAILURE_FRACTION))
    rc = 1 if (produced == 0 or too_many_failures) else 0
    return rc, seconds, speed_val, stderr_tail
