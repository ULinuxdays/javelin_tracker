/* eslint-disable no-undef */
/* global React, ReactDOM, Plotly */

(function () {
  'use strict';

  if (typeof React === 'undefined' || typeof ReactDOM === 'undefined') {
    console.error('BiomechanicsViewer: React not loaded.');
    return;
  }

  const { useCallback, useEffect, useMemo, useRef, useState } = React;

  function showToast(message, type = 'info') {
    const toast = document.getElementById('appToast');
    if (!toast) return;
    toast.classList.remove('success', 'error');
    if (type === 'success') toast.classList.add('success');
    if (type === 'error') toast.classList.add('error');
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3500);
  }

  function formatNumber(value, digits = 2) {
    const num = typeof value === 'number' ? value : Number(value);
    if (!Number.isFinite(num)) return '—';
    return num.toFixed(digits);
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const text = await response.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch (_err) {
      data = null;
    }
    return { response, data, rawText: text };
  }

  function SeverityBadge({ status }) {
    const norm = String(status || '').toLowerCase();
    const cls =
      norm === 'good'
        ? 'severity-badge good'
        : norm === 'warning'
          ? 'severity-badge warning'
          : norm === 'poor'
            ? 'severity-badge poor'
            : 'severity-badge';

    const label =
      norm === 'good'
        ? 'good'
        : norm === 'warning'
          ? 'warning'
          : norm === 'poor'
            ? 'poor'
            : '—';

    return (
      <span className={cls}>
        <span className="severity-dot" aria-hidden="true" />
        {label}
      </span>
    );
  }

  function SkeletonLine({ style }) {
    return <div className="skeleton-line" style={style} />;
  }

  function IssuesCard({ comparison }) {
    const issues = Array.isArray(comparison?.top_issues) ? comparison.top_issues : [];
    const top = issues.slice(0, 5);
    const overall = comparison?.overall_score;

    return (
      <div className="biomech-card">
        <div className="biomech-card__header">
          <div>
            <h4>Issues</h4>
            <p className="muted small">Top deviations vs elite reference.</p>
          </div>
          <div className="biomech-score-pill" title="Overall score">
            {overall != null ? formatNumber(overall, 0) : '—'}
          </div>
        </div>
        {!top.length ? (
          <p className="muted small">No comparison issues available yet.</p>
        ) : (
          <ul className="biomech-issue-list">
            {top.map((issue, idx) => (
              <li key={idx} className="biomech-issue">
                <div className="biomech-issue__main">
                  <div className="biomech-issue__metric">{issue.metric || 'metric'}</div>
                  <div className="muted small">
                    z={formatNumber(issue.z_score, 2)} · score={formatNumber(issue.score, 0)}
                  </div>
                </div>
                <SeverityBadge status={issue.status} />
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  function _angleSeries(anglesData, angleName) {
    const rows = Array.isArray(anglesData) ? anglesData : [];
    const filtered = rows.filter((row) => row?.angle_name === angleName);
    const x = [];
    const y = [];
    for (const row of filtered) {
      const t = Number(row.timestamp_ms);
      const v = Number(row.value_degrees);
      if (!Number.isFinite(t) || !Number.isFinite(v)) continue;
      if (row.valid === false) continue;
      x.push(t);
      y.push(v);
    }
    return { x, y };
  }

  function _phaseWindowsMs(metrics, xFallback) {
    const pb = metrics?.phase_boundaries || {};
    const meta = metrics?.video_metadata || {};
    const fps = Number(meta.fps) || 30;
    const approach = Number(pb.approach_start_frame ?? 0);
    const delivery = Number(pb.delivery_start_frame ?? 0);
    const release = Number(pb.release_frame ?? 0);
    const frameToMs = (f) => (Number.isFinite(f) ? (f / fps) * 1000.0 : 0);

    const startMs = xFallback?.[0] ?? 0;
    const endMs = xFallback?.[xFallback.length - 1] ?? frameToMs(release + 10);
    const approachStart = frameToMs(approach);
    const deliveryStart = frameToMs(delivery);
    const releaseStart = frameToMs(release);

    return {
      approach: [Math.max(startMs, approachStart), Math.max(startMs, deliveryStart)],
      delivery: [Math.max(startMs, deliveryStart), Math.max(startMs, releaseStart)],
      release: [Math.max(startMs, releaseStart), Math.max(startMs, endMs)],
    };
  }

  function _refStat(reference, phase, metricName) {
    const key = `${phase}.${metricName}`;
    const entry = reference?.metrics?.[key] || reference?.[key] || null;
    if (!entry) return null;
    const mean = Number(entry.mean);
    const std = Number(entry.std);
    if (!Number.isFinite(mean)) return null;
    return { mean, std: Number.isFinite(std) ? std : 0, unit: entry.unit || 'degrees' };
  }

  function AngleChart({ title, angleName, metrics, reference, referenceMetricName }) {
    const plotRef = useRef(null);
    const anglesData = metrics?.angles?.data;
    const series = useMemo(() => _angleSeries(anglesData, angleName), [anglesData, angleName]);
    const windows = useMemo(() => _phaseWindowsMs(metrics, series.x), [metrics, series.x]);

    useEffect(() => {
      if (typeof Plotly === 'undefined') return;
      const node = plotRef.current;
      if (!node) return;

      const traces = [
        {
          x: series.x,
          y: series.y,
          type: 'scatter',
          mode: 'lines',
          name: 'Athlete',
          line: { color: '#2563eb', width: 2 },
          hovertemplate: '%{y:.2f}°<br>%{x:.0f}ms<extra></extra>',
        },
      ];

      const shapes = [];
      if (reference && referenceMetricName) {
        for (const phase of ['approach', 'delivery', 'release']) {
          const stat = _refStat(reference, phase, referenceMetricName);
          if (!stat) continue;
          const [x0, x1] = windows[phase] || [];
          if (!Number.isFinite(x0) || !Number.isFinite(x1) || x1 <= x0) continue;
          const lo = stat.mean - (stat.std || 0);
          const hi = stat.mean + (stat.std || 0);
          if (!Number.isFinite(lo) || !Number.isFinite(hi)) continue;
          shapes.push({
            type: 'rect',
            xref: 'x',
            yref: 'y',
            x0,
            x1,
            y0: lo,
            y1: hi,
            fillcolor: 'rgba(16,185,129,0.12)',
            line: { width: 0 },
            layer: 'below',
          });
          shapes.push({
            type: 'line',
            xref: 'x',
            yref: 'y',
            x0,
            x1,
            y0: stat.mean,
            y1: stat.mean,
            line: { color: 'rgba(16,185,129,0.55)', width: 1, dash: 'dot' },
            layer: 'below',
          });
        }
      }

      const layout = {
        margin: { l: 38, r: 16, t: 30, b: 30 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'time (ms)', tickformat: '.0f', gridcolor: 'rgba(148,163,184,0.25)' },
        yaxis: { title: 'degrees', gridcolor: 'rgba(148,163,184,0.25)' },
        shapes,
        showlegend: false,
        title: { text: title, font: { size: 14 } },
      };

      Plotly.react(node, traces, layout, { displayModeBar: false, responsive: true });
      return () => Plotly.purge(node);
    }, [series.x, series.y, reference, referenceMetricName, title, windows]);

    const hasData = series.x.length > 1;
    return (
      <div className="biomech-card">
        <div className="biomech-card__header">
          <div>
            <h4>{title}</h4>
            <p className="muted small">
              Athlete angle trajectory{referenceMetricName ? ' + elite band (mean±std)' : ''}.
            </p>
          </div>
        </div>
        {!hasData ? (
          <p className="muted small">No valid angle samples available.</p>
        ) : (
          <div ref={plotRef} className="plotly-chart" />
        )}
      </div>
    );
  }

  function _coerceLandmarkTuple(value) {
    if (!value) return null;
    if (Array.isArray(value)) {
      const x = Number(value[0]);
      const y = Number(value[1]);
      const z = Number(value[2]);
      const c = value.length >= 4 ? Number(value[3]) : 1.0;
      return { x, y, z: Number.isFinite(z) ? z : 0, c: Number.isFinite(c) ? c : 1.0 };
    }
    if (typeof value === 'object') {
      const x = Number(value.x);
      const y = Number(value.y);
      const z = Number(value.z);
      const c = Number(value.visibility ?? value.confidence ?? value.c ?? 1.0);
      return { x, y, z: Number.isFinite(z) ? z : 0, c: Number.isFinite(c) ? c : 1.0 };
    }
    return null;
  }

  function _toPixels(point, naturalW, naturalH, canvasW, canvasH) {
    if (!point) return null;
    const x = point.x;
    const y = point.y;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    const px = x >= 0 && x <= 1 ? x * naturalW : x;
    const py = y >= 0 && y <= 1 ? y * naturalH : y;
    return {
      x: (px / naturalW) * canvasW,
      y: (py / naturalH) * canvasH,
      c: Number.isFinite(point.c) ? point.c : 1.0,
    };
  }

  function _angleDeg(p1, p2, p3) {
    if (!p1 || !p2 || !p3) return NaN;
    const v1x = p1.x - p2.x;
    const v1y = p1.y - p2.y;
    const v2x = p3.x - p2.x;
    const v2y = p3.y - p2.y;
    const n1 = Math.hypot(v1x, v1y);
    const n2 = Math.hypot(v2x, v2y);
    if (!Number.isFinite(n1) || !Number.isFinite(n2) || n1 === 0 || n2 === 0) return NaN;
    const dot = v1x * v2x + v1y * v2y;
    const cos = Math.max(-1, Math.min(1, dot / (n1 * n2)));
    return (Math.acos(cos) * 180) / Math.PI;
  }

  const POSE_EDGES = [
    [11, 12],
    [11, 13],
    [13, 15],
    [12, 14],
    [14, 16],
    [23, 24],
    [11, 23],
    [12, 24],
    [23, 25],
    [25, 27],
    [24, 26],
    [26, 28],
  ];

  // Hide very low-confidence joints/edges. This prevents the overlay from
  // "locking" onto random background pixels when a limb is occluded.
  const MIN_DRAW_CONF = 0.25;

  function VideoWithOverlay({ sessionId, onFps, seekFrameRef, analysisWindow, className }) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const stageRef = useRef(null);
    const skeletonStageRef = useRef(null);
    const skeletonCanvasRef = useRef(null);
    const rafRef = useRef(null);
    const [pose, setPose] = useState(null);
    const [poseError, setPoseError] = useState(null);
    const [poseLoading, setPoseLoading] = useState(false);
    const [stride, setStride] = useState(2);
    const [playbackRate, setPlaybackRate] = useState(0.25);
    const [autoSeekDone, setAutoSeekDone] = useState(false);
    const [isStageFullscreen, setIsStageFullscreen] = useState(false);

    const fps = Number(pose?.video_metadata?.fps) || 30;
    const frames = Array.isArray(pose?.frames) ? pose.frames : [];

    const timestampsMs = useMemo(() => frames.map((f) => Number(f.timestamp_ms)).filter((t) => Number.isFinite(t)), [frames]);

    const seekToFrame = useCallback(
      (frameIdx) => {
        const v = videoRef.current;
        if (!v) return;
        const idx = Number(frameIdx);
        if (!Number.isFinite(idx) || fps <= 0) return;
        v.currentTime = idx / fps;
        v.pause();
      },
      [fps]
    );

    useEffect(() => {
      if (seekFrameRef && typeof seekFrameRef === 'object') {
        seekFrameRef.current = seekToFrame;
      }
    }, [seekFrameRef, seekToFrame]);

    useEffect(() => {
      onFps?.(fps);
    }, [fps, onFps]);

    useEffect(() => {
      const video = videoRef.current;
      if (!video) return;
      if (autoSeekDone) return;
      const startFrame = Number(analysisWindow?.start_frame);
      if (!Number.isFinite(startFrame) || fps <= 0) return;
      // Auto-focus the coach on the engaged delivery→release window.
      video.currentTime = Math.max(0, startFrame / fps);
      setAutoSeekDone(true);
    }, [analysisWindow?.start_frame, autoSeekDone, fps]);

    useEffect(() => {
      const video = videoRef.current;
      if (!video) return;
      const rate = Number(playbackRate);
      if (!Number.isFinite(rate) || rate <= 0) return;
      video.playbackRate = rate;
    }, [playbackRate]);

    const findNearestFrame = useCallback(
      (tMs) => {
        if (!frames.length) return null;
        if (timestampsMs.length === frames.length) {
          let lo = 0;
          let hi = timestampsMs.length - 1;
          while (lo < hi) {
            const mid = Math.floor((lo + hi) / 2);
            if (timestampsMs[mid] < tMs) lo = mid + 1;
            else hi = mid;
          }
          const idx = Math.max(0, Math.min(lo, frames.length - 1));
          const prev = Math.max(0, idx - 1);
          const a = timestampsMs[prev];
          const b = timestampsMs[idx];
          if (Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - tMs) < Math.abs(b - tMs)) {
            return frames[prev];
          }
          return frames[idx];
        }
        // Fallback: frame index ~ time * fps.
        const idx = Math.round((tMs / 1000.0) * fps);
        return frames[Math.max(0, Math.min(idx, frames.length - 1))];
      },
      [fps, frames, timestampsMs]
    );

    const draw = useCallback(() => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const skeletonCanvas = skeletonCanvasRef.current;
      const skeletonStage = skeletonStageRef.current;
      if (!video || !canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const rect = video.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const cw = Math.max(1, Math.round(rect.width * dpr));
      const ch = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== cw || canvas.height !== ch) {
        canvas.width = cw;
        canvas.height = ch;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!frames.length) return;

      const currentMs = (video.currentTime || 0) * 1000.0;
      const frame = findNearestFrame(currentMs);
      if (!frame || !Array.isArray(frame.landmarks)) return;

      const naturalW = Number(pose?.video_metadata?.width) || video.videoWidth || canvas.width;
      const naturalH = Number(pose?.video_metadata?.height) || video.videoHeight || canvas.height;

      const pts = frame.landmarks.map(_coerceLandmarkTuple);
      const px = pts.map((p) => _toPixels(p, naturalW, naturalH, canvas.width, canvas.height));

      // Draw edges.
      ctx.lineWidth = 2;
      for (const [a, b] of POSE_EDGES) {
        const p1 = px[a];
        const p2 = px[b];
        if (!p1 || !p2) continue;
        const conf = Math.min(Number(p1.c), Number(p2.c));
        if (!Number.isFinite(conf) || conf < MIN_DRAW_CONF) continue;
        const alpha = Math.min(0.95, 0.15 + 0.85 * Math.max(0, Math.min(1, conf)));
        ctx.strokeStyle = `rgba(59,130,246,${alpha})`;
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }

      // Draw joints.
      for (const p of px) {
        if (!p) continue;
        const conf = Number(p.c);
        if (!Number.isFinite(conf) || conf < MIN_DRAW_CONF) continue;
        const alpha = Number.isFinite(conf) ? Math.min(0.95, 0.12 + 0.88 * Math.max(0, Math.min(1, conf))) : 0.25;
        ctx.fillStyle = `rgba(15,23,42,${alpha})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3.0 * dpr, 0, Math.PI * 2);
        ctx.fill();
      }

      // Angles: right elbow + right shoulder.
      const rightShoulder = px[12];
      const rightElbow = px[14];
      const rightWrist = px[16];
      const rightHip = px[24];
      const elbowDeg = _angleDeg(rightShoulder, rightElbow, rightWrist);
      const shoulderDeg = _angleDeg(rightElbow, rightShoulder, rightHip);

      ctx.fillStyle = 'rgba(15,23,42,0.8)';
      ctx.font = `${12 * dpr}px Manrope, system-ui, sans-serif`;
      ctx.fillText(`Elbow: ${Number.isFinite(elbowDeg) ? elbowDeg.toFixed(0) + '°' : '—'}`, 12 * dpr, 18 * dpr);
      ctx.fillText(
        `Shoulder: ${Number.isFinite(shoulderDeg) ? shoulderDeg.toFixed(0) + '°' : '—'}`,
        12 * dpr,
        36 * dpr
      );

      // Skeleton-only side canvas (no background video).
      if (skeletonCanvas && skeletonStage) {
        const stageRect = skeletonStage.getBoundingClientRect();
        const sw = Math.max(1, Math.round(stageRect.width * dpr));
        const sh = Math.max(1, Math.round(stageRect.height * dpr));
        if (skeletonCanvas.width !== sw || skeletonCanvas.height !== sh) {
          skeletonCanvas.width = sw;
          skeletonCanvas.height = sh;
        }
        const sctx = skeletonCanvas.getContext('2d');
        if (sctx) {
          sctx.fillStyle = '#0b1220';
          sctx.fillRect(0, 0, skeletonCanvas.width, skeletonCanvas.height);

          const px2 = pts.map((p) => _toPixels(p, naturalW, naturalH, skeletonCanvas.width, skeletonCanvas.height));

          sctx.lineWidth = 2;
          for (const [a, b] of POSE_EDGES) {
            const p1 = px2[a];
            const p2 = px2[b];
            if (!p1 || !p2) continue;
            const conf = Math.min(Number(p1.c), Number(p2.c));
            if (!Number.isFinite(conf) || conf < MIN_DRAW_CONF) continue;
            const alpha = Math.min(0.98, 0.18 + 0.82 * Math.max(0, Math.min(1, conf)));
            sctx.strokeStyle = `rgba(34,211,238,${alpha})`;
            sctx.beginPath();
            sctx.moveTo(p1.x, p1.y);
            sctx.lineTo(p2.x, p2.y);
            sctx.stroke();
          }

          for (const p of px2) {
            if (!p) continue;
            const conf = Number(p.c);
            if (!Number.isFinite(conf) || conf < MIN_DRAW_CONF) continue;
            const alpha = Number.isFinite(conf) ? Math.min(0.98, 0.18 + 0.82 * Math.max(0, Math.min(1, conf))) : 0.25;
            sctx.fillStyle = `rgba(248,250,252,${alpha})`;
            sctx.beginPath();
            sctx.arc(p.x, p.y, 3.0 * dpr, 0, Math.PI * 2);
            sctx.fill();
          }
        }
      }
    }, [findNearestFrame, frames, pose?.video_metadata?.height, pose?.video_metadata?.width]);

    const toggleFullscreen = useCallback(() => {
      const target = stageRef.current;
      if (!target) return;

      const fsEl = document.fullscreenElement || document.webkitFullscreenElement || null;
      if (fsEl) {
        const exit = document.exitFullscreen || document.webkitExitFullscreen;
        if (exit) exit.call(document);
        return;
      }

      const req = target.requestFullscreen || target.webkitRequestFullscreen;
      if (req) req.call(target);
    }, []);

    const startLoop = useCallback(() => {
      const loop = () => {
        draw();
        rafRef.current = window.requestAnimationFrame(loop);
      };
      if (rafRef.current) window.cancelAnimationFrame(rafRef.current);
      rafRef.current = window.requestAnimationFrame(loop);
    }, [draw]);

    const stopLoop = useCallback(() => {
      if (rafRef.current) window.cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }, []);

    useEffect(() => {
      const video = videoRef.current;
      if (!video) return;
      const onPlay = () => startLoop();
      const onPause = () => {
        stopLoop();
        draw();
      };
      const onSeek = () => draw();
      const onTimeUpdate = () => draw();
      const onLoaded = () => draw();

      video.addEventListener('play', onPlay);
      video.addEventListener('pause', onPause);
      video.addEventListener('seeked', onSeek);
      video.addEventListener('timeupdate', onTimeUpdate);
      video.addEventListener('loadedmetadata', onLoaded);
      return () => {
        video.removeEventListener('play', onPlay);
        video.removeEventListener('pause', onPause);
        video.removeEventListener('seeked', onSeek);
        video.removeEventListener('timeupdate', onTimeUpdate);
        video.removeEventListener('loadedmetadata', onLoaded);
        stopLoop();
      };
    }, [draw, startLoop, stopLoop]);

    useEffect(() => {
      const onFs = () => {
        const fsEl = document.fullscreenElement || document.webkitFullscreenElement || null;
        setIsStageFullscreen(Boolean(fsEl && stageRef.current && fsEl === stageRef.current));
        draw();
      };
      document.addEventListener('fullscreenchange', onFs);
      document.addEventListener('webkitfullscreenchange', onFs);
      window.addEventListener('resize', onFs);
      return () => {
        document.removeEventListener('fullscreenchange', onFs);
        document.removeEventListener('webkitfullscreenchange', onFs);
        window.removeEventListener('resize', onFs);
      };
    }, [draw]);

    useEffect(() => {
      let cancelled = false;
      async function loadPose() {
        setPoseLoading(true);
        setPoseError(null);
        try {
          const { response, data } = await fetchJson(`/api/sessions/${sessionId}/biomechanics/pose?stride=${stride}`);
          if (!response.ok) throw new Error(data?.error || 'Unable to load pose data.');
          if (!cancelled) setPose(data);
        } catch (err) {
          if (!cancelled) setPoseError(err?.message || 'Unable to load pose data.');
        } finally {
          if (!cancelled) setPoseLoading(false);
        }
      }
      loadPose();
      return () => {
        cancelled = true;
      };
    }, [sessionId, stride]);

    return (
      <div className={`biomech-card${className ? ` ${className}` : ''}`}>
        <div className="biomech-card__header">
          <div>
            <h4>Video</h4>
            <p className="muted small">Left: original video + overlay · Right: skeleton-only view.</p>
          </div>
          <div className="biomech-actions no-print">
            <button type="button" className="btn btn-ghost" onClick={toggleFullscreen}>
              {isStageFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            </button>
            <label className="input-label small">
              Speed
              <select className="input" value={playbackRate} onChange={(e) => setPlaybackRate(Number(e.target.value) || 1)}>
                <option value={0.25}>0.25×</option>
                <option value={0.5}>0.5×</option>
                <option value={0.75}>0.75×</option>
                <option value={1}>1×</option>
                <option value={1.25}>1.25×</option>
              </select>
            </label>
            <label className="input-label small">
              Stride
              <select className="input" value={stride} onChange={(e) => setStride(Number(e.target.value) || 1)}>
                <option value={1}>1 (full)</option>
                <option value={2}>2</option>
                <option value={3}>3</option>
                <option value={4}>4</option>
              </select>
            </label>
          </div>
        </div>

        <div className="biomech-video-split">
          <div className="video-stage biomech-video-stage" ref={stageRef}>
            <video
              ref={videoRef}
              className="biomech-video"
              controls
              controlsList="nofullscreen noremoteplayback nodownload"
              disablePictureInPicture
              playsInline
              preload="metadata"
              src={`/api/sessions/${sessionId}/biomechanics/video`}
            />
            <canvas ref={canvasRef} className="biomech-overlay" />
          </div>
          <div
            ref={skeletonStageRef}
            className="video-stage biomech-skeleton-stage"
            style={{
              aspectRatio:
                pose?.video_metadata?.width && pose?.video_metadata?.height
                  ? `${pose.video_metadata.width} / ${pose.video_metadata.height}`
                  : undefined,
            }}
          >
            <div className="biomech-skeleton-label">Skeleton</div>
            <canvas ref={skeletonCanvasRef} className="biomech-skeleton-canvas" />
          </div>
        </div>

        {poseLoading ? (
          <p className="muted small">Loading pose frames…</p>
        ) : poseError ? (
          <p className="muted small">Pose overlay unavailable: {poseError}</p>
        ) : (
          <p className="muted small">
            Frames: {frames.length ? frames.length : '—'} · fps: {fps ? formatNumber(fps, 1) : '—'}
          </p>
        )}
      </div>
    );
  }

  function FeedbackPanel({ feedback, onSeekToFrame }) {
    const cues = Array.isArray(feedback) ? feedback : [];
    const top = cues.slice(0, 5);
    return (
      <div className="biomech-card">
        <div className="biomech-card__header">
          <div>
            <h4>Feedback</h4>
            <p className="muted small">Ranked coaching cues + suggested drills.</p>
          </div>
        </div>
        {!top.length ? (
          <p className="muted small">No coaching cues available.</p>
        ) : (
          <ul className="biomech-feedback-list">
            {top.map((cue, idx) => (
              <li key={idx} className="biomech-feedback">
                <div className="biomech-feedback__head">
                  <div className="biomech-feedback__metric">
                    {cue.metric_name || cue.metric || 'metric'}
                    {cue.severity ? <span className={`severity-tag ${cue.severity}`}>{cue.severity}</span> : null}
                  </div>
                  {cue.z_score != null ? <span className="muted small">z={formatNumber(cue.z_score, 2)}</span> : null}
                </div>
                <div className="biomech-feedback__text">{cue.feedback_text || cue.actionable_cue || ''}</div>
                {cue.drill_suggestion ? (
                  <div className="muted small">
                    <strong>Drill:</strong> {cue.drill_suggestion}
                  </div>
                ) : null}
                {cue.why_it_matters ? <div className="muted small">{cue.why_it_matters}</div> : null}
                {Array.isArray(cue.frame_ranges) && cue.frame_ranges.length ? (
                  <div className="muted small">
                    <strong>Review:</strong>{' '}
                    {cue.frame_ranges.slice(0, 2).map((r, i) => (
                      <button
                        key={i}
                        type="button"
                        className="link-button no-print"
                        onClick={() => onSeekToFrame?.(r.start_frame)}
                      >
                        {r.label || r.phase || 'range'} ({r.start_frame}–{r.end_frame})
                      </button>
                    ))}
                  </div>
                ) : null}
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  function TrendView({ athleteName, currentSessionId }) {
    const plotRef = useRef(null);
    const [status, setStatus] = useState('idle');
    const [rows, setRows] = useState([]);

    useEffect(() => {
      let cancelled = false;
      async function loadTrend() {
        if (!athleteName) return;
        setStatus('loading');
        try {
          const { response, data } = await fetchJson('/api/sessions');
          if (!response.ok) throw new Error(data?.error || 'Unable to load sessions.');
          const sessions = Array.isArray(data?.sessions) ? data.sessions : [];
          const mine = sessions
            .filter((s) => String(s?.athlete || '') === String(athleteName))
            .filter((s) => String(s?.biomechanics_status || '').toLowerCase() === 'complete')
            .sort((a, b) => String(a?.date || '').localeCompare(String(b?.date || '')));
          const recent = mine.slice(-8);
          const enriched = [];
          for (const s of recent) {
            const sid = String(s.id);
            const { response: r2, data: d2 } = await fetchJson(`/api/sessions/${sid}/biomechanics`);
            if (!r2.ok) continue;
            const metrics = d2?.metrics || {};
            const throwMetrics = metrics?.throw_metrics || {};
            enriched.push({
              session_id: sid,
              date: s.date,
              overall_score: d2?.comparison?.overall_score ?? null,
              wrist_speed: throwMetrics?.release_metrics?.wrist_speed ?? null,
              hip_rotation: throwMetrics?.release_metrics?.hip_rotation_deg ?? null,
              power_chain_lag: throwMetrics?.power_chain?.hip_to_wrist_ms ?? throwMetrics?.power_chain_lag_ms ?? null,
              symmetry: throwMetrics?.symmetry?.knee_angle_score ?? null,
            });
          }
          if (!cancelled) {
            setRows(enriched);
            setStatus('ready');
          }
        } catch (err) {
          if (!cancelled) setStatus('error');
        }
      }
      loadTrend();
      return () => {
        cancelled = true;
      };
    }, [athleteName, currentSessionId]);

    useEffect(() => {
      if (typeof Plotly === 'undefined') return;
      const node = plotRef.current;
      if (!node) return;
      if (!rows.length) return;

      const x = rows.map((r) => r.date || r.session_id);
      const metrics = [
        { key: 'overall_score', label: 'Overall score', color: '#0f172a', invert: false },
        { key: 'wrist_speed', label: 'Wrist speed', color: '#2563eb', invert: false },
        { key: 'hip_rotation', label: 'Hip rotation', color: '#f59e0b', invert: false },
        { key: 'power_chain_lag', label: 'Power chain (lower lag better)', color: '#ef4444', invert: true },
        { key: 'symmetry', label: 'Symmetry', color: '#10b981', invert: false },
      ];

      const traces = [];
      for (const metric of metrics) {
        const raw = rows.map((r) => {
          const v = Number(r[metric.key]);
          return Number.isFinite(v) ? v : null;
        });
        const finite = raw.filter((v) => v != null);
        if (finite.length < 2) continue;
        const lo = Math.min(...finite);
        const hi = Math.max(...finite);
        const denom = hi - lo || 1;
        const y = raw.map((v) => {
          if (v == null) return null;
          const norm = ((v - lo) / denom) * 100.0;
          return metric.invert ? 100.0 - norm : norm;
        });
        traces.push({
          x,
          y,
          type: 'scatter',
          mode: 'lines+markers',
          name: metric.label,
          line: { color: metric.color, width: 2 },
        });
      }

      if (!traces.length) return;
      const layout = {
        margin: { l: 38, r: 16, t: 30, b: 40 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: { title: 'session', gridcolor: 'rgba(148,163,184,0.25)' },
        yaxis: { title: 'normalized (0–100)', gridcolor: 'rgba(148,163,184,0.25)' },
        title: { text: 'Trend: top metrics (normalized)', font: { size: 14 } },
        legend: { orientation: 'h' },
      };
      Plotly.react(node, traces, layout, { displayModeBar: false, responsive: true });
      return () => Plotly.purge(node);
    }, [rows]);

    return (
      <div className="biomech-card">
        <div className="biomech-card__header">
          <div>
            <h4>Trend</h4>
            <p className="muted small">Recent sessions for this athlete (if available).</p>
          </div>
        </div>
        {status === 'loading' ? (
          <>
            <SkeletonLine />
            <SkeletonLine style={{ width: '75%' }} />
          </>
        ) : rows.length < 2 ? (
          <p className="muted small">Not enough analyzed sessions to show trends.</p>
        ) : (
          <div ref={plotRef} className="plotly-chart" />
        )}
      </div>
    );
  }

  function BiomechanicsViewer({ sessionId, athleteName }) {
    const [status, setStatus] = useState('loading');
    const [percent, setPercent] = useState(0);
    const [error, setError] = useState(null);
    const [bundle, setBundle] = useState(null);
    const [reference, setReference] = useState(null);
    const [isPrinting, setIsPrinting] = useState(false);
    const seekFrameRef = useRef(null);
    const fpsRef = useRef(30);

    const onFps = useCallback((fps) => {
      if (Number.isFinite(Number(fps))) fpsRef.current = Number(fps);
    }, []);

    const seekToFrame = useCallback((frameIdx) => {
      if (typeof seekFrameRef.current === 'function') {
        seekFrameRef.current(frameIdx);
        return;
      }
      const fallbackFps = fpsRef.current || 30;
      const video = document.querySelector('#biomechanicsViewerRoot video');
      if (!video) return;
      const idx = Number(frameIdx);
      if (!Number.isFinite(idx) || fallbackFps <= 0) return;
      video.currentTime = idx / fallbackFps;
      video.pause();
    }, []);

    const loadBundle = useCallback(async () => {
      setError(null);
      const { response, data } = await fetchJson(`/api/sessions/${sessionId}/biomechanics`);
      if (response.status === 202) {
        setStatus('processing');
        setPercent(Number(data?.percent_complete) || 0);
        return { kind: 'processing', job: data };
      }
      if (!response.ok) {
        setStatus('error');
        setError(data?.error || 'Biomechanics not available.');
        return { kind: 'error', error: data };
      }
      setBundle(data);
      setStatus('ready');
      return { kind: 'ready', bundle: data };
    }, [sessionId]);

    useEffect(() => {
      let cancelled = false;
      let timer = null;
      async function boot() {
        const first = await loadBundle();
        if (cancelled) return;
        if (first.kind === 'processing') {
          timer = window.setInterval(async () => {
            const { response, data } = await fetchJson(`/api/sessions/${sessionId}/biomechanics/progress`);
            if (cancelled) return;
            if (response.ok) {
              const p = Number(data?.percent_complete) || 0;
              setPercent(p);
              if (String(data?.status || '').toLowerCase() === 'complete') {
                window.clearInterval(timer);
                timer = null;
                showToast('Biomechanics analysis complete.', 'success');
                await loadBundle();
              }
              if (String(data?.status || '').toLowerCase() === 'error') {
                window.clearInterval(timer);
                timer = null;
                setStatus('error');
                setError(data?.error_message || 'Biomechanics processing failed.');
                showToast('Biomechanics analysis failed.', 'error');
              }
            }
          }, 1500);
        }
      }
      boot();
      return () => {
        cancelled = true;
        if (timer) window.clearInterval(timer);
      };
    }, [loadBundle, sessionId]);

    useEffect(() => {
      let cancelled = false;
      async function loadReference() {
        try {
          const { response, data } = await fetchJson('/api/elite-db/reference/overall');
          if (!response.ok) return;
          if (!cancelled) setReference(data);
        } catch (_err) {
          // Ignore; reference is optional.
        }
      }
      loadReference();
      return () => {
        cancelled = true;
      };
    }, []);

    const rerun = useCallback(async () => {
      try {
        const { response, data } = await fetchJson(`/api/sessions/${sessionId}/biomechanics/rerun`, { method: 'POST' });
        if (!response.ok) throw new Error(data?.error || 'Unable to rerun analysis.');
        showToast('Re-running biomechanics analysis…');
        setStatus('processing');
        setPercent(0);
        setBundle(null);
      } catch (err) {
        showToast(err?.message || 'Unable to rerun analysis.', 'error');
      }
    }, [sessionId]);

    const exportPdf = useCallback(() => {
      setIsPrinting(true);
      showToast('Preparing print view…');
      setTimeout(() => {
        window.print();
        setIsPrinting(false);
      }, 250);
    }, []);

    if (status === 'loading') {
      return (
        <div className="biomech-grid">
          <div className="biomech-card">
            <h4>Loading biomechanics…</h4>
            <SkeletonLine />
            <SkeletonLine style={{ width: '85%' }} />
            <SkeletonLine style={{ width: '70%' }} />
          </div>
        </div>
      );
    }

    if (status === 'processing') {
      return (
        <div className="biomech-card">
          <div className="biomech-card__header">
            <div>
              <h4>Biomechanics processing</h4>
              <p className="muted small">This can take ~30–90 seconds depending on video length.</p>
            </div>
            <div className="biomech-actions no-print">
              <button type="button" className="btn btn-ghost" onClick={rerun}>
                Rerun
              </button>
            </div>
          </div>
          <div className="progress-bar" role="progressbar" aria-valuenow={percent} aria-valuemin="0" aria-valuemax="100">
            <div className="progress-bar__fill" style={{ width: `${Math.max(0, Math.min(100, percent))}%` }} />
          </div>
          <p className="muted small">{formatNumber(percent, 0)}%</p>
        </div>
      );
    }

    if (status === 'error') {
      return (
        <div className="biomech-card">
          <div className="biomech-card__header">
            <div>
              <h4>Biomechanics unavailable</h4>
              <p className="muted small">{error || 'No biomechanics found for this session.'}</p>
            </div>
            <div className="biomech-actions no-print">
              <button type="button" className="btn btn-primary" onClick={rerun}>
                Rerun analysis
              </button>
            </div>
          </div>
        </div>
      );
    }

    const metrics = bundle?.metrics || {};
    const comparison = bundle?.comparison || null;
    const feedback = bundle?.feedback || [];

    const printCls = isPrinting ? 'is-printing' : '';

    return (
      <div className={`biomechanics-viewer ${printCls}`}>
        <div className="biomech-toolbar no-print">
          <button type="button" className="btn btn-soft" onClick={exportPdf}>
            Export PDF
          </button>
          <button type="button" className="btn btn-ghost" onClick={rerun}>
            Rerun
          </button>
        </div>

        <div className="biomech-grid">
          <VideoWithOverlay
            sessionId={sessionId}
            onFps={onFps}
            seekFrameRef={seekFrameRef}
            analysisWindow={metrics?.analysis_window}
            className="biomech-video-card"
          />
          <IssuesCard comparison={comparison} />
          <AngleChart
            title="Elbow angle"
            angleName="right_elbow"
            metrics={metrics}
            reference={reference}
            referenceMetricName="throwing_elbow_flexion_deg_mean"
          />
          <AngleChart
            title="Shoulder angle"
            angleName="right_shoulder"
            metrics={metrics}
            reference={reference}
            referenceMetricName="throwing_shoulder_angle_deg_mean"
          />
          <AngleChart
            title="Hip–shoulder separation"
            angleName="thorax_rotation"
            metrics={metrics}
            reference={reference}
            referenceMetricName="shoulder_hip_separation_deg_mean"
          />
          <FeedbackPanel feedback={feedback} onSeekToFrame={seekToFrame} />
          <TrendView athleteName={athleteName} currentSessionId={sessionId} />
        </div>
      </div>
    );
  }

  function initBiomechTabs() {
    const tabButtons = Array.from(document.querySelectorAll('[data-biomech-tab]'));
    const tabViews = Array.from(document.querySelectorAll('[data-biomech-view]'));
    if (!tabButtons.length || !tabViews.length) return;

    const activate = (name) => {
      tabButtons.forEach((btn) => {
        const active = btn.dataset.biomechTab === name;
        btn.classList.toggle('is-active', active);
        btn.setAttribute('aria-selected', active ? 'true' : 'false');
      });
      tabViews.forEach((view) => {
        const active = view.dataset.biomechView === name;
        view.classList.toggle('is-active', active);
      });
    };

    tabButtons.forEach((btn) => {
      btn.addEventListener('click', () => activate(btn.dataset.biomechTab));
    });
  }

  function mountInto(root, opts = {}) {
    const element = typeof root === 'string' ? document.getElementById(root) : root;
    if (!element) return;

    const sessionId = opts.sessionId || element.dataset.sessionId;
    if (!sessionId) return;
    const athleteName =
      opts.athleteName != null ? String(opts.athleteName) : String(element.dataset.athleteName || '');

    if (ReactDOM.createRoot) {
      let reactRoot = element.__biomechReactRoot;
      if (!reactRoot) {
        reactRoot = ReactDOM.createRoot(element);
        element.__biomechReactRoot = reactRoot;
      }
      reactRoot.render(<BiomechanicsViewer sessionId={sessionId} athleteName={athleteName} />);
      return;
    }
    // React 17 fallback.
    ReactDOM.render(<BiomechanicsViewer sessionId={sessionId} athleteName={athleteName} />, element);
  }

  function unmountInto(root) {
    const element = typeof root === 'string' ? document.getElementById(root) : root;
    if (!element) return;
    if (ReactDOM.createRoot && element.__biomechReactRoot) {
      element.__biomechReactRoot.unmount();
      element.__biomechReactRoot = null;
      return;
    }
    if (ReactDOM.unmountComponentAtNode) {
      ReactDOM.unmountComponentAtNode(element);
      return;
    }
    element.innerHTML = '';
  }

  window.BiomechanicsViewer = window.BiomechanicsViewer || {};
  window.BiomechanicsViewer.mount = mountInto;
  window.BiomechanicsViewer.unmount = unmountInto;

  function boot() {
    initBiomechTabs();
    mountInto('biomechanicsViewerRoot');
  }

  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
