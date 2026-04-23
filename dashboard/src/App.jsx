import { useEffect, useMemo, useRef, useState } from 'react';
import { apiFetch } from './api';
import MeshPreview from './MeshPreview';
import LayoutPoseMap2D from './LayoutPoseMap2D';
import RelationGraphEditor from './RelationGraphEditor';
import SceneViewer from './SceneViewer';
import SegmentationEditor from './SegmentationEditor';
import SimulationPassTree from './SimulationPassTree';
import TrajectoryViewer3D from './TrajectoryViewer3D';
import stageBackendCatalog from './stageBackends.json';
import {
  artifactUrl,
  asList,
  buildSceneBundle,
  collectArtifactPaths,
  evaluateStageStatuses,
  extractLayout,
  extractPhysicsDebugReportPath,
  formatDuration,
  formatObjectDisplayName,
  formatTimestamp,
  getSceneId,
  getStageModels,
  isLiveJobState,
  objectInventoryRows,
  summarizeJobProgress,
  setArtifactVersion,
  safeText,
  sizePriorRows,
  stageDefinitions,
} from './runtimeViewModel';

function statusClass(level) {
  if (level === 'aligned' || level === 'completed') return 'status-aligned';
  if (level === 'partial' || level === 'cancelled') return 'status-partial';
  if (level === 'fallback' || level === 'failed') return 'status-fallback';
  if (level === 'running') return 'status-running';
  if (level === 'pending' || level === 'queued') return 'status-pending';
  return 'status-partial';
}

function toneClass(level) {
  if (level === 'aligned' || level === 'completed') return 'tone-aligned';
  if (level === 'partial' || level === 'cancelled') return 'tone-partial';
  if (level === 'fallback' || level === 'failed') return 'tone-fallback';
  if (level === 'running') return 'tone-running';
  if (level === 'pending' || level === 'queued') return 'tone-pending';
  return 'tone-partial';
}

const SVG_CHART_PALETTE = [
  'var(--plot-accent-1)',
  'var(--plot-accent-2)',
  'var(--plot-accent-3)',
  'var(--plot-accent-4)',
  'var(--plot-accent-5)',
  'var(--plot-accent-6)',
];
const SVG_GRAPH_SURFACE = 'var(--graph-surface)';
const SVG_GRAPH_BORDER = 'var(--graph-border)';
const PAT3D_FULL_TITLE = 'PAT3D: Physics-Augmented Text-to-3D Scene Generation';
const SVG_GRAPH_GRID = 'var(--graph-grid)';
const SVG_GRAPH_AXIS = 'var(--graph-axis)';
const SVG_GRAPH_TEXT = 'var(--graph-text)';
const SVG_GRAPH_MUTED = 'var(--graph-muted)';

function parseApiError(payload, fallbackMessage) {
  const errorPayload = payload?.error;
  if (errorPayload && typeof errorPayload === 'object') {
    return {
      userMessage: errorPayload.userMessage || fallbackMessage,
      detail: errorPayload.detail || '',
      retryable: Boolean(errorPayload.retryable),
      code: errorPayload.code || 'api_error',
    };
  }
  if (typeof payload?.error === 'string') {
    return {
      userMessage: payload.error,
      detail: payload.error,
      retryable: false,
      code: 'api_error',
    };
  }
  return {
    userMessage: fallbackMessage,
    detail: fallbackMessage,
    retryable: false,
    code: 'api_error',
  };
}

function normalizeUiError(error, fallbackMessage, fallbackCode) {
  if (error?.userMessage) {
    return error;
  }
  const message = error?.message || fallbackMessage;
  return {
    userMessage: message,
    detail: error?.detail || message,
      retryable: Boolean(error?.retryable),
      code: error?.code || fallbackCode,
    };
}

function ErrorCallout({ title, error, tone = 'error' }) {
  if (!error) return null;
  const toneClassName = tone === 'warning' || tone === 'info' ? 'warning-box' : '';
  return (
    <div className={`error-box ${toneClassName}`}>
      <strong>{title}</strong>
      <div>{safeText(error.userMessage, 'Unknown error')}</div>
      {error.detail && error.detail !== error.userMessage ? <div className="error-detail">{error.detail}</div> : null}
      <div className="mono">{error.code}{error.retryable ? ' · retryable' : ''}</div>
    </div>
  );
}

function LogPanel({ text, error }) {
  if (!text && !error) return null;
  return (
    <section className="section-panel">
      <h2 className="section-title">Failure details</h2>
      {error ? <ErrorCallout title="Log load failed" error={error} /> : null}
      {text ? <pre className="log-panel">{text}</pre> : null}
    </section>
  );
}

function ProgressBar({ value, tone = 'running', label = '' }) {
  const numeric = Number.isFinite(Number.parseFloat(value)) ? Math.max(0, Math.min(100, Math.round(Number.parseFloat(value)))) : 0;
  const toneName = String(tone || '').startsWith('tone-') ? String(tone) : toneClass(tone);
  return (
    <div className="stage-progress-track" aria-label={label || `Progress ${numeric}%`}>
      <div className={`stage-progress-fill ${toneName}`} style={{ width: `${numeric}%` }} />
      <div className="stage-progress-label">{numeric}%</div>
    </div>
  );
}

function ArtifactLinks({ items }) {
  const resolvedItems = asList(items).filter((item) => item && (typeof item.path === 'string' || typeof item.href === 'string'));
  if (!resolvedItems.length) {
    return <div className="empty">No artifact links recorded for this section.</div>;
  }

  return (
    <div className="artifact-links">
      {resolvedItems.map((item) => (
        <a
          key={`${item.label}:${item.path || item.href}`}
          className="artifact-link"
          href={item.href || artifactUrl(item.path)}
          target="_blank"
          rel="noreferrer"
        >
          {item.label}
        </a>
      ))}
    </div>
  );
}

function trimTrailingZeros(value) {
  return String(value).replace(/(\.\d*?[1-9])0+$/u, '$1').replace(/\.0+$/u, '');
}

function formatAxisTick(value, { scientificBelow = 1e-3 } = {}) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 'n/a';
  }
  const absolute = Math.abs(numeric);
  if (absolute > 0 && absolute < scientificBelow) {
    return numeric.toExponential(1);
  }
  if (Number.isInteger(numeric)) {
    return numeric.toFixed(0);
  }
  if (absolute >= 100) {
    return numeric.toFixed(0);
  }
  if (absolute >= 10) {
    return trimTrailingZeros(numeric.toFixed(1));
  }
  if (absolute >= 1) {
    return trimTrailingZeros(numeric.toFixed(2));
  }
  return trimTrailingZeros(numeric.toFixed(3));
}

function buildLinearTicks(min, max, count = 5) {
  const numericMin = Number(min);
  const numericMax = Number(max);
  if (!Number.isFinite(numericMin) || !Number.isFinite(numericMax)) {
    return [];
  }
  if (Math.abs(numericMax - numericMin) < 1e-9) {
    return [numericMin];
  }
  return Array.from({ length: Math.max(2, count) }, (_, index) => {
    const ratio = index / (Math.max(2, count) - 1);
    return numericMin + ((numericMax - numericMin) * ratio);
  });
}

function buildIntegerTicks(max, count = 5) {
  const numericMax = Math.max(0, Math.round(Number(max) || 0));
  if (numericMax === 0) {
    return [0];
  }
  if (numericMax <= count - 1) {
    return Array.from({ length: numericMax + 1 }, (_, index) => index);
  }
  const step = Math.max(1, Math.ceil(numericMax / (count - 1)));
  const ticks = [0];
  for (let value = step; value < numericMax; value += step) {
    ticks.push(value);
  }
  if (ticks[ticks.length - 1] !== numericMax) {
    ticks.push(numericMax);
  }
  return ticks;
}

function meshLabelFromPath(path) {
  const lower = String(path || '').toLowerCase();
  if (!lower) {
    return 'Mesh';
  }
  if (lower.endsWith('.obj')) {
    return 'OBJ';
  }
  if (lower.endsWith('.glb')) {
    return 'GLB';
  }
  if (lower.endsWith('.gltf')) {
    return 'glTF';
  }
  return 'Mesh';
}

function renderImageCaption(images, index, mode) {
  const allImages = asList(images);
  return `Rendered view · View ${index + 1} of ${allImages.length} · Mode: ${safeText(mode, 'n/a')}`;
}

function preferredVisualizationRenderImages(images) {
  const allImages = asList(images).filter((image) => image?.path);
  const rendered = allImages.filter((image) => {
    const role = String(image?.role || '').toLowerCase();
    const path = String(image?.path || '').toLowerCase();
    return !role.includes('simplified') && !path.includes('simplified');
  });
  if (rendered.length) {
    return rendered;
  }
  const simplified = allImages.filter((image) => {
    const role = String(image?.role || '').toLowerCase();
    const path = String(image?.path || '').toLowerCase();
    return role.includes('simplified') || path.includes('simplified');
  });
  return simplified.length ? simplified : allImages;
}

function MediaCard({ title, path, caption, cacheToken = '' }) {
  return renderMediaCard({ title, path, caption, cacheToken });
}

function appendCacheToken(url, cacheToken, paramName = 'cv') {
  if (!url || !cacheToken) return url;
  try {
    const resolved = new URL(url, typeof window !== 'undefined' ? window.location.origin : 'http://localhost');
    resolved.searchParams.set(paramName, String(cacheToken));
    return resolved.toString();
  } catch {
    return url;
  }
}

function renderMediaCard({ title, path, caption, cacheToken = '' }) {
  if (!path) {
    return (
      <article className="media-card">
        <h4 className="media-title">{title}</h4>
        <div className="empty">No image artifact recorded for this stage.</div>
      </article>
    );
  }

  return (
    <article className="media-card">
      <h4 className="media-title">{title}</h4>
      <div className="media-frame">
        <img src={appendCacheToken(artifactUrl(path), cacheToken)} alt={title} loading="lazy" />
      </div>
      {caption ? <div className="note-box">{caption}</div> : null}
    </article>
  );
}

function GeometryPathLink({ path }) {
  if (!path) {
    return <span className="muted-inline">n/a</span>;
  }
  const href = artifactUrl(path);
  if (!href) {
    return <span className="mono geometry-path-text">{path}</span>;
  }
  return (
    <a className="mono geometry-path-link" href={href} target="_blank" rel="noreferrer">
      {path}
    </a>
  );
}

function geometryTransformLabel(object) {
  if (object?.already_transformed) {
    return 'already transformed';
  }
  if (object?.transform) {
    return 'pose transform applied';
  }
  return 'not specified';
}

function SceneGeometrySources({ bundle, version = '' }) {
  const [sceneBundleData, setSceneBundleData] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const abortController = new AbortController();
    setLoadError(null);

    if (!bundle) {
      setSceneBundleData(null);
      setLoading(false);
      return () => abortController.abort();
    }

    if (bundle.kind === 'inline') {
      setSceneBundleData(bundle.data || null);
      setLoading(false);
      return () => abortController.abort();
    }

    if (bundle.kind !== 'artifact' || !bundle.path) {
      setSceneBundleData(null);
      setLoading(false);
      return () => abortController.abort();
    }

    const bundleUrl = artifactUrl(bundle.path);
    if (!bundleUrl) {
      setSceneBundleData(null);
      setLoading(false);
      return () => abortController.abort();
    }

    setLoading(true);
    fetch(bundleUrl, { signal: abortController.signal })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Scene bundle fetch failed with HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((payload) => {
        setSceneBundleData(payload);
      })
      .catch((error) => {
        if (!abortController.signal.aborted) {
          setSceneBundleData(null);
          setLoadError(normalizeUiError(error, 'Could not load the scene geometry bundle.', 'scene_bundle_load_failed'));
        }
      })
      .finally(() => {
        if (!abortController.signal.aborted) {
          setLoading(false);
        }
      });

    return () => abortController.abort();
  }, [bundle, version]);

  const objects = asList(sceneBundleData?.objects).filter((object) => object?.mesh_obj_path);
  const bundlePath = bundle?.kind === 'artifact' ? bundle.path : null;
  const sourceType = sceneBundleData?.geometry_source_type
    || sceneBundleData?.source_scene_type
    || (bundle?.kind === 'inline' ? bundle.data?.geometry_source_type || bundle.data?.source_scene_type : null);

  return (
    <div className="geometry-source-stack">
      {bundlePath ? (
        <div className="note-box">
          Scene bundle: <GeometryPathLink path={bundlePath} />
        </div>
      ) : null}
      {sourceType ? (
        <div className="note-box">
          Geometry source type: <span className="mono">{sourceType}</span>
        </div>
      ) : null}
      {loadError ? <ErrorCallout title="Geometry path load" error={loadError} tone="warning" /> : null}
      {loading ? <div className="empty">Loading geometry paths...</div> : null}
      {!loading && !objects.length ? (
        <div className="empty">No visualization geometry paths are available yet.</div>
      ) : null}
      {objects.length ? (
        <Table
          headers={['Object', 'Mesh path', 'Texture path', 'Transform']}
          rows={objects.map((object) => [
            <span className="mono" key={`${object.object_id}:id`}>{formatObjectDisplayName(object.object_id, 'object')}</span>,
            <GeometryPathLink key={`${object.object_id}:mesh`} path={object.mesh_obj_path} />,
            <GeometryPathLink key={`${object.object_id}:texture`} path={object.texture_image_path} />,
            <span className="mono" key={`${object.object_id}:transform`}>
              {geometryTransformLabel(object)}
            </span>,
          ])}
        />
      ) : null}
    </div>
  );
}

function downloadNameFromResponse(response, fallbackName) {
  const disposition = response.headers.get('content-disposition') || '';
  const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
  if (utf8Match?.[1]) {
    try {
      return decodeURIComponent(utf8Match[1]);
    } catch {
      return utf8Match[1];
    }
  }
  const quotedMatch = disposition.match(/filename=\"([^\"]+)\"/i);
  if (quotedMatch?.[1]) {
    return quotedMatch[1];
  }
  return fallbackName;
}

function TrajectoryPlot({ trajectory }) {
  return <TrajectoryViewer3D trajectory={trajectory} />;
}

function LossPlot({ metrics, physicsMode = null, diffSimRequested = false, diffSimAttempted = false }) {
  const series = asList(metrics?.series).filter((item) => asList(item?.points).length > 0);
  if (physicsMode === 'forward_only') {
    if (diffSimAttempted) {
      return <div className="empty">This run ended in forward-only mode, so no optimization loss curve is available.</div>;
    }
    return <div className="empty">Diff-sim initialization is disabled for this run, so no optimization loss curve is recorded.</div>;
  }
  if (!diffSimRequested && physicsMode !== 'optimize_then_forward') {
    return <div className="empty">Diff-sim initialization is disabled for this run, so no optimization loss curve is recorded.</div>;
  }
  if (!metrics?.available || !series.length) {
    return <div className="empty">Waiting for diff-sim loss snapshots.</div>;
  }

  const losses = series.flatMap((item) => asList(item.points).map((point) => Number(point.loss ?? 0)));
  const allPoints = series.flatMap((item) => asList(item.points));
  const maxEpoch = Math.max(1, ...allPoints.map((point) => Number(point.epoch ?? point.step ?? 0)));
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const lossRange = Math.max(1e-9, maxLoss - minLoss);
  const palette = SVG_CHART_PALETTE;
  const plotBounds = { left: 64, right: 446, top: 44, bottom: 214 };
  const plotWidth = plotBounds.right - plotBounds.left;
  const plotHeight = plotBounds.bottom - plotBounds.top;
  const epochTicks = buildIntegerTicks(maxEpoch);
  const lossTicks = buildLinearTicks(minLoss, maxLoss);

  const project = (point) => {
    const x = plotBounds.left + ((Number(point.epoch ?? point.step ?? 0) / Math.max(1, maxEpoch)) * plotWidth);
    const y = plotBounds.bottom - (((Number(point.loss ?? minLoss) - minLoss) / lossRange) * plotHeight);
    return [x, y];
  };

  return (
    <div className="pose-frame">
      <svg viewBox="0 0 480 320" role="img" aria-label="Diff-sim optimization loss plot">
        <rect x="10" y="10" width="460" height="300" rx="22" fill={SVG_GRAPH_SURFACE} stroke={SVG_GRAPH_BORDER} />
        {epochTicks.map((tick) => {
          const x = plotBounds.left + ((tick / Math.max(1, maxEpoch)) * plotWidth);
          return (
            <g key={`loss-epoch-grid-${tick}`}>
              <line
                x1={x}
                y1={plotBounds.top}
                x2={x}
                y2={plotBounds.bottom}
                stroke={SVG_GRAPH_GRID}
                strokeDasharray="4 6"
              />
              <text
                x={x}
                y={plotBounds.bottom + 18}
                textAnchor="middle"
                fontSize="11"
                fontFamily="IBM Plex Mono"
                fill={SVG_GRAPH_MUTED}
              >
                {tick}
              </text>
            </g>
          );
        })}
        {lossTicks.map((tick) => {
          const y = plotBounds.bottom - (((tick - minLoss) / lossRange) * plotHeight);
          return (
            <g key={`loss-y-grid-${tick}`}>
              <line
                x1={plotBounds.left}
                y1={y}
                x2={plotBounds.right}
                y2={y}
                stroke={SVG_GRAPH_GRID}
                strokeDasharray="4 6"
              />
              <text
                x={plotBounds.left - 10}
                y={y + 4}
                textAnchor="end"
                fontSize="11"
                fontFamily="IBM Plex Mono"
                fill={SVG_GRAPH_MUTED}
              >
                {formatAxisTick(tick, { scientificBelow: 1e-4 })}
              </text>
            </g>
          );
        })}
        <line
          x1={plotBounds.left}
          y1={plotBounds.bottom}
          x2={plotBounds.right}
          y2={plotBounds.bottom}
          stroke={SVG_GRAPH_AXIS}
          strokeDasharray="6 6"
        />
        <line
          x1={plotBounds.left}
          y1={plotBounds.top}
          x2={plotBounds.left}
          y2={plotBounds.bottom}
          stroke={SVG_GRAPH_AXIS}
          strokeDasharray="6 6"
        />
        <text x="24" y="34" fontSize="12" fontFamily="IBM Plex Mono" fill={SVG_GRAPH_TEXT}>
          loss
        </text>
        <text
          x="22"
          y={(plotBounds.top + plotBounds.bottom) / 2}
          textAnchor="middle"
          fontSize="12"
          fontFamily="IBM Plex Mono"
          fill={SVG_GRAPH_TEXT}
          transform={`rotate(-90 22 ${(plotBounds.top + plotBounds.bottom) / 2})`}
        >
          Loss
        </text>
        <text
          x={(plotBounds.left + plotBounds.right) / 2}
          y={plotBounds.bottom + 38}
          textAnchor="middle"
          fontSize="12"
          fontFamily="IBM Plex Mono"
          fill={SVG_GRAPH_MUTED}
        >
          epoch
        </text>
        {series.map((item, index) => {
          const color = palette[index % palette.length];
          const points = asList(item.points);
          const polyline = points.map((point) => project(point).join(',')).join(' ');
          return (
            <g key={item.pass_index ?? index}>
              <polyline
                fill="none"
                stroke={color}
                strokeWidth="3"
                strokeLinejoin="round"
                strokeLinecap="round"
                points={polyline}
              />
              {points.map((point, pointIndex) => {
                const [px, py] = project(point);
                return (
                  <circle
                    key={`${item.pass_index ?? index}-${pointIndex}`}
                    cx={px}
                    cy={py}
                    r={pointIndex === points.length - 1 ? 5.5 : 3.5}
                    fill={color}
                    opacity={pointIndex === 0 ? 0.55 : 0.92}
                  />
                );
              })}
            </g>
          );
        })}
        {series.map((item, index) => {
          const color = palette[index % palette.length];
          return (
            <g key={`legend-${item.pass_index ?? index}`}>
              <rect x={30} y={272 + index * 18} width={10} height={10} rx={2} fill={color} />
              <text x={46} y={281 + index * 18} fontSize="12" fontFamily="IBM Plex Mono" fill={SVG_GRAPH_TEXT}>
                {safeText(item.label, `Pass ${index + 1}`)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="note-box">
        Best loss: {metrics?.best_loss !== null && metrics?.best_loss !== undefined ? Number(metrics.best_loss).toFixed(6) : 'n/a'}
      </div>
    </div>
  );
}

function Table({ headers, rows }) {
  if (!rows.length) {
    return <div className="empty">No rows were recorded for this view.</div>;
  }

  return (
    <div className="table-scroll-frame scroll-surface scroll-surface-horizontal">
      <table className="data-table">
        <thead>
          <tr>
            {headers.map((header) => <th key={header}>{header}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ProgressMetric({ label, value, note, tone = 'pending' }) {
  return (
    <article className={`progress-metric ${toneClass(tone)}`}>
      <span className="meta-label">{label}</span>
      <div className="progress-metric-value">{value}</div>
      <div className="stat-note">{note}</div>
    </article>
  );
}

function StageCard({
  stage,
  providerName,
  status,
  progress,
  message,
  children,
  anchorId,
  actions = null,
  actionError = null,
}) {
  const numericProgress = Number.isFinite(Number(progress)) ? Math.max(0, Math.min(100, Number.parseFloat(progress))) : null;
  return (
    <article className={`stage-card ${toneClass(status)}`} id={anchorId}>
      <div className="stage-head">
        <div className="stage-number">{stage.number}</div>
        <div>
          <h2 className="stage-title">{stage.label}</h2>
          <p className="stage-subtitle">{stage.functionality}</p>
        </div>
        {actions ? <div className="stage-actions">{actions}</div> : null}
        <span className={`status-pill ${statusClass(status)}`}>{status}</span>
      </div>
      <div className="stage-body">
        {actionError ? <ErrorCallout title="Retry error" error={actionError} /> : null}
        {numericProgress !== null ? (
          <div className="stage-progress-block">
            <ProgressBar value={numericProgress} tone={status} label={`${stage.label} progress`} />
            {message ? <div className="note-box stage-log-note">{message}</div> : null}
          </div>
        ) : null}
        <div className="stage-grid">
          <div className="info-card">
            <span className="meta-label">Paper role</span>
            <p className="card-text">{stage.paperIntent}</p>
          </div>
          <div className="info-card">
            <span className="meta-label">Current implementation</span>
            <p className="card-text">
              Provider: <strong>{safeText(providerName)}</strong>
            </p>
          </div>
        </div>
        {children}
      </div>
    </article>
  );
}

function describeStageProgress(stage) {
  if (!stage) return 'No stage activity is available yet.';
  if (stage.status === 'completed') {
    return stage.completed_at ? `Finished ${formatTimestamp(stage.completed_at)}` : 'Finished successfully.';
  }
  if (Number.isFinite(Number(stage.progress))) {
    return `Progress ${Math.round(Number(stage.progress))}%`;
  }
  if (stage.status === 'running') {
    return stage.last_log || (stage.started_at ? `Started ${formatTimestamp(stage.started_at)}` : 'Currently executing.');
  }
  if (stage.status === 'failed') {
    return stage.completed_at ? `Failed ${formatTimestamp(stage.completed_at)}` : 'Stopped on an error.';
  }
  if (stage.status === 'cancelled') {
    return stage.completed_at ? `Cancelled ${formatTimestamp(stage.completed_at)}` : 'Stopped manually.';
  }
  if (stage.status === 'awaiting_input') {
    return 'Waiting for manual mask input from the dashboard.';
  }
  if (stage.status === 'queued') {
    return 'Queued behind upstream work.';
  }
  return 'Waiting for upstream stages.';
}

const RETRY_PREREQUISITES_BY_STAGE = {
  'scene-understanding': ['first_contract_slice.reference_image_result'],
  'object-relation': [
    'first_contract_slice.reference_image_result',
    'first_contract_slice.scene_understanding',
  ],
  'object-assets': [
    'first_contract_slice.reference_image_result',
    'first_contract_slice.scene_understanding',
    'first_contract_slice.object_relation',
  ],
  'layout-initialization': [
    'first_contract_slice.reference_image_result',
    'first_contract_slice.scene_understanding',
    'first_contract_slice.object_relation',
    'object_assets',
  ],
  'simulation-preparation': [
    'first_contract_slice.reference_image_result',
    'first_contract_slice.scene_understanding',
    'first_contract_slice.object_relation',
    'object_assets',
    'layout_initialization',
  ],
  'physics-optimization': [
    'first_contract_slice.reference_image_result',
    'first_contract_slice.scene_understanding',
    'first_contract_slice.object_relation',
    'object_assets',
    'layout_initialization',
    'simulation_preparation',
  ],
  visualization: [
    'first_contract_slice.reference_image_result',
    'first_contract_slice.scene_understanding',
    'first_contract_slice.object_relation',
    'object_assets',
    'layout_initialization',
    'simulation_preparation',
    'physics_optimization',
  ],
};

function runtimeHasPrerequisite(runtimePayload, keyPath) {
  if (!runtimePayload || !keyPath) return false;
  const parts = keyPath.split('.');
  let current = runtimePayload;
  for (const part of parts) {
    if (!current || typeof current !== 'object' || !(part in current)) {
      return false;
    }
    current = current[part];
  }
  return current != null;
}

export function ProgressPanel({ job, autoRefresh = false }) {
  const summary = summarizeJobProgress(job);
  const historicalJob = job && !summary.isLive ? summary : null;
  const jobError = job?.error && typeof job.error === 'object'
    ? job.error
    : job?.error
      ? { userMessage: String(job.error), detail: String(job.error), retryable: false, code: 'job_error' }
      : null;

  if (!job || historicalJob) {
    const idleTone = historicalJob ? historicalJob.stateMeta.tone : 'pending';
    const idleLabel = historicalJob ? historicalJob.stateMeta.label : 'Idle';
    return (
      <section className="section-panel progress-panel">
        <div className="section-heading-row">
          <div>
            <span className="section-kicker">Pipeline pulse</span>
            <h2 className="section-title">Progress centered on the current run</h2>
            <p className="section-copy">
              Live pipeline signals only appear while a dashboard job is actively running. Historical runs remain available for inspection below.
            </p>
          </div>
          <div className="section-heading-actions">
            <span className={`chip ${autoRefresh ? 'chip-live' : ''}`}>Auto refresh: {autoRefresh ? 'live' : 'paused'}</span>
            <span className={`status-pill ${statusClass(idleTone)}`}>{idleLabel}</span>
          </div>
        </div>
        <div className="progress-hero progress-hero-empty">
          <div className={`progress-ring-shell ${toneClass(idleTone)}`}>
            <div className="progress-ring-value">{historicalJob ? 'Idle' : '0%'}</div>
            <div className="progress-ring-label">{historicalJob ? 'Historical selection' : 'No active run'}</div>
          </div>
          <div className="progress-empty-copy">
            <div className="progress-heading">No active run</div>
            <div className="progress-subtitle">
              {historicalJob
                ? `Inspecting the selected ${historicalJob.stateMeta.label.toLowerCase()} run for scene ${safeText(job?.scene_id)}. Stage evidence and recovery controls below reflect that saved state.`
                : 'Launch controls are ready. Submit a scene prompt to begin the eight-stage PAT3D pipeline.'}
            </div>
            {historicalJob ? (
              <div className="progress-facts">
                <div className="progress-fact">
                  <span className="meta-label">Last outcome</span>
                  <strong>{historicalJob.stateMeta.label}</strong>
                </div>
                <div className="progress-fact">
                  <span className="meta-label">Last completed</span>
                  <strong>{historicalJob.lastCompletedStage ? `${historicalJob.lastCompletedStage.number} ${historicalJob.lastCompletedStage.label}` : 'None recorded'}</strong>
                </div>
                <div className="progress-fact">
                  <span className="meta-label">Last update</span>
                  <strong>{formatTimestamp(historicalJob.updatedAt)}</strong>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </section>
    );
  }

  const activeStageLabel = summary.activeStage ? `${summary.activeStage.number} ${summary.activeStage.label}` : 'No active stage';
  const metricCards = [
    {
      label: 'Completed',
      value: `${summary.counts.completed}/${summary.total}`,
      note: summary.total ? `${summary.percent}% of the pipeline is across the finish line.` : 'The stage list is empty.',
      tone: 'aligned',
    },
    {
      label: 'In flight',
      value: String(summary.inFlight),
      note: summary.activeStage
        ? `Current focus: ${activeStageLabel}.`
        : 'Nothing is actively running right now.',
      tone: summary.inFlight ? summary.activeStage?.execution?.tone || 'running' : 'pending',
    },
    {
      label: 'Waiting',
      value: String(summary.waiting),
      note: summary.waiting
        ? 'Stages still blocked on upstream outputs.'
        : 'No downstream stages are waiting.',
      tone: 'pending',
    },
    {
      label: 'Elapsed',
      value: summary.durationLabel,
      note: summary.startedAt ? `Started ${formatTimestamp(summary.startedAt)}` : 'The timer starts when the first stage begins.',
      tone: 'running',
    },
  ];

  return (
    <section className="section-panel progress-panel" aria-live="polite">
      <div className="section-heading-row">
        <div>
          <span className="section-kicker">Pipeline pulse</span>
          <h2 className="section-title">Progress centered on the current run</h2>
          <p className="section-copy">
            Each stage reports live execution state, timing, and failure context so progress is visible at a glance.
          </p>
        </div>
        <div className="section-heading-actions">
          <span className={`chip ${autoRefresh ? 'chip-live' : ''}`}>Auto refresh: {autoRefresh ? 'live' : 'paused'}</span>
          <span className={`status-pill ${statusClass(summary.stateMeta.tone)}`}>{summary.stateMeta.label}</span>
        </div>
      </div>

      <div className="progress-hero">
        <div className={`progress-ring-shell ${toneClass(summary.stateMeta.tone)}`}>
          <div
            className="progress-ring"
            style={{ '--progress': `${summary.percent}%` }}
          >
            <div className="progress-ring-value">{summary.percent}%</div>
            <div className="progress-ring-label">Pipeline completion</div>
          </div>
        </div>

        <div className="progress-hero-copy">
          <div className="progress-heading">
            {summary.stateMeta.description}
          </div>
          <div className="progress-subtitle">
            Scene {safeText(job.scene_id)} with {summary.total} tracked stages.
          </div>
          <div className="progress-facts">
            <div className="progress-fact">
              <span className="meta-label">Current focus</span>
              <strong>{summary.activeStage ? activeStageLabel : 'No active stage'}</strong>
            </div>
            <div className="progress-fact">
              <span className="meta-label">Last completed</span>
              <strong>{summary.lastCompletedStage ? `${summary.lastCompletedStage.number} ${summary.lastCompletedStage.label}` : 'None yet'}</strong>
            </div>
            <div className="progress-fact">
              <span className="meta-label">Last update</span>
              <strong>{formatTimestamp(summary.updatedAt)}</strong>
            </div>
          </div>
        </div>
      </div>

      <div className="pipeline-ruler">
        <div className="pipeline-ruler-track">
          <div className="pipeline-ruler-fill" style={{ width: `${summary.percent}%` }} />
        </div>
        <div className="pipeline-ruler-grid">
          {summary.stages.map((stage) => (
            <div className="pipeline-ruler-stage" key={stage.id}>
              <div className={`pipeline-ruler-node ${toneClass(stage.execution.tone)}`}>{stage.number}</div>
              <div className="pipeline-ruler-label">{stage.label}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="progress-metric-grid">
        {metricCards.map((metric) => (
          <ProgressMetric
            key={metric.label}
            label={metric.label}
            value={metric.value}
            note={metric.note}
            tone={metric.tone}
          />
        ))}
      </div>

      <div className="pipeline-stage-grid">
        {summary.stages.map((stage) => (
          <article key={stage.id} className={`pipeline-stage-card ${toneClass(stage.execution.tone)}`}>
            <div className="pipeline-stage-top">
              <span className="meta-label">{stage.id}</span>
              <span className={`stage-badge ${statusClass(stage.execution.tone)}`}>{stage.execution.label}</span>
            </div>
            <div className="pipeline-stage-title-row">
              <div className="pipeline-stage-number">{stage.number}</div>
              <div>
                <div className="job-stage-title">{stage.label}</div>
                <div className="pipeline-stage-copy">{stage.functionality}</div>
              </div>
            </div>
            <div className="pipeline-stage-meta">{describeStageProgress(stage)}</div>
            {stage.error ? <div className="job-stage-error">{stage.error}</div> : null}
          </article>
        ))}
      </div>

      <ErrorCallout title="Job status" error={jobError} />
    </section>
  );
}

const STORAGE_KEYS = {
  prompt: 'pat3d.dashboard.prompt',
  objects: 'pat3d.dashboard.objects',
  sceneId: 'pat3d.dashboard.sceneId',
  segmentationMode: 'pat3d.dashboard.segmentationMode',
  llmModel: 'pat3d.dashboard.llmModel',
  imageModel: 'pat3d.dashboard.imageModel',
  objectCropCompletionEnabled: 'pat3d.dashboard.objectCropCompletionEnabled',
  structuredLlmMaxAttempts: 'pat3d.dashboard.structuredLlmMaxAttempts',
  structuredLlmReasoningBudget: 'pat3d.dashboard.structuredLlmReasoningBudget',
  requestedObjectInferenceBudget: 'pat3d.dashboard.requestedObjectInferenceBudget',
  reasoningEffort: 'pat3d.dashboard.reasoningEffort',
  previewAngleCount: 'pat3d.dashboard.previewAngleCount',
  physicsSettings: 'pat3d.dashboard.physicsSettings',
  stageBackends: 'pat3d.dashboard.stageBackends',
};

const segmentationStageId = 'scene-understanding-segmentation';
const SAM3_SEGMENTER_BACKEND = 'sam3_segmenter';
const DEFAULT_SEGMENTER_BACKEND = 'current_segmenter';
const CHAT_MODEL_OPTIONS = [
  'gpt-5.4',
  'gpt-5.3',
  'gpt-4o',
  'gpt-4o-mini',
  'gpt-4.1',
  'gpt-4.1-mini',
  'o4-mini',
  'o3-mini',
];
const IMAGE_MODEL_OPTIONS = [
  'gpt-image-1',
  'gpt-image-1.5',
  'dall-e-3',
  'dall-e-2',
];
const DEFAULT_OBJECT_CROP_COMPLETION_MODEL = 'gpt-image-1.5';
const REASONING_EFFORT_OPTIONS = ['auto', 'low', 'medium', 'high'];
const STRUCTURED_LLM_REASONING_BUDGET_MIN = 256;
const STRUCTURED_LLM_REASONING_BUDGET_MAX = 65536;
const STRUCTURED_LLM_REASONING_BUDGET_PRESETS = [800, 1200, 1600, 2400, 3200, 6400, 12800, 25600];
const REQUESTED_OBJECT_INFERENCE_BUDGET_MIN = 256;
const REQUESTED_OBJECT_INFERENCE_BUDGET_MAX = 65536;
const REQUESTED_OBJECT_INFERENCE_BUDGET_PRESETS = [256, 512, 800, 1280, 1600, 2400, 3200, 6400];
const DEFAULT_PHYSICS_SETTINGS = {
  diffSimEnabled: false,
  endFrame: 300,
  groundYValue: -1.1,
  totalOptEpoch: 50,
  physLr: 0.001,
  contactDHat: 5e-4,
  contactEpsVelocity: 1e-5,
};

function readUrlParam(key, fallback = '') {
  if (typeof window === 'undefined') return fallback;
  const value = new URLSearchParams(window.location.search).get(key);
  return value && value.trim() ? value.trim() : fallback;
}

function readStoredText(key, fallback = '') {
  if (typeof window === 'undefined') return fallback;
  const value = window.localStorage.getItem(key);
  return value === null ? fallback : value;
}

function readStoredModel(key, options, fallback) {
  if (typeof window === 'undefined') return fallback;
  const value = window.localStorage.getItem(key);
  return options.includes(value) ? value : fallback;
}

function readStoredBoolean(key, fallback = false) {
  if (typeof window === 'undefined') return fallback;
  const value = window.localStorage.getItem(key);
  if (value === null) return fallback;
  return value === 'true';
}

function readDemoRunButtonEnabled(fallback = false) {
  const urlOverride = readUrlParam('real-run');
  if (urlOverride) {
    return ['1', 'true', 'yes', 'on'].includes(urlOverride.toLowerCase());
  }
  return readStoredBoolean(STORAGE_KEYS.demoRunButtonEnabled, fallback);
}

function readStoredReasoningEffort(fallback = 'high') {
  return readStoredModel(STORAGE_KEYS.reasoningEffort, REASONING_EFFORT_OPTIONS, fallback);
}

function formatStructuredLlmBudgetPreset(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return String(value);
  if (numeric < 1000) return String(numeric);
  const compact = numeric / 1000;
  const digits = Number.isInteger(compact) ? 0 : compact < 10 ? 2 : 1;
  return `${compact.toFixed(digits).replace(/\.?0+$/, '')}k`;
}

function readStoredStageBackends() {
  if (typeof window === 'undefined') return defaultStageBackends();
  const raw = window.localStorage.getItem(STORAGE_KEYS.stageBackends);
  if (!raw) return defaultStageBackends();
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return defaultStageBackends();
    }
    const defaults = defaultStageBackends();
    const sanitized = { ...defaults };
    for (const [stageId, entry] of Object.entries(stageBackendCatalog)) {
      const candidate = typeof parsed[stageId] === 'string' ? parsed[stageId].trim() : '';
      const allowedValues = new Set(
        Array.isArray(entry?.options)
          ? entry.options
              .map((option) => (typeof option?.value === 'string' ? option.value.trim() : ''))
              .filter(Boolean)
          : [],
      );
      if (candidate && allowedValues.has(candidate)) {
        sanitized[stageId] = candidate;
      }
    }
    return sanitized;
  } catch {
    return defaultStageBackends();
  }
}

function readStoredSegmentationMode() {
  if (typeof window === 'undefined') return 'automatic';
  const value = window.localStorage.getItem(STORAGE_KEYS.segmentationMode);
  return value === 'manual' ? 'manual' : 'automatic';
}

function sanitizePreviewAngleCount(value) {
  const numeric = Number.parseInt(String(value ?? ''), 10);
  if (!Number.isFinite(numeric)) {
    return 12;
  }
  return Math.min(24, Math.max(1, numeric));
}

function sanitizeStructuredLlmMaxAttempts(value) {
  const numeric = Number.parseInt(String(value ?? ''), 10);
  if (!Number.isFinite(numeric)) {
    return 3;
  }
  return Math.min(10, Math.max(1, numeric));
}

function sanitizeStructuredLlmReasoningBudget(value) {
  const numeric = Number.parseInt(String(value ?? ''), 10);
  if (!Number.isFinite(numeric)) {
    return 12800;
  }
  return Math.min(65536, Math.max(256, numeric));
}

function sanitizeRequestedObjectInferenceBudget(value) {
  const numeric = Number.parseInt(String(value ?? ''), 10);
  if (!Number.isFinite(numeric)) {
    return 1280;
  }
  return Math.min(65536, Math.max(256, numeric));
}

function sanitizePhysicsInteger(value, fallback, minimum, maximum) {
  const numeric = Number.parseInt(String(value ?? ''), 10);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.min(maximum, Math.max(minimum, numeric));
}

function sanitizePhysicsFloat(value, fallback, minimum, maximum) {
  const numeric = Number.parseFloat(String(value ?? ''));
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.min(maximum, Math.max(minimum, numeric));
}

function sanitizePhysicsSettings(value) {
  const raw = value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  return {
    diffSimEnabled: typeof raw.diffSimEnabled === 'boolean'
      ? raw.diffSimEnabled
      : DEFAULT_PHYSICS_SETTINGS.diffSimEnabled,
    endFrame: sanitizePhysicsInteger(raw.endFrame, DEFAULT_PHYSICS_SETTINGS.endFrame, 1, 5000),
    groundYValue: sanitizePhysicsFloat(
      raw.groundYValue ?? raw.ground_y_value,
      DEFAULT_PHYSICS_SETTINGS.groundYValue,
      -10.0,
      10.0,
    ),
    totalOptEpoch: sanitizePhysicsInteger(raw.totalOptEpoch, DEFAULT_PHYSICS_SETTINGS.totalOptEpoch, 1, 1000),
    physLr: sanitizePhysicsFloat(raw.physLr, DEFAULT_PHYSICS_SETTINGS.physLr, 1e-6, 1.0),
    contactDHat: sanitizePhysicsFloat(raw.contactDHat, DEFAULT_PHYSICS_SETTINGS.contactDHat, 1e-7, 1e-1),
    contactEpsVelocity: sanitizePhysicsFloat(
      raw.contactEpsVelocity,
      DEFAULT_PHYSICS_SETTINGS.contactEpsVelocity,
      1e-8,
      1e-1,
    ),
  };
}

function countCatalogInstances(objectCatalog) {
  return asList(objectCatalog?.objects).reduce((total, objectEntry) => {
    const parsed = Number.parseInt(String(objectEntry?.count ?? ''), 10);
    return total + (Number.isFinite(parsed) && parsed > 0 ? parsed : 1);
  }, 0);
}

function objectCatalogSignature(objectCatalog) {
  return JSON.stringify(
    asList(objectCatalog?.objects).map((objectEntry) => ({
      objectId: safeText(objectEntry?.object_id),
      count: Number.parseInt(String(objectEntry?.count ?? ''), 10) || 1,
      sourceInstanceIds: asList(objectEntry?.source_instance_ids).map((value) => safeText(value)),
    })),
  );
}

function pickRelationEditorCatalog(relationCatalog, understandingCatalog) {
  const relationObjects = asList(relationCatalog?.objects);
  const understandingObjects = asList(understandingCatalog?.objects);
  if (!relationObjects.length) return understandingCatalog || relationCatalog || null;
  if (!understandingObjects.length) return relationCatalog;

  const relationInstances = countCatalogInstances(relationCatalog);
  const understandingInstances = countCatalogInstances(understandingCatalog);
  if (understandingInstances !== relationInstances) {
    return understandingInstances > relationInstances ? understandingCatalog : relationCatalog;
  }

  const relationSourceIds = relationObjects.reduce(
    (total, objectEntry) => total + asList(objectEntry?.source_instance_ids).length,
    0,
  );
  const understandingSourceIds = understandingObjects.reduce(
    (total, objectEntry) => total + asList(objectEntry?.source_instance_ids).length,
    0,
  );
  return understandingSourceIds > relationSourceIds ? understandingCatalog : relationCatalog;
}

function readStoredPhysicsSettings() {
  if (typeof window === 'undefined') return DEFAULT_PHYSICS_SETTINGS;
  const raw = window.localStorage.getItem(STORAGE_KEYS.physicsSettings);
  if (!raw) return DEFAULT_PHYSICS_SETTINGS;
  try {
    return sanitizePhysicsSettings(JSON.parse(raw));
  } catch {
    return DEFAULT_PHYSICS_SETTINGS;
  }
}

function readStoredPreviewAngleCount() {
  if (typeof window === 'undefined') return 12;
  return sanitizePreviewAngleCount(window.localStorage.getItem(STORAGE_KEYS.previewAngleCount));
}

function readStoredStructuredLlmMaxAttempts() {
  if (typeof window === 'undefined') return 3;
  return sanitizeStructuredLlmMaxAttempts(window.localStorage.getItem(STORAGE_KEYS.structuredLlmMaxAttempts));
}

function readStoredStructuredLlmReasoningBudget() {
  if (typeof window === 'undefined') return 12800;
  return sanitizeStructuredLlmReasoningBudget(window.localStorage.getItem(STORAGE_KEYS.structuredLlmReasoningBudget));
}

function readStoredRequestedObjectInferenceBudget() {
  if (typeof window === 'undefined') return 1280;
  return sanitizeRequestedObjectInferenceBudget(window.localStorage.getItem(STORAGE_KEYS.requestedObjectInferenceBudget));
}

function parseRequestedObjects(value) {
  return String(value || '')
    .split(/[\n,]/g)
    .map((item) => item.trim())
    .filter(Boolean);
}

function mapStagesById(rows) {
  return asList(rows).reduce((accumulator, row) => {
    if (row?.id) {
      accumulator[row.id] = row;
    }
    return accumulator;
  }, {});
}

function parseProgressFromRow(row) {
  if (!row) {
    return null;
  }
  if (typeof row.progress === 'number' || typeof row.progress === 'string') {
    const parsed = Number.parseFloat(row.progress);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  if (row.progress_completed !== undefined && row.progress_total !== undefined) {
    const completed = Number.parseFloat(row.progress_completed);
    const total = Number.parseFloat(row.progress_total);
    if (Number.isFinite(completed) && Number.isFinite(total) && total > 0) {
      return (completed / total) * 100;
    }
  }
  return null;
}

function resolveStageProgress(activeStageRows, stageId, fallbackStatus = 'pending') {
  const row = activeStageRows?.[stageId];
  if (!row) {
    return { status: fallbackStatus, progress: null, message: null };
  }
  return {
    status: row.status || fallbackStatus,
    progress: parseProgressFromRow(row),
    message: row.last_log || null,
  };
}

function parseInteger(value) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatLivePassTitle(pass, index) {
  const strategy = String(pass?.strategy || 'pass')
    .split(/[_-]/g)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
  return `Pass ${index + 1} · ${strategy || 'Pass'}`;
}

function resolvePhysicsPassProgress(reportPayload) {
  const passList = asList(reportPayload?.passes);
  const reportedTotal = parseInteger(reportPayload?.progressive_pass_count);
  const total = Math.max(passList.length, reportedTotal || 0);
  if (!total) {
    return {
      progress: null,
      completed: 0,
      total: 0,
      activePassIndex: null,
      message: null,
    };
  }

  const reportedCompleted = parseInteger(reportPayload?.completed_pass_count);
  const completed = Math.max(
    0,
    Math.min(
      total,
      reportedCompleted ?? passList.filter((pass) => String(pass?.status || '') === 'completed').length,
    ),
  );
  const reportedActiveIndex = parseInteger(reportPayload?.active_pass_index);
  const runningIndex = passList.findIndex((pass) => String(pass?.status || '') === 'running');
  const activePassIndex = reportedActiveIndex ?? (runningIndex >= 0 ? runningIndex : null);
  const activePass = activePassIndex !== null ? passList[activePassIndex] || null : null;

  let message = `${completed}/${total} pass${total === 1 ? '' : 'es'} completed.`;
  if (activePass && activePassIndex !== null) {
    message = `Running ${formatLivePassTitle(activePass, activePassIndex)}. ${completed}/${total} pass${total === 1 ? '' : 'es'} completed.`;
  } else if (completed >= total) {
    message = `Completed all ${total} pass${total === 1 ? '' : 'es'}.`;
  }

  return {
    progress: total > 0 ? (completed / total) * 100 : null,
    completed,
    total,
    activePassIndex,
    message,
  };
}

function isLivePhysicsStage(activeJob) {
  if (!isLiveJobState(activeJob?.state)) {
    return false;
  }
  if (String(activeJob?.current_stage_id || '') === 'physics-optimization') {
    return true;
  }
  return asList(activeJob?.stages).some((stage) => (
    String(stage?.id || '') === 'physics-optimization'
    && String(stage?.status || '') === 'running'
  ));
}

function buildLivePhysicsDebugReportPath(activeJob, activeRuntimeMatchesJob) {
  if (!activeRuntimeMatchesJob || !isLivePhysicsStage(activeJob)) {
    return null;
  }
  const sceneId = safeText(activeJob?.scene_id, '');
  if (!sceneId) {
    return null;
  }
  const workspaceRoot = safeText(activeJob?.workspace_root, '').replace(/\\/g, '/');
  if (workspaceRoot) {
    return `${workspaceRoot}/physics/${sceneId}/diff_init_report.json`;
  }
  return `_phys_result/${sceneId}/diff_init_report.json`;
}

function defaultStageBackends() {
  return Object.fromEntries(
    Object.entries(stageBackendCatalog).map(([stageId, entry]) => [stageId, entry.default]),
  );
}

function StageBackendSelector({ selected, onSelect, segmentationMode, id }) {
  const visibleStageBackendEntries = Object.entries(stageBackendCatalog).filter(
    ([stageId]) => !['layout-initialization', 'visualization'].includes(stageId),
  );

  return (
    <section className="section-panel control-card" id={id}>
      <span className="section-kicker">Backend routing</span>
      <h2 className="section-title">Stage backends</h2>
      <p className="section-copy">
        Switch providers per stage to compare fallback paths, paper-aligned implementations, and local infrastructure.
      </p>
      <div className="backend-stack">
        {visibleStageBackendEntries.map(([stageId, entry]) => {
          const currentValue = selected[stageId] || entry.default;
          const activeOption = entry.options.find((option) => option.value === currentValue) || entry.options[0];
          return (
            <article className="backend-card" key={stageId}>
              <div className="backend-head">
                <div>
                  <div className="backend-title">{entry.label}</div>
                  <div className="mono">{entry.role}</div>
                </div>
              </div>
              <div className="backend-options">
                {entry.options.map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    className={`backend-option ${
                      currentValue === option.value ? 'backend-option-active' : ''
                    } ${
                      entry.role === 'segmenter' && option.value === SAM3_SEGMENTER_BACKEND && segmentationMode === 'manual'
                        ? 'backend-option-disabled'
                        : ''
                    }`}
                    disabled={entry.role === 'segmenter' && option.value === SAM3_SEGMENTER_BACKEND && segmentationMode === 'manual'}
                    onClick={() => {
                      if (entry.role === 'segmenter' && option.value === SAM3_SEGMENTER_BACKEND && segmentationMode !== 'automatic') {
                        return;
                      }
                      onSelect(stageId, option.value);
                    }}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
              <div className="note-box">{activeOption.description}</div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function formatMetricScore(score, denominator = '') {
  const numeric = Number(score);
  if (!Number.isFinite(numeric)) {
    return 'n/a';
  }
  const formattedScore = numeric.toFixed(2);
  return denominator ? `${formattedScore}/${denominator}` : formattedScore;
}

function formatPhysicsMetricValue(metric, fallback = 'n/a') {
  if (metric?.value === null || metric?.value === undefined) {
    return fallback;
  }
  const numeric = Number(metric?.value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return trimTrailingZeros(numeric.toFixed(4));
}

function metricStatusTone(metric) {
  if (!metric) return 'pending';
  if (metric.status === 'completed') return 'aligned';
  if (metric.status === 'skipped') return 'pending';
  return 'fallback';
}

const CLIP_SCORE_BASELINES = [27.53, 28.93, 29.68];
const CLIP_SCORE_BASELINE_AVERAGE = CLIP_SCORE_BASELINES.reduce((total, score) => total + score, 0) / CLIP_SCORE_BASELINES.length;

function MetricScoreCard({ label, metric, referenceNote, denominator }) {
  const errorMessage = metric?.error?.message || metric?.error?.detail || '';
  return (
    <article className={`stat-card ${toneClass(metricStatusTone(metric))}`}>
      <span className="meta-label">{label}</span>
      <div className="stat-value">{formatMetricScore(metric?.score ?? metric?.mean, denominator)}</div>
      <div className="stat-note">
        {metric?.status === 'completed'
          ? `Model: ${safeText(metric.model, 'n/a')}`
          : safeText(errorMessage || metric?.status, 'Not computed')}
      </div>
      {referenceNote ? <div className="stat-note metric-reference-note">{referenceNote}</div> : null}
    </article>
  );
}

function MetricsPanel({
  metricsPayload,
  metricsBusy,
  metricsError,
  runtime,
  selectedRuntime,
  onCompute,
}) {
  const metricsData = metricsPayload?.metrics || null;
  const renderData = metricsData?.render || null;
  const metrics = metricsData?.metrics || {};
  const representativeImage = renderData?.representative_image || null;
  const metricsCacheToken = metricsData?.generated_at || metricsPayload?.metrics?.generated_at || '';
  const imageCount = asList(renderData?.image_paths).length || renderData?.image_count || 0;
  const renderError = renderData?.status === 'failed' ? renderData?.error?.message : '';
  const canCompute = Boolean(runtime && selectedRuntime);

  return (
    <section className="section-panel metrics-panel-shell" id="scene-metrics">
      <div className="section-heading-row">
        <div>
          <span className="section-kicker">Postflight metrics</span>
          <h2 className="section-title">Metrics computation</h2>
          <p className="section-copy">
            Compute the final render-alignment, semantic, and physical-plausibility scores after visualization export finishes.
          </p>
        </div>
        <div className="control-row">
          <button type="button" onClick={() => onCompute({ force: false })} disabled={!canCompute || metricsBusy}>
            {metricsBusy ? 'Computing...' : 'Compute metrics'}
          </button>
          <button type="button" onClick={() => onCompute({ force: true })} disabled={!canCompute || metricsBusy}>
            Recompute
          </button>
        </div>
      </div>
      <ErrorCallout title="Metrics" error={metricsError} />
      {!runtime ? <div className="empty">Select a run before computing metrics.</div> : null}
      {runtime && !metricsPayload?.available && !metricsBusy ? (
        <div className="note-box">No saved metrics JSON is available for this run yet.</div>
      ) : null}
      {renderError ? (
        <div className="error-box warning-box">
          <strong>Rendering</strong>
          <div>{renderError}</div>
        </div>
      ) : null}
      {metricsData ? (
        <div className="stage-grid stage-grid-wide metrics-panel-grid">
          <div className="info-card">
            <h3 className="card-title">Rendering</h3>
            {representativeImage ? (
              <div className="media-grid media-grid-single">
                <MediaCard
                  title="Representative metric view"
                  path={representativeImage}
                  caption={`View ${Number(renderData?.representative_index ?? 0) + 1} of ${imageCount || 18} rendered metric views.`}
                  cacheToken={metricsCacheToken}
                />
              </div>
            ) : (
              <div className="empty">No representative render is available yet.</div>
            )}
            <div className="note-box">
              Render folder: <span className="mono geometry-path-text">{safeText(renderData?.output_dir, 'n/a')}</span>
            </div>
            <div className="note-box">
              View sampling: three depression angles, six horizontal angles.
            </div>
          </div>
          <div className="info-card">
            <h3 className="card-title">Scores</h3>
            <div className="stats-grid metrics-scores-grid">
              <MetricScoreCard label="VQAScore" metric={metrics.vqa_score} denominator="1" />
              <MetricScoreCard
                label="CLIP score"
                metric={metrics.clip_score}
                referenceNote={`Baseline average: ${CLIP_SCORE_BASELINE_AVERAGE.toFixed(2)}`}
              />
              <MetricScoreCard label="Physical plausibility" metric={metrics.physical_plausibility_score} denominator="100" />
            </div>
            <div className="note-box">
              Metrics JSON: <GeometryPathLink path={metricsPayload.metrics_path || metricsData.metrics_result_path} />
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

export default function App() {
  const [runtimes, setRuntimes] = useState([]);
  const [jobs, setJobs] = useState([]);
  const [selectedRuntime, setSelectedRuntime] = useState(() => readUrlParam('runtime'));
  const [activeJobId, setActiveJobId] = useState(() => readUrlParam('job'));
  const [autoSelectRunningJob, setAutoSelectRunningJob] = useState(() => !readUrlParam('job'));
  const [syncRuntimeToJob, setSyncRuntimeToJob] = useState(true);
  const [runtimePayload, setRuntimePayload] = useState(null);
  const [activeJob, setActiveJob] = useState(null);
  const [submitError, setSubmitError] = useState(null);
  const [runtimesError, setRuntimesError] = useState(null);
  const [jobsError, setJobsError] = useState(null);
  const [runtimeError, setRuntimeError] = useState(null);
  const [runtimePending, setRuntimePending] = useState(null);
  const [trajectoryPayload, setTrajectoryPayload] = useState(null);
  const [trajectoryError, setTrajectoryError] = useState(null);
  const [physicsDebugReportPayload, setPhysicsDebugReportPayload] = useState(null);
  const [physicsDebugReportError, setPhysicsDebugReportError] = useState(null);
  const [activePhysicsPassIndex, setActivePhysicsPassIndex] = useState(-1);
  const [physicsMetricsPayload, setPhysicsMetricsPayload] = useState(null);
  const [physicsMetricsError, setPhysicsMetricsError] = useState(null);
  const [jobLogText, setJobLogText] = useState('');
  const [jobLogError, setJobLogError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastFetchedAt, setLastFetchedAt] = useState(null);
  const [tickMs, setTickMs] = useState(Date.now());
  const [runPrompt, setRunPrompt] = useState(() => readStoredText(STORAGE_KEYS.prompt, ''));
  const [runObjects, setRunObjects] = useState(() => readStoredText(STORAGE_KEYS.objects, ''));
  const [runSceneId, setRunSceneId] = useState(() => readStoredText(STORAGE_KEYS.sceneId, ''));
  const [segmentationMode, setSegmentationMode] = useState(() => readStoredSegmentationMode());
  const [llmModel, setLlmModel] = useState(() => readStoredModel(STORAGE_KEYS.llmModel, CHAT_MODEL_OPTIONS, 'gpt-5.4'));
  const [imageModel, setImageModel] = useState(() => readStoredModel(STORAGE_KEYS.imageModel, IMAGE_MODEL_OPTIONS, 'gpt-image-1.5'));
  const [objectCropCompletionEnabled, setObjectCropCompletionEnabled] = useState(() => (
    readStoredBoolean(STORAGE_KEYS.objectCropCompletionEnabled, true)
  ));
  const [structuredLlmMaxAttempts, setStructuredLlmMaxAttempts] = useState(() => readStoredStructuredLlmMaxAttempts());
  const [structuredLlmReasoningBudget, setStructuredLlmReasoningBudget] = useState(() => readStoredStructuredLlmReasoningBudget());
  const [requestedObjectInferenceBudget, setRequestedObjectInferenceBudget] = useState(() => readStoredRequestedObjectInferenceBudget());
  const [reasoningEffort, setReasoningEffort] = useState(() => readStoredReasoningEffort());
  const [previewAngleCount, setPreviewAngleCount] = useState(() => readStoredPreviewAngleCount());
  const [physicsSettings, setPhysicsSettings] = useState(() => readStoredPhysicsSettings());
  const [stageBackends, setStageBackends] = useState(() => readStoredStageBackends());
  const [submitting, setSubmitting] = useState(false);
  const [manualMaskBusy, setManualMaskBusy] = useState(false);
  const [manualMaskError, setManualMaskError] = useState(null);
  const [manualRedoBusy, setManualRedoBusy] = useState(false);
  const [manualRedoError, setManualRedoError] = useState(null);
  const [manualSizeBusy, setManualSizeBusy] = useState(false);
  const [manualSizeError, setManualSizeError] = useState(null);
  const [manualSizeEntries, setManualSizeEntries] = useState([]);
  const [cancelRunBusy, setCancelRunBusy] = useState(false);
  const [cancelRunError, setCancelRunError] = useState(null);
  const [retryStageBusyId, setRetryStageBusyId] = useState('');
  const [retryStageError, setRetryStageError] = useState(null);
  const [retryStageErrorId, setRetryStageErrorId] = useState('');
  const [metricsPayload, setMetricsPayload] = useState(null);
  const [metricsError, setMetricsError] = useState(null);
  const [metricsBusy, setMetricsBusy] = useState(false);
  const [relationGraphSaveBusy, setRelationGraphSaveBusy] = useState(false);
  const [relationGraphSaveError, setRelationGraphSaveError] = useState(null);
  const [sceneExportMode, setSceneExportMode] = useState('');
  const [sceneExportSource, setSceneExportSource] = useState('');
  const [sceneExportError, setSceneExportError] = useState(null);
  const [cacheDialogOpen, setCacheDialogOpen] = useState(false);
  const [cacheSummary, setCacheSummary] = useState(null);
  const [cacheSummaryBusy, setCacheSummaryBusy] = useState(false);
  const [cacheSummaryError, setCacheSummaryError] = useState(null);
  const [cacheCleanupBusy, setCacheCleanupBusy] = useState(false);
  const [cacheCleanupError, setCacheCleanupError] = useState(null);
  const [cacheConfirmText, setCacheConfirmText] = useState('');
  const [activeAssetIndex, setActiveAssetIndex] = useState(0);
  const [activeAssetPreviewId, setActiveAssetPreviewId] = useState('');
  const [activeSceneViewerMode, setActiveSceneViewerMode] = useState('simplified');
  const assetPreviewAutoInitializedRef = useRef(false);
  const assetSliderTabRefs = useRef(new Map());
  const selectedRuntimeRef = useRef(selectedRuntime);
  const activeJobIdRef = useRef(activeJobId);
  const activeLiveJob = activeJob && isLiveJobState(activeJob.state) ? activeJob : null;
  const globalLiveJob = useMemo(
    () => jobs.find((job) => isLiveJobState(job?.state)) || null,
    [jobs],
  );
  const hasAnyLiveJobs = jobs.some((job) => isLiveJobState(job?.state));
  const runButtonShowsActive = hasAnyLiveJobs;
  const runtimePayloadMatchesSelection = runtimePayload?.name === selectedRuntime;
  const hasSelectedRuntimeData = runtimePayloadMatchesSelection && Boolean(runtimePayload?.data);
  const selectedRuntimeLinkedToLiveJob = Boolean(
    selectedRuntime
    && jobs.some((job) => isLiveJobState(job?.state) && job.runtime_output_name === selectedRuntime),
  );
  const activeRuntimeMatchesJob = !selectedRuntime || activeJob?.runtime_output_name === selectedRuntime;
  const selectedRuntimeDerivedDataKey = !selectedRuntime
    ? ''
    : selectedRuntimeLinkedToLiveJob
      ? `${selectedRuntime}:${runtimePayloadMatchesSelection ? runtimePayload?.updatedAt || '' : ''}`
      : `${selectedRuntime}:${hasSelectedRuntimeData ? 'loaded' : 'pending'}`;
  const physicsDebugReportPath = runtimePayloadMatchesSelection
    ? extractPhysicsDebugReportPath(runtimePayload?.data)
    : null;
  const livePhysicsDebugReportPath = buildLivePhysicsDebugReportPath(activeJob, activeRuntimeMatchesJob);
  const resolvedPhysicsDebugReportPath = physicsDebugReportPath || livePhysicsDebugReportPath;
  const shouldPollPhysicsDebugReport = Boolean(
    selectedRuntime
    && resolvedPhysicsDebugReportPath
    && activeRuntimeMatchesJob
    && isLivePhysicsStage(activeJob)
  );

  useEffect(() => {
    selectedRuntimeRef.current = selectedRuntime;
  }, [selectedRuntime]);

  useEffect(() => {
    activeJobIdRef.current = activeJobId;
  }, [activeJobId]);

  async function downloadSceneExport(mode, targetBundle, sourceLabel) {
    if (!targetBundle) {
      return;
    }
    setSceneExportMode(mode);
    setSceneExportSource(sourceLabel);
    setSceneExportError(null);
    try {
      const response = await apiFetch('/api/scene-export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(
          targetBundle.kind === 'artifact'
            ? { mode, bundlePath: targetBundle.path }
            : { mode, bundle: targetBundle.data },
        ),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw parseApiError(payload, `Scene export failed with HTTP ${response.status}`);
      }

      const blob = await response.blob();
      const downloadName = downloadNameFromResponse(
        response,
        `${getSceneId(runtimePayload)}-${mode}.zip`,
      );
      const objectUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = objectUrl;
      link.download = downloadName;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(objectUrl);
    } catch (error) {
      setSceneExportError(normalizeUiError(error, 'Could not export the stage-8 scene bundle.', 'scene_export_failed'));
    } finally {
      setSceneExportMode('');
    }
  }

  async function fetchLocalCacheSummary() {
    setCacheSummaryBusy(true);
    setCacheSummaryError(null);
    try {
      const response = await apiFetch('/api/local-cache/summary');
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw parseApiError(payload, `Cache summary failed with HTTP ${response.status}`);
      }
      setCacheSummary(payload.summary || null);
    } catch (error) {
      setCacheSummary(null);
      setCacheSummaryError(
        normalizeUiError(error, 'Could not inspect the local dashboard cache.', 'local_cache_summary_failed'),
      );
    } finally {
      setCacheSummaryBusy(false);
    }
  }

  function openCacheCleanupDialog() {
    setCacheDialogOpen(true);
    setCacheConfirmText('');
    setCacheCleanupError(null);
    void fetchLocalCacheSummary();
  }

  function closeCacheCleanupDialog() {
    if (cacheCleanupBusy) return;
    setCacheDialogOpen(false);
    setCacheConfirmText('');
    setCacheCleanupError(null);
  }

  async function confirmLocalCacheCleanup() {
    setCacheCleanupBusy(true);
    setCacheCleanupError(null);
    try {
      const response = await apiFetch('/api/local-cache/cleanup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw parseApiError(payload, `Cache cleanup failed with HTTP ${response.status}`);
      }
      setCacheSummary(payload.result?.after || null);
      setRuntimes([]);
      setJobs([]);
      setSelectedRuntime('');
      setActiveJobId('');
      setAutoSelectRunningJob(true);
      setSyncRuntimeToJob(true);
      setRuntimePayload(null);
      setActiveJob(null);
      setRuntimeError(null);
      setRuntimePending(null);
      setTrajectoryPayload(null);
      setTrajectoryError(null);
      setMetricsPayload(null);
      setMetricsError(null);
      setPhysicsMetricsPayload(null);
      setPhysicsMetricsError(null);
      setJobLogText('');
      setJobLogError(null);
      setLastFetchedAt(null);
      await fetchRuntimeList().catch(() => {});
      await fetchJobs().catch(() => {});
      setCacheDialogOpen(false);
      setCacheConfirmText('');
    } catch (error) {
      setCacheCleanupError(
        normalizeUiError(error, 'Could not clear the local dashboard cache.', 'local_cache_cleanup_failed'),
      );
    } finally {
      setCacheCleanupBusy(false);
    }
  }

  async function fetchRuntimeList() {
    const response = await apiFetch('/api/runtimes');
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const nextError = parseApiError(payload, `Runtime list failed with HTTP ${response.status}`);
      setRuntimesError(nextError);
      throw new Error(nextError.userMessage);
    }
    const payload = await response.json();
    const nextRuntimes = payload.runtimes || [];
    setRuntimesError(null);
    setRuntimes(nextRuntimes);
  }

  async function fetchJobs() {
    const response = await apiFetch('/api/jobs');
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const nextError = parseApiError(payload, `Job list failed with HTTP ${response.status}`);
      setJobsError(nextError);
      throw new Error(nextError.userMessage);
    }
    const payload = await response.json();
    const nextJobs = payload.jobs || [];
    setJobsError(null);
    setJobs(nextJobs);
    if (!nextJobs.length) {
      setActiveJobId('');
      return;
    }
    const selectedJobStillExists = activeJobId && nextJobs.some((job) => job.job_id === activeJobId);
    if (selectedJobStillExists) {
      return;
    }
    const matchedSelectedRuntimeJob = selectedRuntimeRef.current
      ? nextJobs.find((job) => job.runtime_output_name === selectedRuntimeRef.current)
      : null;
    if (matchedSelectedRuntimeJob) {
      setActiveJobId(matchedSelectedRuntimeJob.job_id);
      return;
    }
    if (selectedRuntimeRef.current) {
      setActiveJobId('');
      return;
    }
    const runningJob = nextJobs.find((job) => isLiveJobState(job.state));
    if (runningJob) {
      setActiveJobId(runningJob.job_id);
      return;
    }
    setActiveJobId(nextJobs[0].job_id);
  }

  async function fetchRuntime(name) {
    if (!name) return;
    setLoading(true);
    try {
      const response = await apiFetch(`/api/runtime?name=${encodeURIComponent(name)}`);
      const payload = await response.json().catch(() => ({}));
      if (selectedRuntimeRef.current !== name) {
        return;
      }
      if (response.status === 202) {
        const runtimeMatchesLiveJob = Boolean(
          activeJob
          && isLiveJobState(activeJob.state)
          && (!activeJob.runtime_output_name || activeJob.runtime_output_name === name),
        );
        if (!runtimeMatchesLiveJob) {
          setRuntimePending(null);
          setRuntimeError(null);
          if (runtimePayload?.name !== name) {
            setRuntimePayload(null);
          }
          return;
        }
        const nextRuntimePending = parseApiError(payload, 'Runtime output is still pending.');
        const jobState = payload?.job_state;
        const activeStage = payload?.active_stage;
        const activeProgress = payload?.active_stage_progress;
        if (jobState) {
          nextRuntimePending.detail = `Job ${safeText(payload?.job_id, 'current job')} is ${jobState}.`
            + (activeStage ? ` Current stage: ${activeStage}${activeProgress !== null ? ` (${Math.round(activeProgress)}%)` : ''}.` : '');
        }
        setRuntimePending(nextRuntimePending);
        setRuntimeError(null);
        if (runtimePayload?.name !== name) {
          setRuntimePayload(null);
        }
        return;
      }
      if (!response.ok) {
        const nextError = parseApiError(payload, `Runtime fetch failed with HTTP ${response.status}`);
        setRuntimeError(nextError);
        setRuntimePending(null);
        throw new Error(nextError.userMessage);
      }
      setRuntimePayload(payload);
      setRuntimePending(null);
      setRuntimeError(null);
      setLastFetchedAt(new Date().toISOString());
    } catch (fetchError) {
      if (!runtimeError) {
        setRuntimeError(normalizeUiError(fetchError, 'Could not load the selected run outputs.', 'runtime_fetch_failed'));
      }
    } finally {
      setLoading(false);
    }
  }

  async function fetchPhysicsMetrics() {
    const shouldUseJobEndpoint = Boolean(
      activeJobId
      && (!selectedRuntime || activeJob?.runtime_output_name === selectedRuntime)
      && isLiveJobState(activeJob?.state)
    );
    const endpoint = shouldUseJobEndpoint
      ? `/api/jobs/${encodeURIComponent(activeJobId)}/physics-metrics`
      : selectedRuntime
        ? `/api/physics-metrics?runtime=${encodeURIComponent(selectedRuntime)}`
        : '';
    if (!endpoint) {
      setPhysicsMetricsPayload(null);
      setPhysicsMetricsError(null);
      return;
    }
    const response = await apiFetch(endpoint);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const nextError = parseApiError(payload, `Physics metrics fetch failed with HTTP ${response.status}`);
      setPhysicsMetricsError(nextError);
      throw new Error(nextError.userMessage);
    }
    setPhysicsMetricsPayload(payload);
    setPhysicsMetricsError(null);
  }

  async function fetchJob(jobId) {
    if (!jobId) return;
    const response = await apiFetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const nextError = parseApiError(payload, `Job fetch failed with HTTP ${response.status}`);
      setJobsError(nextError);
      throw new Error(nextError.userMessage);
    }
    const payload = await response.json();
    if (activeJobIdRef.current !== jobId) {
      return;
    }
    setJobsError(null);
    setActiveJob(payload);
    if (syncRuntimeToJob && payload.runtime_output_name) {
      setSelectedRuntime(payload.runtime_output_name);
    }
  }

  async function refreshWorkspace() {
    await fetchRuntimeList().catch(() => {});
    await fetchJobs().catch(() => {});
    if (activeJobIdRef.current) {
      await fetchJob(activeJobIdRef.current).catch(() => {});
    }
    if (selectedRuntimeRef.current) {
      await fetchRuntime(selectedRuntimeRef.current).catch(() => {});
    }
    await fetchPhysicsMetrics().catch(() => {});
  }

  async function submitMetrics({ force = false } = {}) {
    if (!selectedRuntime) return;
    setMetricsBusy(true);
    setMetricsError(null);
    try {
      const response = await apiFetch('/api/metrics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ runtime: selectedRuntime, force }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw parseApiError(payload, `Metrics request failed with HTTP ${response.status}`);
      }
      setMetricsPayload(payload);
    } catch (error) {
      setMetricsError(normalizeUiError(error, 'Could not compute metrics for this run.', 'metrics_compute_failed'));
    } finally {
      setMetricsBusy(false);
    }
  }

  async function submitJob(event) {
    event.preventDefault();
    setSubmitting(true);
    setSubmitError(null);
    setRuntimePending(null);
    setRuntimeError(null);
    try {
      const requestedObjectHints = parseRequestedObjects(runObjects);
      const submittedStageBackends = { ...stageBackends };
      const stageBackendsProfile = readUrlParam('stageProfile');
      if (segmentationMode === 'manual' && submittedStageBackends[segmentationStageId] === SAM3_SEGMENTER_BACKEND) {
        submittedStageBackends[segmentationStageId] = segmenterDefault;
        setStageBackends((current) => ({
          ...current,
          [segmentationStageId]: segmenterDefault,
        }));
      }
      const response = await apiFetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: runPrompt,
          requestedObjects: runObjects,
          sceneId: runSceneId,
          segmentationMode,
          llmModel,
          imageModel,
          objectCropCompletionEnabled,
          objectCropCompletionModel: DEFAULT_OBJECT_CROP_COMPLETION_MODEL,
          structuredLlmMaxAttempts,
          structuredLlmReasoningBudget,
          requestedObjectInferenceBudget,
          reasoningEffort,
          previewAngleCount,
          physicsSettings,
          stageBackends: submittedStageBackends,
          ...(stageBackendsProfile ? { stageBackendsProfile } : {}),
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        const nextError = parseApiError(payload, `Run request failed with HTTP ${response.status}`);
        throw new Error(nextError.userMessage);
      }
      setAutoSelectRunningJob(true);
      setSyncRuntimeToJob(true);
      setActiveJobId(payload.job_id);
      setActiveJob(payload);
      setRuntimePayload(null);
      setLoading(true);
      if (payload.runtime_output_name) {
        setSelectedRuntime(payload.runtime_output_name);
      }
      await fetchJobs();
    } catch (submitError) {
      setSubmitError(normalizeUiError(submitError, 'Could not create the dashboard job.', 'job_submit_failed'));
    } finally {
      setSubmitting(false);
    }
  }

  async function submitManualMasks({ action, instances }) {
    if (!activeJobId) return;
    setManualMaskBusy(true);
    setManualMaskError(null);
    try {
      const endpoint = action === 'continue'
        ? `/api/jobs/${encodeURIComponent(activeJobId)}/continue`
        : `/api/jobs/${encodeURIComponent(activeJobId)}/manual-masks`;
      const response = await apiFetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instances }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw parseApiError(payload, `Manual mask request failed with HTTP ${response.status}`);
      }
      setActiveJob(payload);
      setJobsError(null);
      setRuntimePending(null);
      setRuntimeError(null);
      setAutoSelectRunningJob(true);
      setSyncRuntimeToJob(true);
      if (payload.runtime_output_name) {
        setSelectedRuntime(payload.runtime_output_name);
      }
      if (action === 'continue') {
        setLoading(true);
      }
      await fetchJobs();
    } catch (error) {
      setManualMaskError(normalizeUiError(error, 'Could not submit the manual masks.', 'manual_mask_request_failed'));
    } finally {
      setManualMaskBusy(false);
    }
  }

  async function submitManualSizes() {
    if (!activeJobId) return;
    setManualSizeBusy(true);
    setManualSizeError(null);
    try {
      const response = await apiFetch(`/api/jobs/${encodeURIComponent(activeJobId)}/continue-size-priors`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entries: manualSizeEntries }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw parseApiError(payload, `Manual size request failed with HTTP ${response.status}`);
      }
      setActiveJob(payload);
      setJobsError(null);
      setRuntimePending(null);
      setRuntimeError(null);
      setAutoSelectRunningJob(true);
      setSyncRuntimeToJob(true);
      if (payload.runtime_output_name) {
        setSelectedRuntime(payload.runtime_output_name);
      }
      setLoading(true);
      await fetchJobs();
    } catch (error) {
      setManualSizeError(normalizeUiError(error, 'Could not continue the run with manual size priors.', 'manual_size_request_failed'));
    } finally {
      setManualSizeBusy(false);
    }
  }

  async function submitRedoSegmentationManually() {
    if (!activeJobId) return;
    setManualRedoBusy(true);
    setManualRedoError(null);
    setRuntimePending(null);
    setRuntimeError(null);
    try {
      const response = await apiFetch(`/api/jobs/${encodeURIComponent(activeJobId)}/redo-segmentation-manually`, {
        method: 'POST',
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw parseApiError(payload, `Manual segmentation reset failed with HTTP ${response.status}`);
      }
      setActiveJob(payload);
      setJobsError(null);
      setJobLogText('');
      setJobLogError(null);
      if (payload.runtime_output_name) {
        setSelectedRuntime(payload.runtime_output_name);
        await fetchRuntime(payload.runtime_output_name);
      }
      await fetchJobs();
    } catch (error) {
      setManualRedoError(normalizeUiError(error, 'Could not reopen the job for manual segmentation.', 'manual_segmentation_redo_failed'));
    } finally {
      setManualRedoBusy(false);
    }
  }

  async function submitCancelRun() {
    if (!activeJobId || !activeLiveJob) return;
    setCancelRunBusy(true);
    setCancelRunError(null);
    setRuntimePending(null);
    try {
      const response = await apiFetch(`/api/jobs/${encodeURIComponent(activeJobId)}/cancel`, {
        method: 'POST',
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw parseApiError(payload, `Cancel request failed with HTTP ${response.status}`);
      }
      setActiveJob(payload);
      setJobsError(null);
      await fetchJobs();
      if (payload.runtime_output_name) {
        setSelectedRuntime(payload.runtime_output_name);
        await fetchRuntime(payload.runtime_output_name).catch(() => {});
      }
    } catch (error) {
      setCancelRunError(normalizeUiError(error, 'Could not cancel the current run.', 'job_cancel_failed'));
    } finally {
      setCancelRunBusy(false);
    }
  }

  async function submitRetryFromStage(stageId) {
    if (!activeJobId) return;
    setRetryStageBusyId(stageId);
    setRetryStageError(null);
    setRetryStageErrorId('');
    setRuntimePending(null);
    setRuntimeError(null);
    try {
      const retryBody = { resumeFrom: stageId };
      if (physicsControlsEnabled) {
        retryBody.physicsSettings = {
          diffSimEnabled: physicsSettings.diffSimEnabled,
          endFrame: physicsSettings.endFrame,
          groundYValue: physicsSettings.groundYValue,
          totalOptEpoch: physicsSettings.totalOptEpoch,
          physLr: physicsSettings.physLr,
          contactDHat: physicsSettings.contactDHat,
          contactEpsVelocity: physicsSettings.contactEpsVelocity,
        };
      }
      const response = await apiFetch(`/api/jobs/${encodeURIComponent(activeJobId)}/resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(retryBody),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw parseApiError(payload, `Resume request failed with HTTP ${response.status}`);
      }
      setActiveJob(payload);
      setJobsError(null);
      setJobLogText('');
      setJobLogError(null);
      if (payload.runtime_output_name) {
        setSelectedRuntime(payload.runtime_output_name);
      }
      setLoading(true);
      await fetchJobs();
    } catch (error) {
      setRetryStageErrorId(stageId);
      setRetryStageError(normalizeUiError(error, `Could not retry stage '${stageId}'.`, 'job_resume_failed'));
    } finally {
      setRetryStageBusyId('');
    }
  }

  async function submitRelationGraph(graphPayload, { clear = false, retryStage = '' } = {}) {
    if (!activeJobId) return;
    setRelationGraphSaveBusy(true);
    setRelationGraphSaveError(null);
    try {
      const response = await apiFetch(`/api/jobs/${encodeURIComponent(activeJobId)}/relation-graph`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(clear ? { clear: true } : { relationGraph: graphPayload }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw parseApiError(payload, `Relation graph save failed with HTTP ${response.status}`);
      }
      setActiveJob(payload);
      setJobsError(null);
      await fetchJobs();
      if (retryStage) {
        await submitRetryFromStage(retryStage);
      }
    } catch (error) {
      setRelationGraphSaveError(normalizeUiError(error, 'Could not save the edited relation graph.', 'relation_graph_save_failed'));
    } finally {
      setRelationGraphSaveBusy(false);
    }
  }

  useEffect(() => {
    fetchRuntimeList().catch(() => {});
    fetchJobs().catch(() => {});
  }, []);

  useEffect(() => {
    if (!selectedRuntime) return;
    fetchRuntime(selectedRuntime);
  }, [selectedRuntime]);

  useEffect(() => {
    if (!selectedRuntime || !hasSelectedRuntimeData) {
      setTrajectoryPayload(null);
      setTrajectoryError(null);
      return;
    }
    let cancelled = false;
    apiFetch(`/api/trajectory?runtime=${encodeURIComponent(selectedRuntime)}`)
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw parseApiError(payload, `Trajectory fetch failed with HTTP ${response.status}`);
        }
        if (cancelled) return;
        setTrajectoryPayload(payload);
        setTrajectoryError(null);
      })
      .catch((error) => {
        if (cancelled) return;
        setTrajectoryPayload(null);
        setTrajectoryError(normalizeUiError(error, 'Could not load simulation trajectory data.', 'trajectory_fetch_failed'));
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRuntimeDerivedDataKey, hasSelectedRuntimeData]);

  useEffect(() => {
    if (!selectedRuntime || !resolvedPhysicsDebugReportPath || (!hasSelectedRuntimeData && !livePhysicsDebugReportPath)) {
      setPhysicsDebugReportPayload(null);
      setPhysicsDebugReportError(null);
      return;
    }
    let cancelled = false;
    let intervalId = null;

    const loadPhysicsDebugReport = () => apiFetch(artifactUrl(resolvedPhysicsDebugReportPath))
      .then(async (response) => {
        if (response.status === 404) {
          if (cancelled) return;
          setPhysicsDebugReportPayload(null);
          setPhysicsDebugReportError(null);
          return;
        }
        const payload = await response.json().catch(() => null);
        if (!response.ok || !payload) {
          throw new Error(`Physics debug report fetch failed with HTTP ${response.status}`);
        }
        if (cancelled) return;
        setPhysicsDebugReportPayload(payload);
        setPhysicsDebugReportError(null);
      })
      .catch((error) => {
        if (cancelled) return;
        setPhysicsDebugReportPayload(null);
        setPhysicsDebugReportError(
          normalizeUiError(error, 'Could not load the layered physics pass report.', 'physics_debug_report_failed'),
        );
      });

    loadPhysicsDebugReport();
    if (shouldPollPhysicsDebugReport) {
      intervalId = window.setInterval(loadPhysicsDebugReport, 1500);
    }
    return () => {
      cancelled = true;
      if (intervalId !== null) {
        window.clearInterval(intervalId);
      }
    };
  }, [
    selectedRuntime,
    selectedRuntimeDerivedDataKey,
    hasSelectedRuntimeData,
    livePhysicsDebugReportPath,
    resolvedPhysicsDebugReportPath,
    shouldPollPhysicsDebugReport,
  ]);

  useEffect(() => {
    if (!selectedRuntime || !hasSelectedRuntimeData) {
      setMetricsPayload(null);
      setMetricsError(null);
      return;
    }
    let cancelled = false;
    apiFetch(`/api/metrics?runtime=${encodeURIComponent(selectedRuntime)}`)
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw parseApiError(payload, `Metrics fetch failed with HTTP ${response.status}`);
        }
        if (cancelled) return;
        setMetricsPayload(payload);
        setMetricsError(null);
      })
      .catch((error) => {
        if (cancelled) return;
        setMetricsPayload(null);
        setMetricsError(normalizeUiError(error, 'Could not load metrics for this run.', 'metrics_fetch_failed'));
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRuntimeDerivedDataKey, hasSelectedRuntimeData]);

  useEffect(() => {
    if (!selectedRuntime && !activeJobId) {
      setPhysicsMetricsPayload(null);
      setPhysicsMetricsError(null);
      return;
    }
    let cancelled = false;
    fetchPhysicsMetrics()
      .catch((error) => {
        if (cancelled) return;
        setPhysicsMetricsPayload(null);
        setPhysicsMetricsError(
          normalizeUiError(error, 'Could not load physics loss history.', 'physics_metrics_fetch_failed'),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [selectedRuntime, activeJobId, activeJob?.runtime_output_name, activeJob?.state]);

  useEffect(() => {
    setArtifactVersion(runtimePayload?.updatedAt || '');
  }, [runtimePayload?.updatedAt]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.prompt, runPrompt);
  }, [runPrompt]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.objects, runObjects);
  }, [runObjects]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.sceneId, runSceneId);
  }, [runSceneId]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.segmentationMode, segmentationMode);
  }, [segmentationMode]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.llmModel, llmModel);
  }, [llmModel]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.imageModel, imageModel);
  }, [imageModel]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(
      STORAGE_KEYS.objectCropCompletionEnabled,
      objectCropCompletionEnabled ? 'true' : 'false',
    );
  }, [objectCropCompletionEnabled]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.structuredLlmMaxAttempts, String(structuredLlmMaxAttempts));
  }, [structuredLlmMaxAttempts]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.structuredLlmReasoningBudget, String(structuredLlmReasoningBudget));
  }, [structuredLlmReasoningBudget]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.requestedObjectInferenceBudget, String(requestedObjectInferenceBudget));
  }, [requestedObjectInferenceBudget]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.reasoningEffort, reasoningEffort);
  }, [reasoningEffort]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.previewAngleCount, String(previewAngleCount));
  }, [previewAngleCount]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.physicsSettings, JSON.stringify(physicsSettings));
  }, [physicsSettings]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.stageBackends, JSON.stringify(stageBackends));
  }, [stageBackends]);

  useEffect(() => {
    if (!activeJobId) return;
    fetchJob(activeJobId).catch(() => {});
  }, [activeJobId]);

  useEffect(() => {
    if (!runtimePending) return;
    const runtimeMatchesLiveJob = Boolean(
      activeJob
      && isLiveJobState(activeJob.state)
      && (!selectedRuntime || activeJob.runtime_output_name === selectedRuntime),
    );
    if (!runtimeMatchesLiveJob) {
      setRuntimePending(null);
    }
  }, [runtimePending, activeJob?.state, activeJob?.runtime_output_name, selectedRuntime]);

  useEffect(() => {
    if (activeJobId) return;
    setActiveJob(null);
    setJobLogText('');
    setJobLogError(null);
  }, [activeJobId]);

  useEffect(() => {
    if (!activeJob) return undefined;
    if (!isLiveJobState(activeJob.state)) return undefined;

    const interval = window.setInterval(() => {
      setTickMs(Date.now());
    }, 1000);
    return () => window.clearInterval(interval);
  }, [activeJob?.state, activeJob?.job_id]);

  useEffect(() => {
    if (!activeJobId || activeJob?.state !== 'failed') {
      setJobLogText('');
      setJobLogError(null);
      return;
    }
    apiFetch(`/api/jobs/${encodeURIComponent(activeJobId)}/log?tail=200`)
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw parseApiError(payload, `Job log fetch failed with HTTP ${response.status}`);
        }
        setJobLogText(payload.log || '');
        setJobLogError(null);
      })
      .catch((error) => {
        setJobLogError(normalizeUiError(error, 'Could not load the dashboard job log.', 'job_log_failed'));
      });
  }, [activeJobId, activeJob?.state]);

  useEffect(() => {
    if (activeJob?.state !== 'awaiting_mask_input') {
      setManualMaskError(null);
    }
  }, [activeJob?.job_id, activeJob?.state]);

  useEffect(() => {
    if (activeJob?.state !== 'awaiting_size_input') {
      setManualSizeError(null);
      return;
    }
    const nextEntries = Array.isArray(activeJob?.manual_size_priors?.entries)
      ? activeJob.manual_size_priors.entries.map((entry) => ({
        object_id: safeText(entry?.object_id, ''),
        canonical_name: safeText(entry?.canonical_name, ''),
        display_name: safeText(entry?.display_name, entry?.object_id || ''),
        dimensions_m: {
          x: entry?.dimensions_m?.x ?? '',
          y: entry?.dimensions_m?.y ?? '',
          z: entry?.dimensions_m?.z ?? '',
        },
      }))
      : [];
    setManualSizeEntries(nextEntries);
  }, [activeJob?.job_id, activeJob?.state, activeJob?.manual_size_priors?.updated_at]);

  useEffect(() => {
    if (activeJob?.state === 'awaiting_mask_input') {
      setManualRedoError(null);
    }
  }, [activeJob?.job_id, activeJob?.state]);

  useEffect(() => {
    if (!['failed', 'completed'].includes(activeJob?.state || '')) {
      setRetryStageError(null);
      setRetryStageErrorId('');
    }
  }, [activeJob?.job_id, activeJob?.state]);

  useEffect(() => {
    if (!autoRefresh || !hasAnyLiveJobs) return undefined;
    const interval = window.setInterval(() => {
      fetchRuntimeList().catch(() => {});
      fetchJobs().catch(() => {});
      if (selectedRuntime && selectedRuntimeLinkedToLiveJob) {
        fetchRuntime(selectedRuntime).catch(() => {});
      }
      if (activeJobId && (!activeJob || isLiveJobState(activeJob.state))) {
        fetchJob(activeJobId).catch(() => {});
      }
      if (activeLiveJob?.job_id) {
        fetchPhysicsMetrics().catch(() => {});
      }
    }, 5000);
    return () => window.clearInterval(interval);
  }, [
    autoRefresh,
    hasAnyLiveJobs,
    selectedRuntime,
    selectedRuntimeLinkedToLiveJob,
    activeJobId,
    activeJob?.state,
    activeLiveJob?.job_id,
  ]);

  const runtime = runtimePayload?.name === selectedRuntime ? runtimePayload?.data : null;
  const selectedJobListEntry = useMemo(
    () => jobs.find((job) => job.job_id === activeJobId) || null,
    [jobs, activeJobId],
  );
  const selectedJobLivePayload = activeJob?.job_id === activeJobId ? activeJob : null;
  const selectedJob = selectedJobLivePayload || selectedJobListEntry || null;
  const selectedJobSummary = useMemo(() => summarizeJobProgress(selectedJob), [selectedJob]);
  const jobSummary = useMemo(() => summarizeJobProgress(activeLiveJob), [activeLiveJob]);
  const historicalJobSummary = selectedJob && !isLiveJobState(selectedJob?.state) ? selectedJobSummary : null;
  const previousJobs = useMemo(
    () => jobs.filter((job) => job.job_id !== activeJobId),
    [jobs, activeJobId],
  );
  const stageModels = useMemo(() => (runtime ? getStageModels(runtime) : null), [runtime]);
  const liveStageRows = useMemo(() => mapStagesById(selectedJob?.stages), [selectedJob?.stages]);
  const statuses = useMemo(() => (runtime ? evaluateStageStatuses(runtime) : null), [runtime]);
  const sceneId = runtime ? getSceneId(runtime) : selectedJob?.scene_id || 'unknown';
  const artifactCount = runtime ? collectArtifactPaths(runtime).length : 0;
  const assetEntries = asList(stageModels?.assets?.data?.assets);
  const layout = extractLayout(runtime);
  const sceneBundleSimplified = useMemo(() => (runtime ? buildSceneBundle(runtime) : null), [runtime]);
  const sceneBundleOriginal = useMemo(
    () => (runtime ? buildSceneBundle(runtime, { meshSource: 'original' }) : null),
    [runtime],
  );
  const activeSceneBundle = activeSceneViewerMode === 'original' ? sceneBundleOriginal : sceneBundleSimplified;
  const activeSceneSourceLabel = activeSceneViewerMode === 'original' ? 'original' : 'simplified';
  const stage8PreviewImages = useMemo(
    () => preferredVisualizationRenderImages(stageModels?.visualization?.data?.render_images),
    [stageModels?.visualization?.data?.render_images],
  );
  const finalSceneObjectCount = asList(layout?.object_poses).length;
  const waitingForManualMasks = activeJob?.state === 'awaiting_mask_input' && activeJob?.segmentation_mode === 'manual';
  const waitingForManualSizes = activeJob?.state === 'awaiting_size_input';
  const manualSizeStageError = activeJob?.manual_size_priors?.error
    ? {
      userMessage: activeJob.manual_size_priors.error.userMessage || activeJob.manual_size_priors.error.user_message || 'Structured size inference failed.',
      detail: activeJob.manual_size_priors.error.detail || activeJob.manual_size_priors.error.technical_message || '',
      retryable: Boolean(activeJob.manual_size_priors.error.retryable),
      code: activeJob.manual_size_priors.error.code || 'size_inference_requires_input',
    }
    : null;
  const manualMaskCount = asList(activeJob?.manual_segmentation?.instances).length;
  const canRedoSegmentationManually = Boolean(
    activeJob?.job_id
    && ['completed', 'failed'].includes(activeJob?.state)
    && (
      runtime?.first_contract_slice?.reference_image_result?.image?.path
      || activeJob?.manual_segmentation?.reference_image_path
      || activeJob?.runtime_output_name
    )
  );
  const canRetryAnyStage = Boolean(activeJob?.job_id);
  const selectedSegmentationBackend = stageBackends['scene-understanding-segmentation']
    || stageBackendCatalog['scene-understanding-segmentation']?.default
    || 'current_segmenter';
  const selectedPhysicsBackend = stageBackends['physics-optimization']
    || stageBackendCatalog['physics-optimization']?.default
    || 'legacy_physics';
  const physicsControlsEnabled = selectedPhysicsBackend === 'legacy_physics';
  const physicsNotes = asList(stageModels?.physics?.notes).map((note) => String(note || ''));
  const runtimeUsesDiffSim = physicsNotes.some((note) => (
    note.includes('diff_sim_initialization_used') || note.includes('diff_sim_initialization_failed')
  ));
  const runtimeUsesForwardOnly = physicsNotes.some((note) => note.includes('forward_only_simulation_used'));
  const relationOverrideGraph = activeRuntimeMatchesJob ? activeJob?.relation_graph_override?.relation_graph : null;
  const relationData = useMemo(() => {
    const base = stageModels?.relation?.data;
    if (!base) return null;
    return relationOverrideGraph ? { ...base, relation_graph: relationOverrideGraph } : base;
  }, [stageModels?.relation?.data, relationOverrideGraph]);
  const relationEditorCatalog = useMemo(
    () => pickRelationEditorCatalog(
      relationData?.object_catalog,
      stageModels?.understanding?.data?.object_catalog,
    ),
    [relationData?.object_catalog, stageModels?.understanding?.data?.object_catalog],
  );
  const relationSourceKey = useMemo(
    () => JSON.stringify({
      runtime: selectedRuntime || '',
      overrideUpdatedAt: activeJob?.relation_graph_override?.updated_at || '',
      relationGraph: relationData?.relation_graph || null,
      objectCatalog: objectCatalogSignature(relationEditorCatalog),
    }),
    [selectedRuntime, activeJob?.relation_graph_override?.updated_at, relationData?.relation_graph, relationEditorCatalog],
  );
  const relationRetryStage = useMemo(() => {
    if (!activeRuntimeMatchesJob || !activeJob?.job_id) {
      return null;
    }
    if (liveStageRows?.['object-assets']?.status !== 'completed') {
      return { id: 'object-assets', label: 'object assets' };
    }
    return { id: 'layout-initialization', label: 'layout' };
  }, [activeRuntimeMatchesJob, activeJob?.job_id, activeJob?.state, liveStageRows]);
  const diffSimRequested = Boolean(activeRuntimeMatchesJob && activeJob?.physics_settings?.diff_sim_enabled);
  const diffSimAttempted = Boolean(runtimeUsesDiffSim || diffSimRequested);
  const physicsModeLabel = physicsMetricsPayload?.mode
    || (runtimeUsesForwardOnly ? 'forward_only' : runtimeUsesDiffSim ? 'optimize_then_forward' : diffSimRequested ? 'optimize_then_forward' : null);
  const penetrationMetric = physicsMetricsPayload?.final_metrics?.penetration_metric || null;
  const displacementMetric = physicsMetricsPayload?.final_metrics?.displacement_metric || null;
  const physicsPasses = asList(physicsDebugReportPayload?.passes);
  const physicsPassProgress = resolvePhysicsPassProgress(physicsDebugReportPayload);
  const physicsStageBaseProgress = resolveStageProgress(
    liveStageRows,
    'physics-optimization',
    stageModels?.physics?.status || 'pending',
  );
  const physicsStageProgress = physicsPassProgress.total > 0
    ? {
        ...physicsStageBaseProgress,
        progress: physicsPassProgress.progress,
        message: physicsPassProgress.message || physicsStageBaseProgress.message,
      }
    : physicsStageBaseProgress;
  const simulationPreparationProgress = resolveStageProgress(
    liveStageRows,
    'simulation-preparation',
    stageModels?.simulation?.status || 'pending',
  );
  const activePhysicsPass = physicsPasses[activePhysicsPassIndex] || null;
  const clampedActiveAssetIndex = assetEntries.length
    ? Math.max(0, Math.min(assetEntries.length - 1, activeAssetIndex))
    : 0;
  const activeAssetEntry = assetEntries[clampedActiveAssetIndex] || null;
  const selectAssetAtIndex = (nextIndex) => {
    if (!assetEntries.length) {
      setActiveAssetIndex(0);
      setActiveAssetPreviewId('');
      return;
    }
    const boundedIndex = Math.max(0, Math.min(assetEntries.length - 1, nextIndex));
    const nextAsset = assetEntries[boundedIndex] || null;
    const currentAssetId = activeAssetEntry?.object_id || '';
    setActiveAssetIndex(boundedIndex);
    if (nextAsset?.object_id !== currentAssetId) {
      if (nextAsset?.mesh_obj?.path) {
        setActiveAssetPreviewId(nextAsset.object_id);
      } else {
        setActiveAssetPreviewId('');
      }
    }
  };

  const setAssetSliderTabRef = (objectId, node) => {
    const refs = assetSliderTabRefs.current;
    if (!objectId) return;
    if (node) {
      refs.set(objectId, node);
      return;
    }
    refs.delete(objectId);
  };
  const stageNavigationLinks = stageDefinitions.map((stage) => ({
    href: `#stage-${stage.id}`,
    label: `${stage.number} ${stage.label}`,
  }));

  useEffect(() => {
    assetPreviewAutoInitializedRef.current = false;
    setActiveAssetPreviewId('');
  }, [selectedRuntime]);

  useEffect(() => {
    if (!assetEntries.length) {
      setActiveAssetIndex(0);
      return;
    }
    setActiveAssetIndex((current) => Math.max(0, Math.min(assetEntries.length - 1, current)));
  }, [assetEntries.length]);

  useEffect(() => {
    if (!activeAssetPreviewId) return;
    const previewExists = assetEntries.some((asset) => (
      asset.object_id === activeAssetPreviewId && asset?.mesh_obj?.path
    ));
    if (!previewExists) {
      setActiveAssetPreviewId('');
    }
  }, [activeAssetPreviewId, assetEntries]);

  useEffect(() => {
    if (assetPreviewAutoInitializedRef.current) return;
    if (activeAssetPreviewId) {
      assetPreviewAutoInitializedRef.current = true;
      return;
    }
    if (!activeAssetEntry?.mesh_obj?.path) return;
    setActiveAssetPreviewId(activeAssetEntry.object_id);
    assetPreviewAutoInitializedRef.current = true;
  }, [
    activeAssetEntry?.mesh_obj?.path,
    activeAssetEntry?.object_id,
    activeAssetPreviewId,
  ]);

  useEffect(() => {
    const activeObjectId = activeAssetEntry?.object_id;
    if (!activeObjectId) return;
    const activeTabNode = assetSliderTabRefs.current.get(activeObjectId);
    if (!activeTabNode || typeof activeTabNode.scrollIntoView !== 'function') return;
    activeTabNode.scrollIntoView({
      block: 'nearest',
      inline: 'center',
      behavior: 'smooth',
    });
  }, [activeAssetEntry?.object_id]);

  useEffect(() => {
    if (activeSceneViewerMode === 'original' && sceneBundleOriginal) return;
    if (activeSceneViewerMode === 'simplified' && sceneBundleSimplified) return;
    if (sceneBundleSimplified) {
      setActiveSceneViewerMode('simplified');
      return;
    }
    if (sceneBundleOriginal) {
      setActiveSceneViewerMode('original');
    }
  }, [activeSceneViewerMode, sceneBundleOriginal, sceneBundleSimplified]);

  useEffect(() => {
    if (!physicsPasses.length) {
      setActivePhysicsPassIndex(-1);
      return;
    }
    setActivePhysicsPassIndex((current) => {
      const liveActiveIndex = physicsPassProgress.activePassIndex;
      if (
        activeRuntimeMatchesJob
        && isLivePhysicsStage(activeJob)
        && liveActiveIndex !== null
      ) {
        return Math.max(0, Math.min(physicsPasses.length - 1, liveActiveIndex));
      }
      if (current < 0) {
        return physicsPasses.length - 1;
      }
      if (current >= physicsPasses.length) {
        return physicsPasses.length - 1;
      }
      return current;
    });
  }, [
    physicsPasses.length,
    physicsPassProgress.activePassIndex,
    activeRuntimeMatchesJob,
    activeJob,
  ]);

  const quickJumpSetupLinks = [
    { href: '#pipeline-configuration', label: 'Configuration', detail: 'Dataset, prompt, and launch setup' },
    { href: '#prompt-configuration', label: 'Prompt setup', detail: 'Reference image and prompt controls' },
    { href: '#segmentation-configuration', label: 'Segmentation', detail: 'Masking and extraction controls' },
    { href: '#model-configuration', label: 'Models', detail: 'Provider routing and model choices' },
    { href: '#physics-configuration', label: 'Physics', detail: 'Simulation and diff-sim tuning' },
  ];
  const quickJumpObservationLinks = [
    { href: '#stage-backends', label: 'Backend routing', detail: 'Per-stage provider wiring' },
    { href: '#runtime-telemetry', label: 'Run telemetry', detail: 'Live state, timings, and events' },
    { href: '#stage-evidence', label: 'Stage evidence', detail: 'Artifacts and final scene inspection' },
    { href: '#scene-metrics', label: 'Metrics', detail: 'Final scoring and physical checks' },
  ];
  const stageConfigLinks = stageNavigationLinks.map((stageLink) => ({
    ...stageLink,
    href: stageLink.href,
  }));
  const hasStageEvidence = Boolean(runtime && stageModels);
  const canJumpToStages = Boolean(activeJob?.job_id) || hasStageEvidence;
  const resolveStageTarget = (href) => (canJumpToStages ? href : '#stage-evidence');
  const quickJumpSections = [
    {
      title: 'Configure the run',
      copy: 'Set the inputs, providers, and simulation controls before a launch starts.',
      links: quickJumpSetupLinks,
    },
    {
      title: 'Monitor the system',
      copy: 'Track backend routing, runtime telemetry, and the metrics that matter while the pipeline moves.',
      links: quickJumpObservationLinks,
    },
    {
      title: 'Walk the paper stages',
      copy: 'Jump directly into each stage once a run is available for inspection.',
      links: stageConfigLinks.map((link) => ({
        href: resolveStageTarget(link.href),
        clickHref: link.href,
        label: link.label,
        detail: link.description || 'Paper stage output',
        disabled: !canJumpToStages,
      })),
    },
  ];
  const currentStageFallbackLabel = selectedJob?.current_stage_id
    ? stageDefinitions.find((stage) => stage.id === selectedJob.current_stage_id)?.label || selectedJob.current_stage_id
    : 'Waiting';
  const currentRunStageLabel = selectedJob
    ? safeText(selectedJobSummary?.activeStage?.label || currentStageFallbackLabel, 'Waiting')
    : 'Ready';
  const currentRunStateLabel = selectedJob
    ? selectedJobSummary?.stateMeta?.label || safeText(selectedJob?.state, 'Unknown')
    : 'Ready';
  const currentRunStatusTone = selectedJob
    ? selectedJobSummary?.stateMeta?.tone || 'pending'
    : 'pending';
  const requestedObjectLabels = activeJob?.requested_objects?.length
    ? activeJob.requested_objects
    : parseRequestedObjects(runObjects);
  const hasRequestedObjects = requestedObjectLabels.length > 0;
  const requestedObjectSourceLabel = activeJob?.requested_objects_inferred === true
    ? 'Inferred from prompt'
    : activeJob?.requested_objects_inferred === false && hasRequestedObjects
      ? 'Provided from prompt form'
      : 'Source not available';
  const requestedObjectSourceTone = activeJob?.requested_objects_inferred === true
    ? 'running'
    : activeJob?.requested_objects_inferred === false && hasRequestedObjects
      ? 'aligned'
      : 'pending';
  const understandingProviderName = waitingForManualMasks ? 'manual mask editor' : stageModels?.understanding?.providerName;
  const heroMetrics = [
    {
      label: 'Pipeline pulse',
      value: activeLiveJob ? `${jobSummary.percent}%` : 'Idle',
    },
    {
      label: 'Completed stages',
      value: activeLiveJob ? `${jobSummary.counts.completed}/${jobSummary.total || stageDefinitions.length}` : '--',
    },
  ];
  const insightCards = [
    {
      title: 'Execution posture',
      text: activeLiveJob
        ? `${jobSummary.stateMeta.description} ${jobSummary.activeStage ? `Current focus: ${jobSummary.activeStage.label}.` : ''}`
        : historicalJobSummary
        ? `No run is active. Inspecting the selected ${historicalJobSummary.stateMeta.label.toLowerCase()} run for scene ${safeText(selectedJob?.scene_id)}.`
        : 'Start a new run or select an existing job to surface live execution telemetry.',
      tone: activeLiveJob ? jobSummary.stateMeta.tone : 'pending',
    },
    {
      title: 'Scene readiness',
      text: runtime
        ? `${artifactCount} artifacts and ${finalSceneObjectCount} placed objects are available for direct inspection in the browser.`
        : 'Select a run to unlock artifact coverage, object placement, and the final scene viewer.',
      tone: runtime ? 'aligned' : 'pending',
    },
    {
      title: 'Attention point',
      text: waitingForManualMasks
        ? 'Scene understanding is paused at the manual masking checkpoint. Paint the masks in stage 2 and continue when the object coverage looks right.'
        : selectedJob?.state === 'failed'
        ? `${activeLiveJob ? 'This run' : 'The selected run'} stopped at ${safeText(selectedJobSummary.activeStage?.label, 'an unknown stage')}. Check the failure log before rerunning.`
        : 'No stage-level failure is active right now. The dashboard is ready for geometry review and backend iteration.',
      tone: waitingForManualMasks
        ? 'partial'
        : selectedJob?.state === 'failed'
          ? 'fallback'
          : 'running',
    },
  ];

  const statusCards = runtime && statuses ? [
    {
      label: 'Aligned stages',
      value: Object.values(statuses).filter((value) => value === 'aligned').length,
      note: 'Stages that are already close to the intended paper behavior in this run.',
      tone: 'aligned',
    },
    {
      label: 'Partial stages',
      value: Object.values(statuses).filter((value) => value === 'partial').length,
      note: 'Stages still bridged through legacy code or mixed execution seams.',
      tone: 'partial',
    },
    {
      label: 'Fallback stages',
      value: Object.values(statuses).filter((value) => value === 'fallback').length,
      note: 'Stages that still report a fallback path in their saved output metadata.',
      tone: 'fallback',
    },
    {
      label: 'Tracked artifacts',
      value: artifactCount,
      note: 'Distinct artifact paths referenced by the loaded run outputs.',
      tone: 'running',
    },
    {
      label: 'Stage-4 assets',
      value: assetEntries.length,
      note: 'Per-object textured assets produced during object asset generation.',
      tone: 'running',
    },
    {
      label: 'Final scene objects',
      value: finalSceneObjectCount,
      note: 'Objects currently placed in the final scene. Floor/grid helpers are not counted.',
      tone: 'aligned',
    },
  ] : [];

  const segmenterDefault = DEFAULT_SEGMENTER_BACKEND;
  const cacheCleanupBlocked = hasAnyLiveJobs || Boolean(cacheSummary?.has_active_jobs);
  const canConfirmCacheCleanup = cacheConfirmText.trim() === 'DELETE' && !cacheCleanupBlocked && !cacheSummaryBusy;
  const hasConflictingLiveJob = Boolean(globalLiveJob && globalLiveJob.job_id !== activeJobId);
  const retryAvailabilityByStage = useMemo(() => {
    const availability = {};
    for (const stage of stageDefinitions) {
      if (!activeJob?.job_id) {
        availability[stage.id] = { visible: false, enabled: false, reason: '' };
        continue;
      }
      if (hasConflictingLiveJob) {
        availability[stage.id] = {
          visible: true,
          enabled: false,
          reason: 'Finish or retry the active current run before restarting this one.',
        };
        continue;
      }
      if (stage.id === 'reference-image') {
        availability[stage.id] = { visible: true, enabled: true, reason: '' };
        continue;
      }
      const prerequisites = RETRY_PREREQUISITES_BY_STAGE[stage.id] || [];
      const missing = prerequisites.filter((pathKey) => !runtimeHasPrerequisite(runtime, pathKey));
      availability[stage.id] = {
        visible: true,
        enabled: missing.length === 0,
        reason: missing.length ? `Waiting for required upstream outputs: ${missing.join(', ')}` : '',
      };
    }
    return availability;
  }, [activeJob?.job_id, hasConflictingLiveJob, runtime]);

  useEffect(() => {
    if (segmentationMode === 'manual' && selectedSegmentationBackend === SAM3_SEGMENTER_BACKEND) {
      setStageBackends((current) => {
        const next = {
          ...current,
          [segmentationStageId]: segmenterDefault,
        };
        return next;
      });
    }
  }, [segmentationMode, selectedSegmentationBackend, segmenterDefault]);

  const handlePanelJump = (href) => {
    if (typeof window === 'undefined') {
      return (event) => {
        event.preventDefault();
      };
    }

    return (event) => {
      event.preventDefault();
      window.location.hash = href;
      const targetId = href.replace(/^#/, '');
      const targetElement = document.getElementById(targetId);
      if (!targetElement) {
        return;
      }

      const navHeaderOffset = 16;
      const targetTop = targetElement.getBoundingClientRect().top + window.scrollY - navHeaderOffset;
      const nextTop = Math.max(0, targetTop);
      if (Number.isFinite(nextTop)) {
        window.scrollTo({
          top: nextTop,
          behavior: 'smooth',
        });
      }
    };
  };

  const renderRetryAction = (stageId) => {
    const availability = retryAvailabilityByStage[stageId];
    if (!availability?.visible) {
      return null;
    }
    const isBusy = retryStageBusyId === stageId;
    return (
      <div className="stage-retry-action">
        <button
          className="stage-retry-button"
          type="button"
          onClick={() => submitRetryFromStage(stageId)}
          disabled={Boolean(retryStageBusyId) || !availability.enabled}
          title={availability.reason || undefined}
          aria-label="Retry from this stage"
        >
          <span className="stage-retry-button-kicker">
            {availability.enabled ? 'Resume pipeline' : 'Retry unavailable'}
          </span>
          <span className="stage-retry-button-label">
            {isBusy ? 'Retrying…' : 'Retry from this stage'}
          </span>
        </button>
        {!availability.enabled && availability.reason ? (
          <span className="stage-action-note">{availability.reason}</span>
        ) : null}
      </div>
    );
  };

  const retryErrorForStage = (stageId) => (
    retryStageBusyId === '' && retryStageError && retryStageErrorId === stageId ? retryStageError : null
  );

  return (
    <div className="page-shell">
      <div className="page">
        <section className="hero">
          <div className="hero-grid">
            <div>
              <h1>{PAT3D_FULL_TITLE}</h1>
            </div>
            <div className="hero-aside">
              <div className="hero-panel hero-panel-primary">
                <span className="meta-label">Live pipeline pulse</span>
                <div className="hero-panel-value">{activeLiveJob ? `${jobSummary.percent}%` : 'Idle'}</div>
                <p className="hero-panel-copy">
                  {activeLiveJob
                    ? `${jobSummary.stateMeta.description} ${jobSummary.activeStage ? `Focus: ${jobSummary.activeStage.label}.` : ''}`
                    : historicalJobSummary
                      ? `No run is active. Inspecting the selected ${historicalJobSummary.stateMeta.label.toLowerCase()} run and its saved outputs.`
                      : 'Launch a new PAT3D run from the command deck.'}
                </p>
                <div className="hero-inline-meta">
                  <div className="meta-card">
                    <span className="meta-label">State</span>
                    <div className="meta-value">{activeLiveJob ? jobSummary.stateMeta.label : historicalJobSummary ? 'History' : 'Ready'}</div>
                  </div>
                  <div className="meta-card">
                    <span className="meta-label">Focus</span>
                    <div className="meta-value">{activeLiveJob ? safeText(jobSummary.activeStage?.label, 'Waiting') : historicalJobSummary ? 'Review outputs' : 'Launch control'}</div>
                  </div>
                </div>
              </div>

              <div className="hero-panel hero-panel-secondary">
                <div className="meta-grid">
                  {heroMetrics.map((metric) => (
                    <div className="meta-card" key={metric.label}>
                      <span className="meta-label">{metric.label}</span>
                      <div className="meta-value">{metric.value}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        <div className="main-grid">
          <aside className="sidebar">
            <section className="section-panel control-card sidebar-quick-jump">
              <span className="section-kicker">Quick rail</span>
              <h2 className="section-title">Jump to section</h2>
              <p className="section-copy">
                Jump between configuration, backend controls, telemetry, and each pipeline stage from the left rail.
              </p>
              <div className="rail-jump-stack">
                {quickJumpSections.map((section) => (
                  <article className="rail-jump-group" key={section.title}>
                    <div className="rail-jump-head">
                      <h3 className="rail-jump-title">{section.title}</h3>
                      <p className="rail-jump-copy">{section.copy}</p>
                    </div>
                    <div className="rail-jump-links">
                      {section.links.map((link) => (
                        <a
                          className={`rail-jump-link ${link.disabled ? 'rail-jump-link-disabled' : ''}`}
                          href={link.href}
                          key={`${section.title}-${link.href}-${link.label}`}
                          onClick={link.disabled ? undefined : handlePanelJump(link.clickHref || link.href)}
                          aria-disabled={link.disabled ? 'true' : undefined}
                          aria-label={link.label}
                        >
                          <span className="rail-jump-label" aria-disabled={link.disabled ? 'true' : undefined}>{link.label}</span>
                          <span className="rail-jump-detail">{link.detail}</span>
                        </a>
                      ))}
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <section className="section-panel control-card">
              <span className="section-kicker">Run workspace</span>
              <h2 className="section-title">Current run</h2>
              <p className="section-copy">
                The main canvas follows one dashboard run at a time. Previous runs stay available for inspection without exposing backend file names.
              </p>
              {selectedJob ? (
                <div className="current-run-card">
                  <div className="current-run-header">
                    <div>
                      <div className="meta-label">Run summary</div>
                      <div className="object-name">{safeText(selectedJob.scene_id)}</div>
                    </div>
                    <div className="current-run-actions">
                      <span className={`status-pill ${statusClass(currentRunStatusTone)}`}>{currentRunStateLabel}</span>
                      {activeLiveJob ? (
                        <button
                          type="button"
                          className="secondary-button destructive-button"
                          onClick={() => {
                            void submitCancelRun();
                          }}
                          disabled={cancelRunBusy}
                        >
                          {cancelRunBusy ? 'Cancelling…' : 'Cancel current run'}
                        </button>
                      ) : null}
                    </div>
                  </div>
                  <div className="current-run-grid">
                    <div className="progress-fact">
                      <span className="meta-label">Current stage</span>
                      <strong>{currentRunStageLabel}</strong>
                    </div>
                    <div className="progress-fact">
                      <span className="meta-label">Last update</span>
                      <strong>{formatTimestamp(selectedJob.updated_at)}</strong>
                    </div>
                    <div className="progress-fact">
                      <span className="meta-label">Segmentation</span>
                      <strong>{safeText(selectedJob.segmentation_mode, 'automatic')}</strong>
                    </div>
                  </div>
                  <div className="note-box">
                    {selectedJob.prompt
                      ? safeText(selectedJob.prompt)
                      : 'This run does not have a saved prompt string in the current job payload.'}
                  </div>
                  <ErrorCallout title="Cancel run" error={cancelRunError} />
                </div>
              ) : (
                <div className="empty">No dashboard runs have been recorded yet.</div>
              )}
              <ErrorCallout title="Job polling" error={jobsError} tone="warning" />
              <ErrorCallout title="Run outputs" error={runtimesError} tone="warning" />
              {previousJobs.length ? (
                <details className="run-history">
                  <summary>Switch to a previous run</summary>
                  <label className="form-label" htmlFor="job-history-select">Run history</label>
                  <select
                    id="job-history-select"
                    className="select"
                    value={activeJobId}
                    onChange={(event) => {
                      const nextJobId = event.target.value;
                      const nextJob = jobs.find((job) => job.job_id === nextJobId) || null;
                      setActiveJobId(nextJobId);
                      setActiveJob(nextJob);
                      if (nextJob?.runtime_output_name) {
                        setSelectedRuntime(nextJob.runtime_output_name);
                      }
                      setAutoSelectRunningJob(false);
                      setSyncRuntimeToJob(Boolean(nextJobId));
                    }}
                  >
                    {[selectedJob, ...previousJobs].filter(Boolean).map((job) => (
                      <option key={job.job_id} value={job.job_id}>{job.scene_id} - {job.state}</option>
                    ))}
                  </select>
                </details>
              ) : null}
              <div className="output-browser-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    void refreshWorkspace();
                  }}
                >
                  Refresh now
                </button>
                <button type="button" className="secondary-button" onClick={() => setAutoRefresh((current) => !current)}>
                  {autoRefresh ? 'Pause refresh' : 'Resume refresh'}
                </button>
                <button
                  type="button"
                  className="secondary-button destructive-button"
                  onClick={openCacheCleanupDialog}
                  disabled={hasAnyLiveJobs || cacheCleanupBusy}
                >
                  Clear local run cache
                </button>
              </div>
              <div className="note-box">
                This only clears dashboard-generated local run artifacts and preview caches. It does not touch source code, archived runs, or raw dataset assets.
              </div>
            </section>

            <section className="section-panel">
              <span className="section-kicker">Paper map</span>
              <h2 className="section-title">Paper stage map</h2>
              <div className="timeline paper-map-timeline">
                {stageDefinitions.map((stage) => (
                  <a
                    className={`timeline-item timeline-link ${canJumpToStages ? '' : 'timeline-link-disabled'}`}
                    href={resolveStageTarget(`#stage-${stage.id}`)}
                    key={stage.id}
                    onClick={canJumpToStages ? handlePanelJump(`#stage-${stage.id}`) : undefined}
                    aria-disabled={canJumpToStages ? undefined : 'true'}
                  >
                    <div className="timeline-number">{stage.number}</div>
                    <div>
                      <h3>{stage.label}</h3>
                      <p>{stage.functionality}</p>
                    </div>
                  </a>
                ))}
              </div>
              {canJumpToStages ? null : (
                <p className="note-box">Select a job or run the pipeline, then jump directly into each stage.</p>
              )}
            </section>
          </aside>

          <main className="content">
            <section className="section-panel control-card" id="pipeline-configuration">
              <span className="section-kicker">Command deck</span>
              <h2 className="section-title">Launch from prompt</h2>
              <p className="section-copy">
                Start a new run with a scene prompt, optional object hints, and the routing/model controls below.
              </p>
              <form className="prompt-form" onSubmit={submitJob}>
                <section className="config-subsection prompt-config-subsection" id="prompt-configuration">
                  <div className="prompt-config-head">
                    <div>
                      <span className="form-label">Scene prompt</span>
                      <h3 className="card-title prompt-config-title">Write the scene, keep the rest optional.</h3>
                      <p className="prompt-config-copy">
                        Put the full scene description here. Only use object hints or a scene name when you want tighter control for reruns.
                      </p>
                    </div>
                    <div className="prompt-config-chips">
                      <span className="chip">Prompt required</span>
                      <span className="chip">Objects optional</span>
                    </div>
                  </div>
                  <div className="prompt-config-grid">
                    <div className="prompt-config-panel prompt-config-panel-primary">
                      <label className="form-label" htmlFor="prompt-input">Prompt</label>
                      <textarea
                        id="prompt-input"
                        className="textarea prompt-config-textarea"
                        value={runPrompt}
                        onChange={(event) => setRunPrompt(event.target.value)}
                        rows={5}
                        placeholder="Describe the scene you want to generate"
                      />
                      <div className="prompt-config-helper">
                        Use one clear sentence that covers the objects, composition, and background.
                      </div>
                    </div>
                    <div className="prompt-config-panel prompt-config-panel-secondary">
                      <label className="form-label" htmlFor="objects-input">
                        Requested objects
                        <span className="muted-inline"> · optional</span>
                      </label>
                      <input
                        id="objects-input"
                        className="text-input"
                        value={runObjects}
                        onChange={(event) => setRunObjects(event.target.value)}
                        placeholder="chair, side table"
                      />
                      <div className="prompt-config-helper">
                        Leave blank to let the prompt drive object discovery.
                      </div>

                      <label className="form-label" htmlFor="scene-id-input">Scene name</label>
                      <input
                        id="scene-id-input"
                        className="text-input"
                        value={runSceneId}
                        onChange={(event) => setRunSceneId(event.target.value)}
                        placeholder="optional"
                      />
                      <div className="prompt-config-helper">
                        Only set this when you need a stable scene name for repeated runs or comparisons.
                      </div>
                    </div>
                  </div>
                </section>

                <section className="config-subsection" id="segmentation-configuration">
                  <span className="form-label">Segmentation controls</span>
                  <div className="mode-section">
                    <span className="form-label">Segmentation mode</span>
                    <div className="mode-toggle">
                      <button
                        type="button"
                        className={`backend-option ${segmentationMode === 'automatic' ? 'backend-option-active' : ''}`}
                        onClick={() => setSegmentationMode('automatic')}
                      >
                        Automatic
                      </button>
                      <button
                        type="button"
                        className={`backend-option ${segmentationMode === 'manual' ? 'backend-option-active' : ''}`}
                        onClick={() => setSegmentationMode('manual')}
                      >
                        Hand-tuned masking
                      </button>
                    </div>
                  </div>
                </section>

                <section className="config-subsection" id="model-configuration">
                  <span className="form-label">Model and tuning</span>
                  <label className="form-label" htmlFor="llm-model-select">Structured LLM model</label>
                  <select
                    id="llm-model-select"
                    className="select"
                    value={llmModel}
                    onChange={(event) => setLlmModel(event.target.value)}
                  >
                    {CHAT_MODEL_OPTIONS.map((value) => (
                      <option value={value} key={value}>{value}</option>
                    ))}
                  </select>

                    <label className="form-label" htmlFor="image-model-select">Reference image model</label>
                  <select
                    id="image-model-select"
                    className="select"
                    value={imageModel}
                    onChange={(event) => setImageModel(event.target.value)}
                  >
                    {IMAGE_MODEL_OPTIONS.map((value) => (
                      <option value={value} key={value}>{value}</option>
                    ))}
                  </select>

                  <label className="toggle-card" htmlFor="object-crop-completion-toggle">
                    <input
                      id="object-crop-completion-toggle"
                      className="toggle-input"
                      type="checkbox"
                      aria-label="Enable stage-04 crop completion"
                      checked={objectCropCompletionEnabled}
                      onChange={(event) => setObjectCropCompletionEnabled(event.target.checked)}
                    />
                    <div className="toggle-copy">
                      <span className="form-label">Enable stage-04 crop completion</span>
                      <span className="mono">
                        Complete segmented object crops with {DEFAULT_OBJECT_CROP_COMPLETION_MODEL} before Hunyuan generation. White background is added automatically.
                      </span>
                    </div>
                  </label>

                  <label className="form-label" htmlFor="structured-llm-attempts-input">Structured LLM max attempts</label>
                  <input
                    id="structured-llm-attempts-input"
                    className="text-input"
                    type="number"
                    min="1"
                    max="10"
                    step="1"
                    value={structuredLlmMaxAttempts}
                    onChange={(event) => setStructuredLlmMaxAttempts(sanitizeStructuredLlmMaxAttempts(event.target.value))}
                  />
                  <div className="note-box">
                    Stage 03 will retry structured size inference up to this many times before pausing for retry or manual dimensions. Default: 3.
                  </div>

                  <label className="form-label" htmlFor="requested-object-inference-budget-input">Requested object inference budget</label>
                  <input
                    id="requested-object-inference-budget-input"
                    className="text-input"
                    type="number"
                    min="256"
                    max="65536"
                    step="1"
                    value={requestedObjectInferenceBudget}
                    onChange={(event) => setRequestedObjectInferenceBudget(sanitizeRequestedObjectInferenceBudget(event.target.value))}
                  />
                  <div className="budget-slider-row">
                    <div className="budget-slider-stack">
                      <input
                        id="requested-object-inference-budget-slider"
                        className="range-input"
                        type="range"
                        min={REQUESTED_OBJECT_INFERENCE_BUDGET_MIN}
                        max={REQUESTED_OBJECT_INFERENCE_BUDGET_MAX}
                        step="256"
                        list="requested-object-inference-budget-preset-list"
                        aria-label="Requested object inference budget slider"
                        value={requestedObjectInferenceBudget}
                        onChange={(event) => setRequestedObjectInferenceBudget(sanitizeRequestedObjectInferenceBudget(event.target.value))}
                      />
                      <datalist id="requested-object-inference-budget-preset-list">
                        {REQUESTED_OBJECT_INFERENCE_BUDGET_PRESETS.map((value) => (
                          <option key={value} value={value} label={formatStructuredLlmBudgetPreset(value)} />
                        ))}
                      </datalist>
                      <div className="budget-mark-row" aria-hidden="true">
                        {REQUESTED_OBJECT_INFERENCE_BUDGET_PRESETS.map((value) => {
                          const active = requestedObjectInferenceBudget === value;
                          const positionPercent = ((value - REQUESTED_OBJECT_INFERENCE_BUDGET_MIN) / (REQUESTED_OBJECT_INFERENCE_BUDGET_MAX - REQUESTED_OBJECT_INFERENCE_BUDGET_MIN)) * 100;
                          return (
                            <span
                              key={value}
                              className={`budget-mark ${active ? 'budget-mark-active' : ''}`}
                              style={{ left: `${positionPercent}%` }}
                            />
                          );
                        })}
                      </div>
                      <div className="budget-preset-row" role="group" aria-label="Requested object inference budget presets">
                        {REQUESTED_OBJECT_INFERENCE_BUDGET_PRESETS.map((value) => {
                          const active = requestedObjectInferenceBudget === value;
                          return (
                            <button
                              key={value}
                              type="button"
                              className={`budget-preset ${active ? 'budget-preset-active' : ''}`}
                              aria-pressed={active}
                              onClick={() => setRequestedObjectInferenceBudget(value)}
                            >
                              {formatStructuredLlmBudgetPreset(value)}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    <span className="meta-label mono">{requestedObjectInferenceBudget.toLocaleString()} tokens</span>
                  </div>
                  <div className="note-box">
                    Maximum completion-token budget for prompt-to-object inference before stage 01. Larger budgets help GPT-5.x emit the requested-object JSON instead of exhausting reasoning first. Default: 1280.
                  </div>

                  <label className="form-label" htmlFor="structured-llm-budget-input">Structured LLM reasoning budget</label>
                  <input
                    id="structured-llm-budget-input"
                    className="text-input"
                    type="number"
                    min="256"
                    max="65536"
                    step="1"
                    value={structuredLlmReasoningBudget}
                    onChange={(event) => setStructuredLlmReasoningBudget(sanitizeStructuredLlmReasoningBudget(event.target.value))}
                  />
                  <div className="budget-slider-row">
                    <div className="budget-slider-stack">
                      <input
                        id="structured-llm-budget-slider"
                        className="range-input"
                        type="range"
                        min={STRUCTURED_LLM_REASONING_BUDGET_MIN}
                        max={STRUCTURED_LLM_REASONING_BUDGET_MAX}
                        step="256"
                        list="structured-llm-budget-preset-list"
                        aria-label="Structured LLM reasoning budget slider"
                        value={structuredLlmReasoningBudget}
                        onChange={(event) => setStructuredLlmReasoningBudget(sanitizeStructuredLlmReasoningBudget(event.target.value))}
                      />
                      <datalist id="structured-llm-budget-preset-list">
                        {STRUCTURED_LLM_REASONING_BUDGET_PRESETS.map((value) => (
                          <option key={value} value={value} label={formatStructuredLlmBudgetPreset(value)} />
                        ))}
                      </datalist>
                      <div className="budget-mark-row" aria-hidden="true">
                        {STRUCTURED_LLM_REASONING_BUDGET_PRESETS.map((value) => {
                          const active = structuredLlmReasoningBudget === value;
                          const positionPercent = ((value - STRUCTURED_LLM_REASONING_BUDGET_MIN) / (STRUCTURED_LLM_REASONING_BUDGET_MAX - STRUCTURED_LLM_REASONING_BUDGET_MIN)) * 100;
                          return (
                            <span
                              key={value}
                              className={`budget-mark ${active ? 'budget-mark-active' : ''}`}
                              style={{ left: `${positionPercent}%` }}
                            />
                          );
                        })}
                      </div>
                      <div className="budget-preset-row" role="group" aria-label="Structured LLM reasoning budget presets">
                        {STRUCTURED_LLM_REASONING_BUDGET_PRESETS.map((value) => {
                          const active = structuredLlmReasoningBudget === value;
                          return (
                            <button
                              key={value}
                              type="button"
                              className={`budget-preset ${active ? 'budget-preset-active' : ''}`}
                              aria-pressed={active}
                              onClick={() => setStructuredLlmReasoningBudget(value)}
                            >
                              {formatStructuredLlmBudgetPreset(value)}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    <span className="meta-label mono">{structuredLlmReasoningBudget.toLocaleString()} tokens</span>
                  </div>
                  <div className="note-box">
                    Maximum completion-token budget for structured LLM calls. Larger budgets help GPT-5.x emit JSON for scene-scale inference. Default: 12800.
                  </div>

                  <label className="form-label" htmlFor="reasoning-effort-select">Structured LLM reasoning effort</label>
                  <select
                    id="reasoning-effort-select"
                    className="select"
                    value={reasoningEffort}
                    onChange={(event) => setReasoningEffort(event.target.value)}
                  >
                    {REASONING_EFFORT_OPTIONS.map((value) => (
                      <option value={value} key={value}>{value}</option>
                    ))}
                  </select>

                  <label className="form-label" htmlFor="preview-angle-input">Preview angles</label>
                  <input
                    id="preview-angle-input"
                    className="text-input"
                    type="number"
                    min="1"
                    max="24"
                    step="1"
                    value={previewAngleCount}
                    onChange={(event) => setPreviewAngleCount(sanitizePreviewAngleCount(event.target.value))}
                  />
                  <div className="note-box">
                    The geometry renderer will export this many orbiting preview cameras for the final-stage gallery. Default: 12.
                  </div>
                  <div className="note-box">
                    {segmentationMode === 'manual'
                      ? 'Manual mode generates the reference image first, then pauses at stage 2 so you can paint per-object masks before continuing.'
                      : selectedSegmentationBackend === SAM3_SEGMENTER_BACKEND
                        ? 'Automatic mode will run SAM 3 using the requested-object hints as text prompts. Leave them empty to infer objects from the prompt.'
                        : 'Automatic mode runs the full pipeline end-to-end with the configured segmenter.'}
                  </div>
                </section>

                <section className="config-subsection" id="physics-configuration">
                  <span className="form-label">Physics controls</span>
                  <label className="toggle-card" htmlFor="diff-sim-toggle">
                    <input
                      id="diff-sim-toggle"
                      className="toggle-input"
                      type="checkbox"
                      aria-label="Enable diff-sim initialization"
                      checked={physicsSettings.diffSimEnabled}
                      disabled={!physicsControlsEnabled}
                      onChange={(event) => setPhysicsSettings((current) => ({
                        ...current,
                        diffSimEnabled: event.target.checked,
                      }))}
                    />
                    <div className="toggle-copy">
                      <span className="form-label">Enable diff-sim initialization</span>
                    </div>
                  </label>
                  {!physicsControlsEnabled ? (
                    <div className="note-box">
                      The selected physics backend does not expose dashboard physics controls.
                    </div>
                  ) : null}
                  <div className="control-grid">
                    <label>
                      <span className="form-label">End frame</span>
                      <input
                        className="text-input"
                        type="number"
                        min="1"
                        max="5000"
                        step="1"
                        value={physicsSettings.endFrame}
                        disabled={!physicsControlsEnabled}
                        onChange={(event) => setPhysicsSettings((current) => ({
                          ...current,
                          endFrame: sanitizePhysicsInteger(event.target.value, current.endFrame, 1, 5000),
                        }))}
                      />
                    </label>
                    <label>
                      <span className="form-label">Optimization epochs</span>
                      <input
                        className="text-input"
                        type="number"
                        min="1"
                        max="1000"
                        step="1"
                        value={physicsSettings.totalOptEpoch}
                        disabled={!physicsControlsEnabled || !physicsSettings.diffSimEnabled}
                        onChange={(event) => setPhysicsSettings((current) => ({
                          ...current,
                          totalOptEpoch: sanitizePhysicsInteger(event.target.value, current.totalOptEpoch, 1, 1000),
                        }))}
                      />
                    </label>
                    <label>
                      <span className="form-label">Ground Y</span>
                      <input
                        className="text-input"
                        type="number"
                        min="-10"
                        max="10"
                        step="any"
                        value={physicsSettings.groundYValue}
                        disabled={!physicsControlsEnabled}
                        onChange={(event) => setPhysicsSettings((current) => ({
                          ...current,
                          groundYValue: sanitizePhysicsFloat(
                            event.target.value,
                            current.groundYValue,
                            -10.0,
                            10.0,
                          ),
                        }))}
                      />
                    </label>
                    <label>
                      <span className="form-label">Learning rate</span>
                      <input
                        className="text-input"
                        type="number"
                        min="0.000001"
                        max="1"
                        step="any"
                        value={physicsSettings.physLr}
                        disabled={!physicsControlsEnabled || !physicsSettings.diffSimEnabled}
                        onChange={(event) => setPhysicsSettings((current) => ({
                          ...current,
                          physLr: sanitizePhysicsFloat(event.target.value, current.physLr, 1e-6, 1.0),
                        }))}
                      />
                    </label>
                    <label>
                      <span className="form-label">d_hat</span>
                      <input
                        className="text-input"
                        type="number"
                        min="0.0000001"
                        max="0.1"
                        step="any"
                        value={physicsSettings.contactDHat}
                        disabled={!physicsControlsEnabled}
                        onChange={(event) => setPhysicsSettings((current) => ({
                          ...current,
                          contactDHat: sanitizePhysicsFloat(event.target.value, current.contactDHat, 1e-7, 1e-1),
                        }))}
                      />
                    </label>
                    <label>
                      <span className="form-label">eps_velocity</span>
                      <input
                        className="text-input"
                        type="number"
                        min="0.00000001"
                        max="0.1"
                        step="any"
                        value={physicsSettings.contactEpsVelocity}
                        disabled={!physicsControlsEnabled}
                        onChange={(event) => setPhysicsSettings((current) => ({
                          ...current,
                          contactEpsVelocity: sanitizePhysicsFloat(
                            event.target.value,
                            current.contactEpsVelocity,
                            1e-8,
                            1e-1,
                          ),
                        }))}
                      />
                    </label>
                  </div>
                  <div className="note-box">
                    Forward simulation and diff-sim both start from the same requested ground Y. When diff-sim is off, stage 7 still uses the requested ground Y together with end frame, d_hat, and eps_velocity.
                  </div>
                </section>

                <div className="control-row">
                  <button
                    className="prompt-submit-button"
                    type="submit"
                    disabled={submitting || runButtonShowsActive}
                  >
                    {submitting
                      ? 'Starting…'
                      : runButtonShowsActive
                        ? 'Current run active'
                      : segmentationMode === 'manual'
                        ? 'Generate reference image'
                        : 'Generate'}
                  </button>
                  <button type="button" onClick={() => setAutoRefresh((current) => !current)}>{autoRefresh ? 'Pause refresh' : 'Resume refresh'}</button>
                </div>
                {hasAnyLiveJobs ? (
                  <div className="note-box">
                    A run is already active. Retry the current run from the stage you want, or wait for it to finish before launching a new one.
                  </div>
                ) : null}
              </form>
              <ErrorCallout title="Submit error" error={submitError} />
            </section>

            <StageBackendSelector
              id="stage-backends"
              selected={stageBackends}
              segmentationMode={segmentationMode}
              onSelect={(stageId, value) => {
                setStageBackends((current) => ({ ...current, [stageId]: value }));
              }}
            />

            <ProgressPanel job={activeJob} autoRefresh={autoRefresh && hasAnyLiveJobs} />
            <ErrorCallout
              title="Run output status"
              error={runtimePending && activeRuntimeMatchesJob && Boolean(activeLiveJob) ? runtimePending : null}
              tone="warning"
            />
            <ErrorCallout title="Run output load" error={runtimeError} />
            <LogPanel text={jobLogText} error={jobLogError} />
            {canRedoSegmentationManually || canRetryAnyStage ? (
              <section className="section-panel control-card">
                <span className="section-kicker">Recovery</span>
                <h2 className="section-title">Recovery controls</h2>
                <p className="section-copy">
                  Reopen manual masking from the saved reference image, or use the per-stage retry buttons below to restart any stage. Retrying a live run interrupts the current worker immediately and restarts from the selected stage.
                </p>
                <div className="control-row">
                  {canRedoSegmentationManually ? (
                    <button type="button" onClick={submitRedoSegmentationManually} disabled={manualRedoBusy}>
                      {manualRedoBusy ? 'Reopening…' : 'Redo segmentation manually'}
                    </button>
                  ) : null}
                </div>
                <ErrorCallout title="Manual segmentation reset" error={manualRedoError} />
                <ErrorCallout title="Retry error" error={retryStageError} />
              </section>
            ) : null}

            <section className="section-panel" id="runtime-telemetry">
              <div className="section-heading-row">
                <div>
                  <span className="section-kicker">Run telemetry</span>
                  <h2 className="section-title">Status snapshot</h2>
                  <p className="section-copy">
                    This reads directly from the selected run outputs and flags where the implementation still diverges from the paper.
                  </p>
                </div>
                {runtime ? <span className="chip">Artifacts: {artifactCount}</span> : null}
              </div>
              {loading && !runtime ? <div className="empty">Loading run outputs...</div> : null}
              {!loading && !runtime ? <div className="empty">Select a run to inspect pipeline artifacts and stage alignment.</div> : null}
              {statusCards.length ? (
                <div className="stats-grid">
                  {statusCards.map((card) => (
                    <article className={`stat-card ${toneClass(card.tone)}`} key={card.label}>
                      <span className="meta-label">{card.label}</span>
                      <div className="stat-value">{card.value}</div>
                      <div className="stat-note">{card.note}</div>
                    </article>
                  ))}
                </div>
              ) : null}
            </section>

            <section className="overview-grid">
              {insightCards.map((card) => (
                <article className={`info-card ${toneClass(card.tone)}`} key={card.title}>
                  <h3 className="card-title">{card.title}</h3>
                  <p className="card-text">{card.text}</p>
                </article>
              ))}
            </section>

            {runtime && stageModels ? (
              <section className="stage-stack" id="stage-evidence">
                <section className="section-panel">
                  <span className="section-kicker">Stage evidence</span>
                  <h2 className="section-title">Detailed paper-stage review</h2>
                  <p className="section-copy">
                    Drill into each stage output, compare providers, and inspect geometry, relations, and final rendering artifacts in sequence.
                  </p>
                </section>

                <StageCard
                  anchorId={`stage-${stageDefinitions[0].id}`}
                  stage={stageDefinitions[0]}
                  {...resolveStageProgress(liveStageRows, stageDefinitions[0].id, stageModels.reference.status)}
                  providerName={stageModels.reference.providerName}
                  actions={renderRetryAction(stageDefinitions[0].id)}
                  actionError={retryErrorForStage(stageDefinitions[0].id)}
                >
                  <div className="media-grid media-grid-single">
                    <MediaCard title="Reference image" path={stageModels.reference.data?.image?.path} caption={`Prompt: ${safeText(stageModels.reference.data?.generation_prompt)}`} />
                  </div>
                </StageCard>

                <StageCard
                  anchorId={`stage-${stageDefinitions[1].id}`}
                  stage={stageDefinitions[1]}
                  {...resolveStageProgress(
                    liveStageRows,
                    stageDefinitions[1].id,
                    waitingForManualMasks ? 'partial' : stageModels.understanding.status,
                  )}
                  providerName={understandingProviderName}
                  actions={renderRetryAction(stageDefinitions[1].id)}
                  actionError={retryErrorForStage(stageDefinitions[1].id)}
                >
                  <div className="info-card">
                    <div className="requested-objects-head">
                      <h3 className="card-title">Requested objects</h3>
                      <span className={`chip ${statusClass(requestedObjectSourceTone)}`}>
                        {requestedObjectSourceLabel}
                      </span>
                    </div>
                    {hasRequestedObjects ? (
                      <div className="requested-object-chips">
                        {requestedObjectLabels.map((item) => (
                          <span className="requested-object-chip" key={item}>{item}</span>
                        ))}
                      </div>
                    ) : (
                      <div className="empty">No requested objects were recorded for this run.</div>
                    )}
                  </div>
                  <div className="media-grid">
                    <MediaCard title="Depth visualization" path={stageModels.understanding.data?.depth_result?.depth_visualization?.path} caption={`Focal length: ${safeText(stageModels.understanding.data?.depth_result?.focal_length_px, 'n/a')}`} />
                    {waitingForManualMasks ? (
                      <article className="media-card">
                        <h4 className="media-title">Manual masking checkpoint</h4>
                        <div className="note-box">
                          {manualMaskCount
                            ? `${manualMaskCount} saved mask layers are ready for review. Keep painting or continue the run when the object coverage looks right.`
                            : 'The reference image is ready. Add a layer for each object you want to segment, paint the masks, then continue the pipeline.'}
                        </div>
                      </article>
                    ) : (
                      <MediaCard title="Segmentation composite" path={stageModels.understanding.data?.segmentation_result?.composite_visualization?.path} caption={`${asList(stageModels.understanding.data?.segmentation_result?.instances).length} segmented instances`} />
                    )}
                  </div>
                  {waitingForManualMasks ? (
                    <div className="stage-section-gap">
                      <SegmentationEditor
                        referencePath={stageModels.reference.data?.image?.path}
                        versionToken={activeJob?.manual_segmentation?.updated_at || activeJob?.updated_at || activeJobId}
                        initialInstances={asList(activeJob?.manual_segmentation?.instances)}
                        suggestedLabels={requestedObjectLabels}
                        busy={manualMaskBusy}
                        onSubmit={submitManualMasks}
                      />
                      <ErrorCallout title="Manual masking" error={manualMaskError} />
                    </div>
                  ) : (
                    <div className="info-card stage-section-gap">
                      <h3 className="card-title">Object inventory</h3>
                      <Table
                        headers={['ID', 'Name', 'Count', 'Instances']}
                        rows={objectInventoryRows(stageModels.understanding.data?.object_catalog)}
                      />
                    </div>
                  )}
                </StageCard>

                <StageCard
                  anchorId={`stage-${stageDefinitions[2].id}`}
                  stage={stageDefinitions[2]}
                  {...resolveStageProgress(
                    liveStageRows,
                    stageDefinitions[2].id,
                    stageModels.relation.status,
                  )}
                  providerName={stageModels.relation.providerName}
                  actions={renderRetryAction(stageDefinitions[2].id)}
                  actionError={retryErrorForStage(stageDefinitions[2].id)}
                >
                  <RelationGraphEditor
                    graph={relationData?.relation_graph}
                    objectCatalog={relationEditorCatalog}
                    objectDescriptions={relationData?.object_descriptions}
                    editable={Boolean(activeJobId && activeRuntimeMatchesJob)}
                    sourceKey={relationSourceKey}
                    saveBusy={relationGraphSaveBusy}
                    saveError={relationGraphSaveError}
                    hasSavedOverride={Boolean(relationOverrideGraph)}
                    relationRetryStage={relationRetryStage}
                    onSave={(graphPayload) => submitRelationGraph(graphPayload)}
                    onClearSavedOverride={() => submitRelationGraph(null, { clear: true })}
                    onSaveAndRetry={(graphPayload) => submitRelationGraph(graphPayload, { retryStage: relationRetryStage?.id || '' })}
                  />
                  <div className="stage-grid-single stage-section-gap">
                    <div className="info-card">
                      <h3 className="card-title">Size priors</h3>
                      {waitingForManualSizes ? (
                        <div className="stage-section-gap">
                          <div className="note-box">
                            Structured size inference did not return a usable JSON result. Retry stage 03 or enter manual dimensions to continue from object assets.
                          </div>
                          <ErrorCallout
                            title="Size inference"
                            error={manualSizeError || manualSizeStageError}
                            tone="warning"
                          />
                          <table className="data-table">
                            <thead>
                              <tr>
                                <th>Object</th>
                                <th>Width</th>
                                <th>Depth</th>
                                <th>Height</th>
                              </tr>
                            </thead>
                            <tbody>
                              {manualSizeEntries.map((entry, index) => (
                                <tr key={entry.object_id || index}>
                                  <td>{formatObjectDisplayName(entry.display_name || entry.object_id)}</td>
                                  {['x', 'y', 'z'].map((axis) => (
                                    <td key={axis}>
                                      <input
                                        className="text-input"
                                        type="number"
                                        min="0.001"
                                        step="0.001"
                                        value={entry?.dimensions_m?.[axis] ?? ''}
                                        onChange={(event) => {
                                          const value = event.target.value;
                                          setManualSizeEntries((current) => current.map((item, itemIndex) => (
                                            itemIndex === index
                                              ? {
                                                ...item,
                                                dimensions_m: {
                                                  ...item.dimensions_m,
                                                  [axis]: value,
                                                },
                                              }
                                              : item
                                          )));
                                        }}
                                      />
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          <div className="control-row">
                            <button
                              type="button"
                              onClick={() => submitManualSizes()}
                              disabled={manualSizeBusy || !manualSizeEntries.length}
                            >
                              {manualSizeBusy ? 'Continuing…' : 'Continue with manual dimensions'}
                            </button>
                          </div>
                        </div>
                      ) : (
                        <Table
                          headers={['Object', 'Width', 'Depth', 'Height']}
                          rows={sizePriorRows(relationData?.size_priors, relationEditorCatalog)}
                        />
                      )}
                    </div>
                  </div>
                </StageCard>

                <StageCard
                  anchorId={`stage-${stageDefinitions[3].id}`}
                  stage={stageDefinitions[3]}
                  {...resolveStageProgress(
                    liveStageRows,
                    stageDefinitions[3].id,
                    stageModels.assets.status,
                  )}
                  providerName={stageModels.assets.providerName}
                  actions={renderRetryAction(stageDefinitions[3].id)}
                  actionError={retryErrorForStage(stageDefinitions[3].id)}
                >
                  <div className="asset-stage-shell">
                    {activeAssetEntry ? (
                      <section className="asset-slider-shell">
                        <div className="asset-slider-summary">
                          <span className="meta-label">Object asset navigator</span>
                          <h3 className="card-title">{formatObjectDisplayName(activeAssetEntry.object_id)}</h3>
                          <p className="asset-section-copy">
                            Slide across the object lineup, then use the tabs directly below to jump to a specific asset without losing context.
                          </p>
                        </div>
                        <div className="asset-slider-range-panel">
                          <div className="asset-slider-range-head">
                            <label className="form-label" htmlFor="object-asset-slider">Selected asset</label>
                            <div className="asset-slider-range-actions">
                              <span className="mono asset-slider-range-value">
                                {clampedActiveAssetIndex + 1} / {assetEntries.length}
                              </span>
                            </div>
                          </div>
                          <input
                            id="object-asset-slider"
                            className="range-input asset-slider-input asset-slider-input-prominent"
                            type="range"
                            min="0"
                            max={Math.max(assetEntries.length - 1, 0)}
                            step="1"
                            value={clampedActiveAssetIndex}
                            onChange={(event) => selectAssetAtIndex(Number(event.target.value))}
                            aria-label="Object asset slider"
                            disabled={assetEntries.length <= 1}
                          />
                          <div
                            className="asset-slider-ticks"
                            aria-hidden="true"
                            style={{ '--asset-tick-count': Math.max(assetEntries.length, 1) }}
                          >
                            {assetEntries.map((asset, index) => (
                              <span
                                key={`asset-slider-tick-${asset.object_id}`}
                                className={`asset-slider-tick ${index === clampedActiveAssetIndex ? 'asset-slider-tick-active' : ''}`}
                              />
                            ))}
                          </div>
                        </div>
                        <div className="asset-slider-rail scroll-surface scroll-surface-horizontal" role="tablist" aria-label="Object asset navigator">
                          {assetEntries.map((asset, index) => (
                            <button
                              ref={(node) => setAssetSliderTabRef(asset.object_id, node)}
                              key={`asset-nav-${asset.object_id}`}
                              type="button"
                              role="tab"
                              className={`asset-slider-card ${index === clampedActiveAssetIndex ? 'asset-slider-card-active' : ''}`}
                              aria-selected={index === clampedActiveAssetIndex}
                              aria-label={`Show asset ${asset.object_id}`}
                              onClick={() => selectAssetAtIndex(index)}
                            >
                              <span className="asset-slider-card-index">{index + 1}</span>
                              <span className="asset-slider-card-title">{formatObjectDisplayName(asset.object_id)}</span>
                              <span className="asset-slider-card-meta">{safeText(asset.metadata?.provider_name)}</span>
                            </button>
                          ))}
                        </div>
                      </section>
                    ) : (
                      <div className="empty">No object assets were recorded for this stage.</div>
                    )}

                    {[activeAssetEntry].filter(Boolean).map((asset) => {
                      const previewStillPath = asset.preview_image?.path || asset.texture_image?.path;
                      const previewIsOpen = activeAssetPreviewId === asset.object_id;
                      const canOpenPreview = Boolean(asset.mesh_obj?.path);

                      return (
                        <article className="object-card asset-card-shell asset-card-shell-wide" key={asset.object_id}>
                          <div className="object-head asset-card-head">
                            <div className="asset-card-title">
                              <div className="object-name">{formatObjectDisplayName(asset.object_id)}</div>
                              <div className="mono">{safeText(asset.metadata?.provider_name)}</div>
                            </div>
                            <div className="asset-card-status-row">
                              <span className={`chip ${canOpenPreview ? '' : 'chip-muted'}`}>
                                {canOpenPreview ? 'mesh ready' : 'mesh pending'}
                              </span>
                              <span className="status-pill status-aligned">textured</span>
                            </div>
                          </div>
                          <div className="object-body asset-card-body">
                            <div className="asset-card-workstation">
                              <section className="asset-section-card asset-card-preview-column">
                                <div className="asset-section-head">
                                  <div>
                                    <span className="meta-label">Object asset preview</span>
                                    <p className="asset-section-copy">
                                      Use the rendered still to confirm silhouette, surface response, and crop quality before inspecting the mesh.
                                    </p>
                                  </div>
                                  <span className="note-inline">{previewStillPath ? 'still captured' : 'still unavailable'}</span>
                                </div>
                                {previewStillPath ? (
                                  <div className="media-frame compact-frame">
                                    <img src={artifactUrl(previewStillPath)} alt={`${asset.object_id} asset preview`} loading="lazy" />
                                  </div>
                                ) : (
                                  <div className="empty">No still preview was recorded for this asset.</div>
                                )}
                              </section>
                              <section className="asset-section-card asset-card-viewer-column">
                                <div className="asset-section-head">
                                  <div>
                                    <span className="meta-label">Interactive mesh</span>
                                    <p className="asset-section-copy">
                                      Orbit the generated mesh for the selected object. The viewer opens automatically when the selected asset is ready, and you can pause it any time.
                                    </p>
                                  </div>
                                  <div className="asset-card-actions">
                                    <button
                                      type="button"
                                      className={`ghost-button ${previewIsOpen ? 'backend-option-active' : ''}`}
                                      onClick={() => setActiveAssetPreviewId((current) => (
                                        current === asset.object_id ? '' : asset.object_id
                                      ))}
                                      disabled={!canOpenPreview}
                                      aria-pressed={previewIsOpen}
                                      aria-label={previewIsOpen ? `Close 3D preview for ${asset.object_id}` : `Open 3D preview for ${asset.object_id}`}
                                    >
                                      {previewIsOpen ? 'Close viewer' : 'Open viewer'}
                                    </button>
                                  </div>
                                </div>
                                {!asset.mesh_obj?.path ? (
                                  <div className="asset-viewer-idle">Mesh preview will appear here as soon as this object finishes generating.</div>
                                ) : previewIsOpen ? (
                                  <MeshPreview
                                    objPath={asset.mesh_obj?.path}
                                    texturePath={asset.texture_image?.path}
                                    previewImagePath={asset.preview_image?.path}
                                  />
                                ) : (
                                  <div className="asset-viewer-idle">
                                    Interactive 3D preview is paused until opened.
                                  </div>
                                )}
                              </section>
                              <section className="asset-section-card asset-card-info-column">
                                <div className="asset-section-head">
                                  <div>
                                    <span className="meta-label">Asset record</span>
                                    <p className="asset-section-copy">
                                      Provider, readiness, and every exported artifact for the selected object.
                                    </p>
                                  </div>
                                  <span className={`status-pill ${canOpenPreview ? 'status-aligned' : 'status-pending'}`}>
                                    {canOpenPreview ? 'ready for inspection' : 'waiting for mesh'}
                                  </span>
                                </div>
                                <div className="asset-meta-grid">
                                  <div className="asset-meta-card">
                                    <span className="meta-label">Provider</span>
                                    <div className="asset-meta-value mono">{safeText(asset.metadata?.provider_name)}</div>
                                  </div>
                                  <div className="asset-meta-card">
                                    <span className="meta-label">Mesh viewer</span>
                                    <div className="asset-meta-value">{previewIsOpen ? 'open' : 'paused'}</div>
                                  </div>
                                  <div className="asset-meta-card">
                                    <span className="meta-label">Still preview</span>
                                    <div className="asset-meta-value">{previewStillPath ? 'recorded' : 'missing'}</div>
                                  </div>
                                </div>
                                {!asset.mesh_obj?.path ? (
                                  <div className="note-box">
                                    This viewer unlocks per object, not per stage. As soon as this asset has a mesh, you can inspect it here.
                                  </div>
                                ) : null}
                                <ArtifactLinks
                                  items={[
                                    { label: meshLabelFromPath(asset.mesh_obj?.path), path: asset.mesh_obj?.path },
                                    { label: 'MTL', path: asset.mesh_mtl?.path },
                                    { label: 'Texture', path: asset.texture_image?.path },
                                  ].filter((item) => item.path || item.href)}
                                />
                              </section>
                            </div>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                </StageCard>

                <StageCard
                  anchorId={`stage-${stageDefinitions[4].id}`}
                  stage={stageDefinitions[4]}
                  {...resolveStageProgress(
                    liveStageRows,
                    stageDefinitions[4].id,
                    stageModels.layout.status,
                  )}
                  providerName={stageModels.layout.providerName}
                  actions={renderRetryAction(stageDefinitions[4].id)}
                  actionError={retryErrorForStage(stageDefinitions[4].id)}
                >
                  <div className="stage-grid-single">
                    <LayoutPoseMap2D
                      layout={layout}
                      sizePriors={relationData?.size_priors}
                      objectCatalog={relationEditorCatalog}
                      title="Footprint map"
                      subtitle="Top-down X/Z footprints per object, recentered on the occupied scene so overlap and spacing read cleanly."
                      ariaLabel="Top-down X/Z footprint map"
                    />
                  </div>
                </StageCard>

                <StageCard
                  anchorId={`stage-${stageDefinitions[5].id}`}
                  stage={stageDefinitions[5]}
                  {...simulationPreparationProgress}
                  providerName={stageModels.simulation.providerName}
                  actions={renderRetryAction(stageDefinitions[5].id)}
                  actionError={retryErrorForStage(stageDefinitions[5].id)}
                >
                  <div className="stage-grid-single">
                    <div className="info-card">
                      <h3 className="card-title">Simulation preparation</h3>
                      <p className="card-text">
                        Stage 06 prepares the low-poly collision package consumed by the stage-07 forward simulator.
                        Live preparation progress stays in the stage header while this step is running.
                      </p>
                      {simulationPreparationProgress.message ? (
                        <div className="note-box">{simulationPreparationProgress.message}</div>
                      ) : null}
                    </div>
                  </div>
                </StageCard>

                <StageCard
                  anchorId={`stage-${stageDefinitions[6].id}`}
                  stage={stageDefinitions[6]}
                  {...physicsStageProgress}
                  providerName={stageModels.physics.providerName}
                  actions={renderRetryAction(stageDefinitions[6].id)}
                  actionError={retryErrorForStage(stageDefinitions[6].id)}
                >
                  <div className="stage-grid">
                    <div className="info-card">
                      <h3 className="card-title">Physics result</h3>
                      <ul className="bullet-list">
                        <li>Provider: {safeText(stageModels.physics.providerName)}</li>
                        <li>Mode: {safeText(physicsModeLabel, 'n/a')}</li>
                        <li>Simulated object poses: {asList(stageModels.physics.data?.optimized_object_poses).length}</li>
                        <li>Best loss: {physicsMetricsPayload?.best_loss !== null && physicsMetricsPayload?.best_loss !== undefined ? Number(physicsMetricsPayload.best_loss).toFixed(6) : 'n/a'}</li>
                        <li>Forward final frame: {stageModels.physics.data?.metrics?.forward_simulator_final_frame ?? 'n/a'}</li>
                        <li>Static stop: {stageModels.physics.data?.metrics?.forward_simulator_stopped_static === 1 ? 'yes' : 'no / unknown'}</li>
                        <li>
                          Penetration metric: {formatPhysicsMetricValue(penetrationMetric)}
                          {!penetrationMetric?.available && penetrationMetric?.note ? ` (${penetrationMetric.note})` : ''}
                        </li>
                        <li>
                          Displacement metric: {formatPhysicsMetricValue(displacementMetric, 'Pending GIPC integration')}
                        </li>
                        <li>This is the final scene state used for visualization export.</li>
                      </ul>
                    </div>
                    <div className="info-card">
                      <h3 className="card-title">Optimization loss</h3>
                      {physicsMetricsError ? <ErrorCallout title="Physics loss" error={physicsMetricsError} /> : null}
                      <LossPlot
                        metrics={physicsMetricsPayload}
                        physicsMode={physicsModeLabel}
                        diffSimRequested={diffSimRequested}
                        diffSimAttempted={diffSimAttempted}
                      />
                    </div>
                  </div>
                  <div className="stage-grid-single stage-section-gap">
                    <div className="info-card">
                      <SimulationPassTree
                        relationGraph={relationData?.relation_graph}
                        objectCatalog={relationEditorCatalog}
                        report={physicsDebugReportPayload}
                        passes={physicsPasses}
                        passProgress={physicsPassProgress}
                        activePassIndex={activePhysicsPassIndex}
                        onSelectPass={setActivePhysicsPassIndex}
                      />
                      {physicsDebugReportError ? <ErrorCallout title="Physics pass report" error={physicsDebugReportError} /> : null}
                      {activePhysicsPass ? (
                        <div className="note-box">
                          Selected pass container: {formatObjectDisplayName(activePhysicsPass.container_name, 'ground')}. Strategy: {safeText(activePhysicsPass.strategy, 'n/a')}. Simulated objects: {safeText(asList(activePhysicsPass.simulated_object_ids).length, '0')}.
                        </div>
                      ) : null}
                    </div>
                  </div>
                  <div className="stage-grid-single stage-section-gap">
                    <div className="info-card">
                      <h3 className="card-title">Simulation trajectory</h3>
                      {trajectoryError ? <ErrorCallout title="Trajectory load" error={trajectoryError} /> : null}
                      <TrajectoryPlot trajectory={trajectoryPayload} />
                    </div>
                  </div>
                </StageCard>

                <StageCard
                  anchorId={`stage-${stageDefinitions[7].id}`}
                  stage={stageDefinitions[7]}
                  {...resolveStageProgress(
                    liveStageRows,
                    stageDefinitions[7].id,
                    stageModels.visualization.status,
                  )}
                  providerName={stageModels.visualization.providerName}
                  actions={renderRetryAction(stageDefinitions[7].id)}
                  actionError={retryErrorForStage(stageDefinitions[7].id)}
                >
                  <div className="stage-grid-single">
                    <div className="info-card">
                      <div className="card-toolbar">
                        <div>
                          <h3 className="card-title">Interactive 3D scene viewer</h3>
                          <div className="meta-label">{activeSceneViewerMode === 'original' ? 'Original textured mesh' : 'Simplified mesh'}</div>
                        </div>
                        <div className="card-actions">
                          <div className="mode-toggle">
                            <button
                              type="button"
                              className={`backend-option ${activeSceneViewerMode === 'simplified' ? 'backend-option-active' : ''}`}
                              onClick={() => setActiveSceneViewerMode('simplified')}
                              disabled={!sceneBundleSimplified}
                              aria-pressed={activeSceneViewerMode === 'simplified'}
                            >
                              Simplified mesh
                            </button>
                            <button
                              type="button"
                              className={`backend-option ${activeSceneViewerMode === 'original' ? 'backend-option-active' : ''}`}
                              onClick={() => setActiveSceneViewerMode('original')}
                              disabled={!sceneBundleOriginal}
                              aria-pressed={activeSceneViewerMode === 'original'}
                            >
                              Original textured mesh
                            </button>
                          </div>
                          <button
                            type="button"
                            className="ghost-button"
                            disabled={!activeSceneBundle || sceneExportMode !== ''}
                            onClick={() => {
                              void downloadSceneExport('merged', activeSceneBundle, activeSceneSourceLabel);
                            }}
                          >
                            {sceneExportMode === 'merged' && sceneExportSource === activeSceneSourceLabel
                              ? 'Preparing merged scene…'
                              : 'Download merged scene'}
                          </button>
                          <button
                            type="button"
                            className="ghost-button"
                            disabled={!activeSceneBundle || sceneExportMode !== ''}
                            onClick={() => {
                              void downloadSceneExport('separate', activeSceneBundle, activeSceneSourceLabel);
                            }}
                          >
                            {sceneExportMode === 'separate' && sceneExportSource === activeSceneSourceLabel
                              ? 'Preparing object ZIP…'
                              : 'Download separate objects'}
                          </button>
                        </div>
                      </div>
                      <SceneViewer bundle={activeSceneBundle} version={runtimePayload?.updatedAt || ''} />
                      {sceneExportError && sceneExportSource === activeSceneSourceLabel
                        ? <ErrorCallout title="Scene export" error={sceneExportError} />
                        : null}
                      <div className="note-box">
                        {activeSceneViewerMode === 'original'
                          ? 'This viewer prefers the stage-4 textured meshes and reapplies the final scene transforms. Layout mesh artifacts are only used as a fallback when original object meshes are missing.'
                          : 'Export uses the simplified stage-8 scene bundle with final transforms applied. Use merged scene for one combined OBJ package, or separate objects for a ZIP with one transformed OBJ per object.'}
                      </div>
                    </div>
                  </div>
                  <div className="stage-grid stage-grid-single stage-section-gap">
                    <div className="info-card">
                      <h3 className="card-title">Rendered view</h3>
                      <div className="render-gallery">
                        {stage8PreviewImages.length ? stage8PreviewImages.map((image, index) => (
                          <MediaCard
                            key={image.path || index}
                            title="Rendered view"
                            path={image.path}
                            caption={renderImageCaption(
                              stage8PreviewImages,
                              index,
                              stageModels.visualization.data?.render_config?.mode,
                            )}
                          />
                        )) : <div className="empty">No simplified preview images were recorded for this run.</div>}
                      </div>
                      <div className="note-box">
                        This gallery prefers the textured stage-8 rendered views and falls back to simplified previews only when textured renders were not recorded.
                      </div>
                    </div>
                  </div>
                </StageCard>
              </section>
            ) : null}
            <MetricsPanel
              metricsPayload={metricsPayload}
              metricsBusy={metricsBusy}
              metricsError={metricsError}
              runtime={runtime}
              selectedRuntime={selectedRuntime}
              onCompute={submitMetrics}
            />
          </main>
        </div>
      </div>
      {cacheDialogOpen ? (
        <div className="modal-backdrop" role="presentation" onClick={closeCacheCleanupDialog}>
          <div
            className="modal-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="cache-cleanup-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <div>
                <span className="section-kicker">Maintenance</span>
                <h2 className="section-title" id="cache-cleanup-title">Clear local run cache</h2>
              </div>
            </div>
            <p className="section-copy">
              This removes dashboard-generated local run artifacts so the portal starts from a clean slate. Archived runs and source assets are not deleted.
            </p>
            {cacheSummaryBusy ? <div className="empty">Inspecting local cache…</div> : null}
            {cacheSummary ? (
              <div className="info-card">
                <h3 className="card-title">Deletion preview</h3>
                <Table
                  headers={['Target', 'Path', 'Entries', 'Size']}
                  rows={asList(cacheSummary.targets).map((target) => [
                    safeText(target.label),
                    <span className="mono" key={`${target.id}-path`}>{safeText(target.path)}</span>,
                    safeText(target.entries, '0'),
                    safeText(target.bytes_human, '0 B'),
                  ])}
                />
                <div className="note-box">
                  Total: {safeText(cacheSummary.totals?.entries, '0')} entries · {safeText(cacheSummary.totals?.bytes_human, '0 B')}
                </div>
              </div>
            ) : null}
            {cacheCleanupBlocked ? (
              <div className="error-box">
                <strong>Cleanup blocked</strong>
                <div>Stop all queued, running, or manual-input jobs before clearing the local run cache.</div>
              </div>
            ) : null}
            <ErrorCallout title="Cache summary" error={cacheSummaryError} />
            <ErrorCallout title="Cache cleanup" error={cacheCleanupError} />
            <label className="form-label" htmlFor="cache-confirm-input">
              Type <span className="mono">DELETE</span> to confirm
            </label>
            <input
              id="cache-confirm-input"
              className="text-input"
              value={cacheConfirmText}
              onChange={(event) => setCacheConfirmText(event.target.value)}
              placeholder="DELETE"
              disabled={cacheCleanupBusy}
            />
            <div className="modal-actions">
              <button type="button" className="secondary-button" onClick={closeCacheCleanupDialog} disabled={cacheCleanupBusy}>
                Cancel
              </button>
              <button
                type="button"
                className="secondary-button destructive-button"
                onClick={() => {
                  void confirmLocalCacheCleanup();
                }}
                disabled={!canConfirmCacheCleanup || cacheCleanupBusy}
              >
                {cacheCleanupBusy ? 'Clearing…' : 'Delete local cache'}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
