import { useMemo } from 'react';
import { buildDraftGraph, buildRenderableGraphLayout } from './RelationGraphEditor';
import {
  PASS_NODE_TONES,
  RELATION_GRAPH_THEME,
  getRelationTone,
  isGroundNodeId,
} from './relationPalette';
import { formatObjectDisplayList, formatObjectDisplayName } from './runtimeViewModel';

function asList(value) {
  return Array.isArray(value) ? value : [];
}

function safeText(value, fallback = 'n/a') {
  if (value === null || value === undefined || value === '') return fallback;
  return String(value);
}

function formatPassMode(pass) {
  if (!pass) return 'Unavailable';
  return pass.strategy === 'ground_drop' ? 'Forward only' : 'Diff sim + forward';
}

function formatPassTitle(pass, index) {
  if (!pass) return `Pass ${index + 1}`;
  const strategy = String(pass.strategy || 'pass')
    .split(/[_-]/g)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
  return `Pass ${index + 1} · ${strategy || 'Pass'}`;
}

function summarizeObjectIds(values) {
  return formatObjectDisplayList(values, 'none');
}

function relationKey(relation) {
  return [
    safeText(relation?.parent_object_id, ''),
    safeText(relation?.relation_type, ''),
    safeText(relation?.child_object_id, ''),
  ].join('::');
}

function parseInteger(value) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizePassStatus(status) {
  const normalized = String(status || '').trim().toLowerCase();
  if (['pending', 'running', 'completed', 'failed'].includes(normalized)) {
    return normalized;
  }
  return '';
}

function resolvePassStatus(pass, index, completedPassCount, runningPassIndex) {
  const explicitStatus = normalizePassStatus(pass?.status);
  if (explicitStatus) {
    return explicitStatus;
  }
  if (runningPassIndex !== null && index === runningPassIndex) {
    return 'running';
  }
  if (index < completedPassCount) {
    return 'completed';
  }
  return 'pending';
}

function passStatusLabel(status) {
  if (status === 'completed') return 'Completed';
  if (status === 'running') return 'Running';
  if (status === 'failed') return 'Failed';
  return 'Pending';
}

function passStatusClassName(status) {
  if (status === 'completed') return 'status-aligned';
  if (status === 'running') return 'status-running';
  if (status === 'failed') return 'status-fallback';
  return 'status-pending';
}

function progressToneClassName(completedPassCount, totalPassCount, runningPassIndex) {
  if (runningPassIndex !== null) {
    return 'tone-running';
  }
  if (totalPassCount > 0 && completedPassCount >= totalPassCount) {
    return 'tone-aligned';
  }
  return 'tone-pending';
}

function buildHighlightedEdges(relations, activePass, activeObjectIds) {
  const highlighted = new Set();
  const containerName = String(activePass?.container_name || '');
  const optimizedIds = new Set(asList(activePass?.optimized_object_ids));
  const dynamicIds = new Set(asList(activePass?.dynamic_object_ids));
  const activeIds = new Set(activeObjectIds);

  asList(relations).forEach((relation) => {
    const parentId = String(relation?.parent_object_id || '');
    const childId = String(relation?.child_object_id || '');
    if (!parentId || !childId) return;

    const containerEdge = containerName && parentId === containerName && activeIds.has(childId);
    const optimizationEdge = optimizedIds.has(parentId) && optimizedIds.has(childId);
    const dynamicEdge = activeIds.has(parentId) && dynamicIds.has(childId);
    if (containerEdge || optimizationEdge || dynamicEdge) {
      highlighted.add(relationKey(relation));
    }
  });

  return highlighted;
}

function buildLegendChipStyle(kind) {
  const tone = kind === 'container'
    ? PASS_NODE_TONES.container
    : kind === 'active'
      ? PASS_NODE_TONES.active
      : PASS_NODE_TONES.context;
  return {
    borderColor: tone.stroke,
    background: tone.fill,
    color: PASS_NODE_TONES.base.label,
  };
}

function buildPassEdgeStyle(relation, highlighted) {
  const syntheticRoot = Boolean(relation?.synthetic_root);
  const tone = getRelationTone(relation?.relation_type, { syntheticRoot });
  return {
    line: {
      stroke: tone.stroke,
      strokeWidth: highlighted ? (syntheticRoot ? 3 : 4.1) : (syntheticRoot ? 2.4 : 2.9),
      opacity: highlighted ? 1 : (syntheticRoot ? 0.44 : 0.28),
      strokeDasharray: syntheticRoot ? '8 10' : undefined,
      filter: highlighted ? `drop-shadow(0 0 10px ${tone.glow})` : undefined,
    },
    label: {
      fill: tone.label,
      opacity: highlighted ? 1 : (syntheticRoot ? 0.6 : 0.4),
      fontWeight: highlighted || syntheticRoot ? 700 : 600,
      letterSpacing: '0.08em',
      paintOrder: 'stroke',
      stroke: RELATION_GRAPH_THEME.canvasFill,
      strokeWidth: 4,
      strokeLinecap: 'round',
      strokeLinejoin: 'round',
    },
  };
}

function buildPassNodeStyle({ isGround, isContainer, isActive, isContext }) {
  let tone = PASS_NODE_TONES.base;
  if (isGround) {
    tone = PASS_NODE_TONES.root;
  } else if (isContainer) {
    tone = PASS_NODE_TONES.container;
  } else if (isActive) {
    tone = PASS_NODE_TONES.active;
  } else if (isContext) {
    tone = PASS_NODE_TONES.context;
  }

  if (isGround && isContainer) {
    return {
      box: {
        fill: PASS_NODE_TONES.root.fill,
        stroke: PASS_NODE_TONES.container.stroke,
        strokeWidth: 2.8,
        filter: `drop-shadow(0 18px 28px ${PASS_NODE_TONES.root.shadow})`,
      },
      label: {
        fill: PASS_NODE_TONES.root.label,
        letterSpacing: '0.06em',
      },
      subtitle: {
        fill: PASS_NODE_TONES.container.subtitle,
        letterSpacing: '0.06em',
      },
    };
  }

  return {
    box: {
      fill: tone.fill,
      stroke: tone.stroke,
      strokeWidth: isGround ? 2.2 : (isContainer ? 2.6 : (isActive ? 2.3 : (isContext ? 2 : 1.5))),
      filter: `drop-shadow(0 18px 28px ${tone.shadow})`,
    },
    label: {
      fill: tone.label,
      letterSpacing: isGround ? '0.06em' : '0.01em',
    },
    subtitle: {
      fill: tone.subtitle,
      letterSpacing: '0.06em',
    },
  };
}

export default function SimulationPassTree({
  relationGraph,
  objectCatalog,
  report,
  passes,
  passProgress,
  activePassIndex,
  onSelectPass,
}) {
  const passList = asList(passes);
  const reportedTotal = parseInteger(report?.progressive_pass_count);
  const totalPassCount = Math.max(
    passList.length,
    parseInteger(passProgress?.total) || 0,
    reportedTotal || 0,
  );
  const completedPassCount = Math.max(
    0,
    Math.min(
      totalPassCount || passList.length,
      parseInteger(passProgress?.completed) ?? passList.filter((pass) => normalizePassStatus(pass?.status) === 'completed').length,
    ),
  );
  const runningPassIndex = (() => {
    const reportedActiveIndex = parseInteger(passProgress?.activePassIndex);
    if (reportedActiveIndex !== null) {
      return reportedActiveIndex;
    }
    const foundIndex = passList.findIndex((pass) => normalizePassStatus(pass?.status) === 'running');
    return foundIndex >= 0 ? foundIndex : null;
  })();
  const resolvedActivePassIndex = passList.length
    ? Math.max(
        0,
        Math.min(
          passList.length - 1,
          activePassIndex >= 0 ? activePassIndex : runningPassIndex ?? (passList.length - 1),
        ),
      )
    : -1;
  const activePass = passList[resolvedActivePassIndex] || null;
  const progressValue = Number.isFinite(Number(passProgress?.progress))
    ? Math.max(0, Math.min(100, Number.parseFloat(passProgress.progress)))
    : null;
  const progressTone = progressToneClassName(completedPassCount, totalPassCount, runningPassIndex);
  const runningPass = runningPassIndex !== null ? passList[runningPassIndex] || null : null;
  const progressTitle = runningPass && runningPassIndex !== null
    ? `Pass ${runningPassIndex + 1} of ${totalPassCount} running`
    : totalPassCount > 0 && completedPassCount >= totalPassCount
      ? `All ${totalPassCount} passes completed`
      : totalPassCount > 0
        ? `${completedPassCount}/${totalPassCount} passes completed`
        : 'Pass schedule ready';
  const progressCopy = safeText(
    passProgress?.message,
    totalPassCount > 0
      ? `${completedPassCount}/${totalPassCount} passes completed.`
      : 'No pass progress is available yet.',
  );
  const draftGraph = useMemo(
    () => buildDraftGraph(relationGraph, objectCatalog),
    [relationGraph, objectCatalog],
  );
  const graphLayout = useMemo(() => buildRenderableGraphLayout(draftGraph), [draftGraph]);

  if (!passList.length) {
    return <div className="empty">No forward or diff-sim pass report is available yet for this run.</div>;
  }

  const containerName = String(activePass?.container_name || '');
  const optimizedIds = new Set(asList(activePass?.optimized_object_ids));
  const dynamicIds = new Set(asList(activePass?.dynamic_object_ids));
  const simulatedIds = new Set(asList(activePass?.simulated_object_ids));
  const activeObjectIds = new Set([
    ...optimizedIds,
    ...dynamicIds,
    ...simulatedIds,
    ...(containerName && containerName !== 'ground' ? [containerName] : []),
  ]);
  const highlightedEdges = buildHighlightedEdges(draftGraph.relations, activePass, activeObjectIds);

  return (
    <div className="simulation-pass-tree">
      <div className="simulation-pass-tree-toolbar">
        <div className="simulation-pass-tree-summary">
          <span className="meta-label">Layered physics schedule</span>
          <h3 className="card-title">Forward / diff-sim pass tree</h3>
          <div className="note-box">
            This mirrors the stage-03 scene tree and highlights the group considered in the selected
            physics pass.
          </div>
        </div>
        <div className="simulation-pass-tree-legend">
          <span className="simulation-pass-legend-chip simulation-pass-legend-chip-container" style={buildLegendChipStyle('container')}>Container</span>
          <span className="simulation-pass-legend-chip simulation-pass-legend-chip-active" style={buildLegendChipStyle('active')}>Optimized group</span>
          <span className="simulation-pass-legend-chip simulation-pass-legend-chip-context" style={buildLegendChipStyle('context')}>Forward-sim set</span>
        </div>
      </div>

      <div className="simulation-pass-progress">
        <div className="simulation-pass-progress-head">
          <div className="simulation-pass-progress-copy">
            <span className="meta-label">Pass progress</span>
            <div className="simulation-pass-progress-title">{progressTitle}</div>
            <div className="simulation-pass-progress-note">{progressCopy}</div>
          </div>
          <span className={`status-pill ${passStatusClassName(runningPassIndex !== null ? 'running' : completedPassCount >= totalPassCount ? 'completed' : 'pending')}`}>
            {runningPassIndex !== null ? 'Running' : completedPassCount >= totalPassCount ? 'Completed' : 'Queued'}
          </span>
        </div>
        {progressValue !== null ? (
          <div className="stage-progress-track" aria-label="Physics pass progress">
            <div className={`stage-progress-fill ${progressTone}`} style={{ width: `${Math.round(progressValue)}%` }} />
            <div className="stage-progress-label">{Math.round(progressValue)}%</div>
          </div>
        ) : null}
      </div>

      <div className="simulation-pass-selector" role="tablist" aria-label="Physics passes">
        {passList.map((pass, index) => {
          const status = resolvePassStatus(pass, index, completedPassCount, runningPassIndex);
          return (
            <button
              key={`${pass.pass_index ?? index}:${pass.strategy ?? 'pass'}`}
              type="button"
              role="tab"
              className={`simulation-pass-tab ${index === resolvedActivePassIndex ? 'simulation-pass-tab-active' : ''}`}
              aria-selected={index === resolvedActivePassIndex}
              onClick={() => onSelectPass(index)}
            >
              <div className="simulation-pass-tab-head">
                <span className="simulation-pass-tab-title">{formatPassTitle(pass, index)}</span>
                <span className={`status-pill simulation-pass-tab-status ${passStatusClassName(status)}`}>
                  {passStatusLabel(status)}
                </span>
              </div>
              <span className="simulation-pass-tab-copy">{formatPassMode(pass)}</span>
            </button>
          );
        })}
      </div>

      <div className="simulation-pass-meta-grid">
        <div className="simulation-pass-meta-card">
          <span className="meta-label">Current layer</span>
          <div className="simulation-pass-meta-value">{formatPassTitle(activePass, resolvedActivePassIndex)}</div>
          <div className="simulation-pass-meta-note">{formatPassMode(activePass)}</div>
        </div>
        <div className="simulation-pass-meta-card">
          <span className="meta-label">Container</span>
          <div className="simulation-pass-meta-value">{formatObjectDisplayName(containerName, 'ground')}</div>
          <div className="simulation-pass-meta-note">
            Optimized: {summarizeObjectIds(activePass?.optimized_object_ids)}
          </div>
        </div>
        <div className="simulation-pass-meta-card">
          <span className="meta-label">Forward simulation set</span>
          <div className="simulation-pass-meta-value">{safeText(asList(activePass?.simulated_object_ids).length, '0')} objects</div>
          <div className="simulation-pass-meta-note">
            Dynamic: {summarizeObjectIds(activePass?.dynamic_object_ids)}
          </div>
        </div>
      </div>

      <div className="card-heading simulation-pass-tree-frame-head">
        <div>
          <h4 className="card-title">Pass tree focus</h4>
          <p className="card-subtle">The selected pass highlights its container, diff-sim group, and forward-sim context directly on the scene tree.</p>
        </div>
        <span className="scroll-affordance">Scroll both directions</span>
      </div>
      <div className="relation-graph-frame simulation-pass-tree-frame scroll-surface scroll-surface-both">
        <svg
          viewBox={`0 0 ${graphLayout.width} ${graphLayout.height}`}
          role="img"
          aria-label="Physics pass scene tree"
        >
          <rect
            x="10"
            y="10"
            width={graphLayout.width - 20}
            height={graphLayout.height - 20}
            rx="22"
            fill={RELATION_GRAPH_THEME.canvasFill}
            stroke={RELATION_GRAPH_THEME.canvasStroke}
          />
          {graphLayout.edges.map((edge) => {
            const syntheticRoot = Boolean(edge.relation?.synthetic_root);
            const highlighted = syntheticRoot
              ? containerName === 'ground' && activeObjectIds.has(edge.relation.child_object_id)
              : highlightedEdges.has(relationKey(edge.relation));
            const visuals = buildPassEdgeStyle(edge.relation, highlighted);
            return (
              <g
                key={`simulation-edge-${edge.index}`}
                className={highlighted ? 'simulation-pass-edge-highlighted' : 'simulation-pass-edge'}
              >
                <path
                  d={edge.pathD}
                  className="relation-edge-line"
                  style={{ ...visuals.line, fill: 'none' }}
                />
                <text
                  x={edge.labelX}
                  y={edge.labelY}
                  textAnchor="middle"
                  className="relation-edge-label"
                  style={visuals.label}
                >
                  {syntheticRoot ? 'root' : safeText(edge.relation?.relation_type)}
                </text>
              </g>
            );
          })}

          {graphLayout.nodes.map((node) => {
            const isGround = Boolean(node.isSyntheticGround || isGroundNodeId(node.id));
            const isContainer = (containerName && node.id === containerName) || (isGround && containerName === 'ground');
            const isActive = optimizedIds.has(node.id);
            const isContext = simulatedIds.has(node.id) || dynamicIds.has(node.id);
            const visuals = buildPassNodeStyle({ isGround, isContainer, isActive, isContext: !isActive && isContext });
            const nodeClassName = [
              'simulation-pass-node',
              isContainer ? 'simulation-pass-node-container' : '',
              isActive ? 'simulation-pass-node-active' : '',
              !isActive && isContext ? 'simulation-pass-node-context' : '',
            ].filter(Boolean).join(' ');
            return (
              <g key={node.id} transform={`translate(${node.x}, ${node.y})`} className={nodeClassName}>
                <rect
                  x={-70}
                  y={-28}
                  width={140}
                  height={56}
                  rx={18}
                  className="relation-node-box"
                  style={visuals.box}
                />
                <text x="0" y={isGround ? '-6' : '-2'} textAnchor="middle" className="relation-node-label" style={visuals.label}>
                  {safeText(node.label, node.id)}
                </text>
                <text x="0" y="16" textAnchor="middle" className="relation-node-subtitle" style={visuals.subtitle}>
                  {isGround
                    ? (isContainer ? 'root container' : safeText(node.subtitle, 'scene root'))
                    : isContainer
                      ? 'container'
                      : isActive
                        ? 'diff sim'
                        : isContext
                          ? 'forward sim'
                          : safeText(node.subtitle, '')}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
