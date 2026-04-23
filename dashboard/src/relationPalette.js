export const GROUND_NODE_ID = '__relation_ground_root__';

const RELATION_TONES = {
  supports: {
    stroke: '#bd93f9',
    label: '#d8c2ff',
    soft: 'rgba(189, 147, 249, 0.1)',
    strongSoft: 'rgba(189, 147, 249, 0.16)',
    ring: 'rgba(189, 147, 249, 0.28)',
    glow: 'rgba(189, 147, 249, 0.24)',
  },
  contains: {
    stroke: '#ff79c6',
    label: '#ffc2e6',
    soft: 'rgba(255, 121, 198, 0.1)',
    strongSoft: 'rgba(255, 121, 198, 0.16)',
    ring: 'rgba(255, 121, 198, 0.28)',
    glow: 'rgba(255, 121, 198, 0.24)',
  },
  on: {
    stroke: '#ffb86c',
    label: '#ffd7ae',
    soft: 'rgba(255, 184, 108, 0.1)',
    strongSoft: 'rgba(255, 184, 108, 0.16)',
    ring: 'rgba(255, 184, 108, 0.28)',
    glow: 'rgba(255, 184, 108, 0.24)',
  },
  in: {
    stroke: '#8be9fd',
    label: '#c9f5ff',
    soft: 'rgba(139, 233, 253, 0.1)',
    strongSoft: 'rgba(139, 233, 253, 0.16)',
    ring: 'rgba(139, 233, 253, 0.28)',
    glow: 'rgba(139, 233, 253, 0.24)',
  },
  root: {
    stroke: '#f1fa8c',
    label: '#fbffd1',
    soft: 'rgba(241, 250, 140, 0.08)',
    strongSoft: 'rgba(241, 250, 140, 0.14)',
    ring: 'rgba(241, 250, 140, 0.22)',
    glow: 'rgba(241, 250, 140, 0.18)',
  },
};

export const RELATION_GRAPH_THEME = {
  canvasFill: '#11151c',
  canvasStroke: 'rgba(121, 142, 170, 0.26)',
  nodeFill: 'rgba(20, 25, 32, 0.94)',
  nodeFillRaised: 'rgba(27, 33, 41, 0.96)',
  nodeStroke: 'rgba(122, 144, 173, 0.34)',
  nodeLabel: '#eef3fb',
  nodeSubtitle: '#98a5b7',
  nodeShadow: 'rgba(4, 10, 18, 0.3)',
  nodeDragStroke: '#7cb8ff',
  portFill: 'rgba(92, 171, 255, 0.14)',
  portStroke: 'rgba(92, 171, 255, 0.68)',
  portText: '#7cb8ff',
  previewStroke: '#7cb8ff',
  previewFill: 'rgba(124, 184, 255, 0.18)',
  rootNodeFill: '#0f1319',
  rootNodeStroke: 'rgba(180, 194, 214, 0.36)',
  rootNodeSubtitle: 'rgba(208, 219, 233, 0.72)',
  rootNodeShadow: 'rgba(3, 8, 15, 0.36)',
  handleFill: 'rgba(92, 171, 255, 0.76)',
  handleStroke: 'rgba(219, 232, 248, 0.44)',
  handleActiveFill: 'rgba(124, 184, 255, 0.88)',
  handleActiveStroke: '#7cb8ff',
  handleHitFill: 'transparent',
  edgeHitStroke: 'transparent',
};

export const PASS_NODE_TONES = {
  base: {
    fill: RELATION_GRAPH_THEME.nodeFill,
    stroke: RELATION_GRAPH_THEME.nodeStroke,
    label: RELATION_GRAPH_THEME.nodeLabel,
    subtitle: RELATION_GRAPH_THEME.nodeSubtitle,
    shadow: RELATION_GRAPH_THEME.nodeShadow,
  },
  root: {
    fill: RELATION_GRAPH_THEME.rootNodeFill,
    stroke: RELATION_GRAPH_THEME.rootNodeStroke,
    label: RELATION_GRAPH_THEME.nodeLabel,
    subtitle: RELATION_GRAPH_THEME.rootNodeSubtitle,
    shadow: RELATION_GRAPH_THEME.rootNodeShadow,
  },
  container: {
    fill: RELATION_GRAPH_THEME.nodeFillRaised,
    stroke: 'rgba(180, 194, 214, 0.32)',
    label: RELATION_GRAPH_THEME.nodeLabel,
    subtitle: RELATION_GRAPH_THEME.nodeSubtitle,
    shadow: 'rgba(6, 12, 20, 0.32)',
  },
  active: {
    fill: RELATION_GRAPH_THEME.nodeFillRaised,
    stroke: 'rgba(172, 187, 208, 0.3)',
    label: RELATION_GRAPH_THEME.nodeLabel,
    subtitle: RELATION_GRAPH_THEME.nodeSubtitle,
    shadow: 'rgba(6, 12, 20, 0.32)',
  },
  context: {
    fill: RELATION_GRAPH_THEME.nodeFill,
    stroke: 'rgba(160, 176, 196, 0.28)',
    label: RELATION_GRAPH_THEME.nodeLabel,
    subtitle: RELATION_GRAPH_THEME.nodeSubtitle,
    shadow: 'rgba(5, 10, 18, 0.28)',
  },
};

function normalizeRelationType(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'contains' || normalized === 'on' || normalized === 'in') {
    return normalized;
  }
  return 'supports';
}

export function getRelationTone(value, options = {}) {
  if (options.syntheticRoot) {
    return RELATION_TONES.root;
  }
  return RELATION_TONES[normalizeRelationType(value)] || RELATION_TONES.supports;
}

export function isGroundNodeId(value) {
  return String(value || '') === GROUND_NODE_ID;
}
