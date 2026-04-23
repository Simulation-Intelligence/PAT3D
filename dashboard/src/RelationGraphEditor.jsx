import { useEffect, useMemo, useRef, useState } from 'react';
import { formatObjectDisplayName } from './runtimeViewModel';
import {
  GROUND_NODE_ID,
  RELATION_GRAPH_THEME,
  getRelationTone,
  isGroundNodeId,
} from './relationPalette';

const RELATION_TYPE_OPTIONS = ['supports', 'contains', 'on', 'in'];
const NODE_WIDTH = 140;
const NODE_HEIGHT = 56;
const NODE_CONNECT_OFFSET = (NODE_WIDTH / 2) + 10;
const DRAG_HANDLE_RADIUS = 8;
const DRAG_HANDLE_HIT_RADIUS = 16;
const EDGE_HIT_STROKE_WIDTH = 18;
const NODE_HIT_PADDING = 10;
const EDGE_ENDPOINT_DROP_TOLERANCE = 14;
const NODE_CENTER_Y = NODE_HEIGHT / 2;
const EDGE_ROUTE_MARGIN = 30;
const EDGE_ROUTE_INSET_MIN = 44;
const EDGE_ROUTE_INSET_MAX = 72;

function asList(value) {
  return Array.isArray(value) ? value : [];
}

function safeText(value, fallback = '') {
  if (value === null || value === undefined || value === '') return fallback;
  return String(value);
}

function normalizeCount(value) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  return Number.isFinite(parsed) && parsed > 1 ? parsed : 1;
}

function dedupePreservingOrder(values) {
  const ordered = [];
  const seen = new Set();
  values.forEach((value) => {
    const normalized = safeText(value);
    if (!normalized || seen.has(normalized)) return;
    seen.add(normalized);
    ordered.push(normalized);
  });
  return ordered;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function distance2(ax, ay, bx, by) {
  const dx = ax - bx;
  const dy = ay - by;
  return (dx * dx) + (dy * dy);
}

function clampEdgeRouteY(value, graphHeight) {
  return clamp(value, 34, Math.max(34, graphHeight - 34));
}

function nodeBounds(node) {
  return {
    left: node.x - ((NODE_WIDTH * 0.5) + 8),
    right: node.x + ((NODE_WIDTH * 0.5) + 8),
    top: node.y - ((NODE_HEIGHT * 0.5) + 6),
    bottom: node.y + ((NODE_HEIGHT * 0.5) + 6),
  };
}

function pointInRect(x, y, rect) {
  return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
}

function orientation(ax, ay, bx, by, cx, cy) {
  const value = ((by - ay) * (cx - bx)) - ((bx - ax) * (cy - by));
  if (Math.abs(value) < 1e-9) return 0;
  return value > 0 ? 1 : 2;
}

function onSegment(ax, ay, bx, by, cx, cy) {
  return (
    bx <= Math.max(ax, cx) + 1e-9
    && bx + 1e-9 >= Math.min(ax, cx)
    && by <= Math.max(ay, cy) + 1e-9
    && by + 1e-9 >= Math.min(ay, cy)
  );
}

function segmentsIntersect(ax, ay, bx, by, cx, cy, dx, dy) {
  const o1 = orientation(ax, ay, bx, by, cx, cy);
  const o2 = orientation(ax, ay, bx, by, dx, dy);
  const o3 = orientation(cx, cy, dx, dy, ax, ay);
  const o4 = orientation(cx, cy, dx, dy, bx, by);

  if (o1 !== o2 && o3 !== o4) return true;
  if (o1 === 0 && onSegment(ax, ay, cx, cy, bx, by)) return true;
  if (o2 === 0 && onSegment(ax, ay, dx, dy, bx, by)) return true;
  if (o3 === 0 && onSegment(cx, cy, ax, ay, dx, dy)) return true;
  if (o4 === 0 && onSegment(cx, cy, bx, by, dx, dy)) return true;
  return false;
}

function segmentIntersectsRect(ax, ay, bx, by, rect) {
  if (pointInRect(ax, ay, rect) || pointInRect(bx, by, rect)) {
    return true;
  }
  return (
    segmentsIntersect(ax, ay, bx, by, rect.left, rect.top, rect.right, rect.top)
    || segmentsIntersect(ax, ay, bx, by, rect.right, rect.top, rect.right, rect.bottom)
    || segmentsIntersect(ax, ay, bx, by, rect.right, rect.bottom, rect.left, rect.bottom)
    || segmentsIntersect(ax, ay, bx, by, rect.left, rect.bottom, rect.left, rect.top)
  );
}

function buildEdgeGeometry({ sourceNode, targetNode, relation, index, nodes, graphHeight }) {
  const x1 = sourceNode.x + NODE_CONNECT_OFFSET;
  const y1 = sourceNode.y;
  const x2 = targetNode.x - NODE_CONNECT_OFFSET;
  const y2 = targetNode.y;
  const straightPath = `M ${x1} ${y1} L ${x2} ${y2}`;
  const midpointY = (y1 + y2) * 0.5;
  const midpointX = (x1 + x2) * 0.5;
  const obstructingNodes = nodes
    .filter((node) => node.id !== relation.parent_object_id && node.id !== relation.child_object_id)
    .filter((node) => segmentIntersectsRect(x1, y1, x2, y2, nodeBounds(node)));

  if (!obstructingNodes.length) {
    return {
      index,
      relation,
      x1,
      y1,
      x2,
      y2,
      labelX: midpointX,
      labelY: midpointY - 10,
      pathD: straightPath,
      routed: false,
    };
  }

  const highestObstacleTop = Math.min(...obstructingNodes.map((node) => nodeBounds(node).top));
  const lowestObstacleBottom = Math.max(...obstructingNodes.map((node) => nodeBounds(node).bottom));
  const aboveLaneY = clampEdgeRouteY(highestObstacleTop - EDGE_ROUTE_MARGIN, graphHeight);
  const belowLaneY = clampEdgeRouteY(lowestObstacleBottom + EDGE_ROUTE_MARGIN, graphHeight);
  const routeY = Math.abs(aboveLaneY - midpointY) <= Math.abs(belowLaneY - midpointY)
    ? aboveLaneY
    : belowLaneY;
  const rawInset = clamp(Math.abs(x2 - x1) * 0.22, EDGE_ROUTE_INSET_MIN, EDGE_ROUTE_INSET_MAX);
  let startX = x1 + rawInset;
  let endX = x2 - rawInset;
  if (endX - startX < 28) {
    const centerX = midpointX;
    startX = centerX - 14;
    endX = centerX + 14;
  }

  return {
    index,
    relation,
    x1,
    y1,
    x2,
    y2,
    labelX: (startX + endX) * 0.5,
    labelY: routeY < midpointY ? routeY - 10 : routeY + 18,
    pathD: `M ${x1} ${y1} L ${startX} ${y1} L ${startX} ${routeY} L ${endX} ${routeY} L ${endX} ${y2} L ${x2} ${y2}`,
    routed: true,
  };
}

function findPortAtPoint(nodes, point, allowedKinds = ['in', 'out']) {
  if (!point) return null;
  const touchRadius = EDGE_ENDPOINT_DROP_TOLERANCE;
  const maxDistance = touchRadius * touchRadius;
  let closest = null;
  for (const node of nodes) {
    if (node.isSyntheticGround) continue;
    const portTypes = allowedKinds.includes('in')
      ? [{ kind: 'in', x: node.x - NODE_CONNECT_OFFSET, y: node.y }]
      : [];
    if (allowedKinds.includes('out')) {
      portTypes.push({ kind: 'out', x: node.x + NODE_CONNECT_OFFSET, y: node.y });
    }
    for (const port of portTypes) {
      const candidate = distance2(port.x, port.y, point.x, point.y);
      if (candidate <= maxDistance && (!closest || candidate < closest.distance)) {
        closest = {
          nodeId: node.id,
          kind: port.kind,
          x: port.x,
          y: port.y,
          distance: candidate,
        };
      }
    }
  }
  return closest;
}

function hasDuplicateRelation(relations, parentObjectId, relationType, childObjectId, ignoreIndex = null) {
  return asList(relations).some((relation, index) => (
    index !== ignoreIndex
    && relation.parent_object_id === parentObjectId
    && relation.child_object_id === childObjectId
    && relation.relation_type === relationType
  ));
}

function resolveInstanceIds(object, reservedIds = new Set()) {
  const count = normalizeCount(object?.count);
  const objectId = safeText(object?.object_id, 'object');
  if (count <= 1) {
    return [objectId];
  }
  const resolvedIds = [];
  const sourceIds = asList(object?.source_instance_ids)
    .map((value) => safeText(value))
    .filter(Boolean);

  for (let index = 0; index < count; index += 1) {
    let candidate = sourceIds[index] || (count === 1 ? objectId : `${objectId}::${index + 1}`);
    if (reservedIds.has(candidate) || resolvedIds.includes(candidate)) {
      candidate = `${objectId}::${index + 1}`;
    }
    let suffix = 1;
    let uniqueCandidate = candidate;
    while (reservedIds.has(uniqueCandidate) || resolvedIds.includes(uniqueCandidate)) {
      suffix += 1;
      uniqueCandidate = `${candidate}:${suffix}`;
    }
    resolvedIds.push(uniqueCandidate);
  }

  return resolvedIds;
}

function buildObjectEntries(objectCatalog, graph) {
  const entries = [];
  const seen = new Set();
  const reservedIds = new Set();
  const idsByObjectId = new Map();

  for (const object of asList(objectCatalog?.objects)) {
    const objectId = safeText(object?.object_id);
    if (!objectId) continue;
    const baseLabel = formatObjectDisplayName(object?.display_name || object?.canonical_name || objectId, objectId);
    const canonicalSubtitle = object?.canonical_name && object?.display_name && object.display_name !== object.canonical_name
      ? formatObjectDisplayName(object.canonical_name)
      : '';
    const instanceIds = resolveInstanceIds(object, reservedIds);
    idsByObjectId.set(objectId, instanceIds);
    instanceIds.forEach((instanceId, index) => {
      if (!instanceId || seen.has(instanceId)) return;
      seen.add(instanceId);
      reservedIds.add(instanceId);
      const label = instanceIds.length > 1
        ? formatObjectDisplayName(instanceId, `${baseLabel}${index + 1}`)
        : baseLabel;
      const subtitleParts = [];
      if (canonicalSubtitle) subtitleParts.push(canonicalSubtitle);
      if (instanceId !== objectId) subtitleParts.push(formatObjectDisplayName(instanceId));
      entries.push({
        id: instanceId,
        label,
        subtitle: subtitleParts.join(' · '),
        isExpandedInstance: instanceIds.length > 1,
      });
    });
  }

  for (const relation of asList(graph?.relations)) {
    for (const objectId of [relation?.parent_object_id, relation?.child_object_id]) {
      const normalizedId = safeText(objectId);
      if (!normalizedId || seen.has(normalizedId) || idsByObjectId.has(normalizedId)) continue;
      seen.add(normalizedId);
      entries.push({ id: normalizedId, label: formatObjectDisplayName(normalizedId), subtitle: '' });
    }
  }

  for (const rootObjectId of asList(graph?.root_object_ids)) {
    const normalizedId = safeText(rootObjectId);
    if (!normalizedId || seen.has(normalizedId) || idsByObjectId.has(normalizedId)) continue;
    seen.add(normalizedId);
    entries.push({ id: normalizedId, label: formatObjectDisplayName(normalizedId), subtitle: '' });
  }

  return { entries, idsByObjectId };
}

function expandObjectIds(objectId, catalogIndex) {
  const normalizedId = safeText(objectId);
  if (!normalizedId) return [];
  return catalogIndex.idsByObjectId.get(normalizedId) || [normalizedId];
}

function expandGraphRelations(graph, catalogIndex) {
  const expandedRelations = [];
  const seenRelationKeys = new Set();

  asList(graph?.relations).forEach((relation) => {
    const normalizedType = RELATION_TYPE_OPTIONS.includes(String(relation?.relation_type || '').toLowerCase())
      ? String(relation.relation_type).toLowerCase()
      : 'supports';
    const parentIds = expandObjectIds(relation?.parent_object_id, catalogIndex);
    const childIds = expandObjectIds(relation?.child_object_id, catalogIndex);
    let pairs = [];
    if (parentIds.length > 1 && childIds.length > 1 && parentIds.length === childIds.length) {
      pairs = parentIds.map((parentId, index) => [parentId, childIds[index]]);
    } else if (parentIds.length === 1 && childIds.length === 1) {
      pairs = [[parentIds[0], childIds[0]]];
    } else if (parentIds.length > 0 && childIds.length > 0) {
      // Ambiguous canonical-to-instance mappings should stay conservative in the editor.
      // Users can then tune the direct edge set explicitly.
      pairs = [[parentIds[0], childIds[0]]];
    }

    pairs.forEach(([parentId, childId]) => {
      if (!parentId || !childId || parentId === childId) return;
      const relationKey = [
        parentId,
        childId,
        normalizedType,
        relation?.confidence ?? null,
        safeText(relation?.evidence),
      ].join('::');
      if (seenRelationKeys.has(relationKey)) return;
      seenRelationKeys.add(relationKey);
      expandedRelations.push({
        parent_object_id: parentId,
        relation_type: normalizedType,
        child_object_id: childId,
        confidence: relation?.confidence ?? null,
        evidence: safeText(relation?.evidence),
      });
    });
  });

  return expandedRelations;
}

function inferRepeatedInstanceChainRelations(objectCatalog, catalogIndex, existingRelations) {
  const seenPairs = new Set(existingRelations.map((relation) => `${relation.parent_object_id}->${relation.child_object_id}`));
  const inferredRelations = [];

  for (const object of asList(objectCatalog?.objects)) {
    const instanceIds = catalogIndex.idsByObjectId.get(safeText(object?.object_id)) || [];
    if (instanceIds.length <= 1) continue;
    const hasExistingParent = existingRelations.some((relation) => (
      instanceIds.includes(relation.child_object_id)
    ));
    if (hasExistingParent) continue;
    const hasInternalRelation = existingRelations.some((relation) => (
      instanceIds.includes(relation.parent_object_id) && instanceIds.includes(relation.child_object_id)
    ));
    if (hasInternalRelation) continue;

    for (let index = 0; index < instanceIds.length - 1; index += 1) {
      const parentId = instanceIds[index];
      const childId = instanceIds[index + 1];
      const pairKey = `${parentId}->${childId}`;
      if (seenPairs.has(pairKey)) continue;
      seenPairs.add(pairKey);
      inferredRelations.push({
        parent_object_id: parentId,
        relation_type: 'supports',
        child_object_id: childId,
        confidence: null,
        evidence: 'editor_inferred_direct_instance_chain',
      });
    }
  }

  return inferredRelations;
}

export function buildDraftGraph(graph, objectCatalog) {
  const catalogIndex = buildObjectEntries(objectCatalog, graph);
  const objectEntries = catalogIndex.entries;
  const fallbackSceneId = safeText(graph?.scene_id || objectCatalog?.scene_id || 'scene');
  const directRelations = expandGraphRelations(graph, catalogIndex);
  const relations = [
    ...directRelations,
    ...inferRepeatedInstanceChainRelations(objectCatalog, catalogIndex, directRelations),
  ];
  const childIds = new Set(relations.map((relation) => relation.child_object_id));
  const explicitRoots = dedupePreservingOrder(
    asList(graph?.root_object_ids).flatMap((value) => expandObjectIds(value, catalogIndex))
  );
  const rootObjectIds = dedupePreservingOrder(
    [...explicitRoots, ...objectEntries.map((entry) => entry.id)]
  ).filter((objectId) => !childIds.has(objectId));
  return {
    scene_id: fallbackSceneId,
    root_object_ids: rootObjectIds,
    relations,
    object_entries: objectEntries,
  };
}

function validateDraftGraph(draftGraph) {
  const issues = [];
  const validObjectIds = new Set(asList(draftGraph?.object_entries).map((entry) => entry.id));
  const edges = new Map();
  const allNodes = new Set(asList(draftGraph?.root_object_ids));
  const seenEdges = new Set();

  asList(draftGraph?.relations).forEach((relation, index) => {
    const relationLabel = `Relation ${index + 1}`;
    if (!relation.parent_object_id || !relation.child_object_id) {
      issues.push(`${relationLabel} is missing a parent or child object.`);
      return;
    }
    if (relation.parent_object_id === relation.child_object_id) {
      issues.push(`${relationLabel} cannot point to the same object.`);
    }
    if (!RELATION_TYPE_OPTIONS.includes(relation.relation_type)) {
      issues.push(`${relationLabel} uses an unsupported relation type.`);
    }
    if (validObjectIds.size && !validObjectIds.has(relation.parent_object_id)) {
      issues.push(`${relationLabel} references unknown parent "${relation.parent_object_id}".`);
    }
    if (validObjectIds.size && !validObjectIds.has(relation.child_object_id)) {
      issues.push(`${relationLabel} references unknown child "${relation.child_object_id}".`);
    }
    const edgeKey = `${relation.parent_object_id}:${relation.relation_type}:${relation.child_object_id}`;
    if (seenEdges.has(edgeKey)) {
      issues.push(`${relationLabel} duplicates an existing edge.`);
    }
    seenEdges.add(edgeKey);
    allNodes.add(relation.parent_object_id);
    allNodes.add(relation.child_object_id);
    const children = edges.get(relation.parent_object_id) || [];
    children.push(relation.child_object_id);
    edges.set(relation.parent_object_id, children);
  });

  const visiting = new Set();
  const visited = new Set();
  let hasCycle = false;
  function visit(node) {
    if (visited.has(node) || hasCycle) return;
    if (visiting.has(node)) {
      hasCycle = true;
      return;
    }
    visiting.add(node);
    for (const child of edges.get(node) || []) {
      visit(child);
    }
    visiting.delete(node);
    visited.add(node);
  }

  for (const node of allNodes) {
    visit(node);
  }
  if (hasCycle) {
    issues.push('The directed relation graph must remain acyclic.');
  }

  return issues;
}

export function buildGraphLayout(draftGraph, positionOverrides = {}) {
  const objectEntries = asList(draftGraph?.object_entries);
  const labelById = new Map(objectEntries.map((entry) => [entry.id, entry]));
  const nodeIds = objectEntries.map((entry) => entry.id);
  const relations = asList(draftGraph?.relations);
  const indegree = new Map(nodeIds.map((id) => [id, 0]));
  const children = new Map();
  const parentMap = new Map();

  for (const relation of relations) {
    if (!indegree.has(relation.parent_object_id)) indegree.set(relation.parent_object_id, 0);
    if (!indegree.has(relation.child_object_id)) indegree.set(relation.child_object_id, 0);
    indegree.set(relation.child_object_id, (indegree.get(relation.child_object_id) || 0) + 1);
    const bucket = children.get(relation.parent_object_id) || [];
    bucket.push(relation.child_object_id);
    children.set(relation.parent_object_id, bucket);
    parentMap.set(relation.child_object_id, relation.parent_object_id);
  }

  const preferredRoots = asList(draftGraph?.root_object_ids).filter((id) => indegree.has(id));
  const queue = [];
  const seenRoots = new Set();
  for (const rootId of preferredRoots) {
    if (seenRoots.has(rootId)) continue;
    seenRoots.add(rootId);
    queue.push(rootId);
  }
  for (const [nodeId, count] of indegree.entries()) {
    if (count === 0 && !seenRoots.has(nodeId)) {
      queue.push(nodeId);
      seenRoots.add(nodeId);
    }
  }

  const levels = new Map();
  for (const rootId of queue) {
    levels.set(rootId, 0);
  }
  while (queue.length) {
    const nodeId = queue.shift();
    const level = levels.get(nodeId) || 0;
    for (const childId of children.get(nodeId) || []) {
      const nextLevel = Math.max(level + 1, levels.get(childId) || 0);
      levels.set(childId, nextLevel);
      queue.push(childId);
    }
  }
  for (const nodeId of indegree.keys()) {
    if (!levels.has(nodeId)) {
      levels.set(nodeId, parentMap.has(nodeId) ? (levels.get(parentMap.get(nodeId)) || 0) + 1 : 0);
    }
  }

  const levelsInOrder = [...new Set([...levels.values()].sort((left, right) => left - right))];
  const nodesByLevel = new Map(levelsInOrder.map((level) => [level, []]));
  for (const nodeId of indegree.keys()) {
    nodesByLevel.get(levels.get(nodeId) || 0).push(nodeId);
  }

  const width = 1120;
  const levelCount = Math.max(1, levelsInOrder.length);
  const columnWidth = levelCount === 1 ? 0 : 820 / (levelCount - 1);
  const positions = new Map();
  let maxColumnHeight = 0;
  levelsInOrder.forEach((level, levelIndex) => {
    const ids = nodesByLevel.get(level) || [];
    maxColumnHeight = Math.max(maxColumnHeight, ids.length);
    const spacing = ids.length <= 1 ? 0 : Math.min(120, 420 / (ids.length - 1));
    const columnHeight = ids.length <= 1 ? 0 : spacing * (ids.length - 1);
    const startY = 110 + Math.max(0, (420 - columnHeight) / 2);
    ids.forEach((nodeId, rowIndex) => {
      positions.set(nodeId, {
        x: 150 + (levelIndex * columnWidth),
        y: startY + (rowIndex * spacing),
      });
    });
  });

  const renderedNodes = [...indegree.keys()].map((nodeId) => {
    const override = positionOverrides?.[nodeId];
    const position = override || positions.get(nodeId) || { x: 100, y: 140 };
    const entry = labelById.get(nodeId) || { id: nodeId, label: nodeId, subtitle: '' };
    return {
      ...entry,
      x: position.x,
      y: position.y,
    };
  });
  const renderedNodePositions = new Map(renderedNodes.map((node) => [node.id, { x: node.x, y: node.y }]));

  const graphHeight = Math.max(520, 220 + maxColumnHeight * 104);
  const renderedEdges = relations.map((relation, index) => {
    const source = renderedNodes.find((node) => node.id === relation.parent_object_id) || { id: relation.parent_object_id, x: 100, y: 100 };
    const target = renderedNodes.find((node) => node.id === relation.child_object_id) || { id: relation.child_object_id, x: 260, y: 180 };
    return buildEdgeGeometry({
      sourceNode: source,
      targetNode: target,
      relation,
      index,
      nodes: renderedNodes,
      graphHeight,
    });
  });

  return {
    width,
    height: graphHeight,
    nodes: renderedNodes,
    edges: renderedEdges,
  };
}

export function buildRenderableGraphLayout(draftGraph, positionOverrides = {}) {
  const rootObjectIds = dedupePreservingOrder(asList(draftGraph?.root_object_ids)).filter(Boolean);
  if (!rootObjectIds.length) {
    return buildGraphLayout(draftGraph, positionOverrides);
  }

  const objectEntries = asList(draftGraph?.object_entries);
  const hasGroundNode = objectEntries.some((entry) => isGroundNodeId(entry.id));
  const renderGraph = {
    ...draftGraph,
    object_entries: hasGroundNode
      ? objectEntries
      : [
        {
          id: GROUND_NODE_ID,
          label: 'Ground',
          subtitle: 'scene root',
          isSyntheticGround: true,
        },
        ...objectEntries,
      ],
    root_object_ids: [GROUND_NODE_ID],
    relations: [
      ...asList(draftGraph?.relations),
      ...rootObjectIds.map((childObjectId) => ({
        parent_object_id: GROUND_NODE_ID,
        relation_type: 'supports',
        child_object_id: childObjectId,
        confidence: null,
        evidence: 'synthetic_ground_root',
        synthetic_root: true,
      })),
    ],
  };

  return buildGraphLayout(renderGraph, positionOverrides);
}

function buildEdgeVisualStyle(relation, { selected = false } = {}) {
  const syntheticRoot = Boolean(relation?.synthetic_root);
  const tone = getRelationTone(relation?.relation_type, { syntheticRoot });
  return {
    tone,
    line: {
      stroke: selected ? tone.label : tone.stroke,
      strokeWidth: syntheticRoot ? 2.6 : (selected ? 4.2 : 3.1),
      opacity: syntheticRoot ? 0.9 : 1,
      strokeDasharray: syntheticRoot ? '8 10' : undefined,
      filter: selected ? `drop-shadow(0 0 10px ${tone.glow})` : undefined,
    },
    label: {
      fill: tone.label,
      fontWeight: syntheticRoot || selected ? 700 : 600,
      letterSpacing: '0.08em',
      textTransform: 'uppercase',
      paintOrder: 'stroke',
      stroke: RELATION_GRAPH_THEME.canvasFill,
      strokeWidth: 4,
      strokeLinecap: 'round',
      strokeLinejoin: 'round',
    },
    handle: {
      fill: selected ? RELATION_GRAPH_THEME.handleActiveFill : RELATION_GRAPH_THEME.handleFill,
      stroke: selected ? RELATION_GRAPH_THEME.handleActiveStroke : RELATION_GRAPH_THEME.handleStroke,
      strokeWidth: 2,
    },
  };
}

function markerIdForStrokeColor(stroke) {
  const normalized = String(stroke || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return `relation-arrow-${normalized || 'default'}`;
}

function buildNodeVisualStyle(node, { dragging = false } = {}) {
  const syntheticGround = Boolean(node?.isSyntheticGround);
  return {
    box: {
      fill: syntheticGround ? RELATION_GRAPH_THEME.rootNodeFill : (dragging ? RELATION_GRAPH_THEME.nodeFillRaised : RELATION_GRAPH_THEME.nodeFill),
      stroke: syntheticGround
        ? RELATION_GRAPH_THEME.rootNodeStroke
        : (dragging ? RELATION_GRAPH_THEME.nodeDragStroke : RELATION_GRAPH_THEME.nodeStroke),
      strokeWidth: syntheticGround ? 2.1 : (dragging ? 2 : 1.5),
      filter: `drop-shadow(0 18px 28px ${syntheticGround ? RELATION_GRAPH_THEME.rootNodeShadow : RELATION_GRAPH_THEME.nodeShadow})`,
      cursor: syntheticGround ? 'default' : (dragging ? 'grabbing' : 'grab'),
    },
    label: {
      fill: RELATION_GRAPH_THEME.nodeLabel,
      letterSpacing: syntheticGround ? '0.06em' : '0.01em',
    },
    subtitle: {
      fill: syntheticGround ? RELATION_GRAPH_THEME.rootNodeSubtitle : RELATION_GRAPH_THEME.nodeSubtitle,
      letterSpacing: '0.06em',
      textTransform: 'uppercase',
    },
    portCircle: {
      fill: RELATION_GRAPH_THEME.portFill,
      stroke: RELATION_GRAPH_THEME.portStroke,
      strokeWidth: 2,
    },
    portText: {
      fill: RELATION_GRAPH_THEME.portText,
    },
  };
}

function buildEdgeRowStyle(_relation, selected) {
  return {
    '--edge-tone': selected ? RELATION_GRAPH_THEME.nodeDragStroke : 'rgba(168, 181, 199, 0.28)',
    '--edge-tone-soft': selected ? RELATION_GRAPH_THEME.previewFill : 'rgba(255, 255, 255, 0.03)',
    borderColor: selected ? 'rgba(124, 184, 255, 0.26)' : 'rgba(121, 142, 170, 0.18)',
    background: selected
      ? 'linear-gradient(135deg, rgba(124, 184, 255, 0.08), rgba(255, 255, 255, 0.02)), rgba(21, 26, 33, 0.92)'
      : 'linear-gradient(135deg, rgba(255, 255, 255, 0.03), rgba(255, 255, 255, 0.02)), rgba(21, 26, 33, 0.86)',
    boxShadow: selected
      ? '0 0 0 1px rgba(124, 184, 255, 0.12), 0 18px 36px rgba(6, 15, 28, 0.22)'
      : 'inset 0 1px 0 rgba(255, 255, 255, 0.04)',
  };
}

function buildRelationTypeBadgeStyle(relation) {
  const tone = getRelationTone(relation?.relation_type);
  return {
    borderColor: 'rgba(121, 142, 170, 0.18)',
    background: 'rgba(255, 255, 255, 0.03)',
    color: tone.label,
  };
}

function buildEdgeChipStyle() {
  return {
    borderColor: 'rgba(124, 184, 255, 0.18)',
    background: 'rgba(124, 184, 255, 0.08)',
    color: '#a7cbf5',
  };
}

function ObjectDescriptionTable({ objectDescriptions }) {
  if (!asList(objectDescriptions).length) {
    return <div className="empty">No object descriptions were recorded for this stage.</div>;
  }

  return (
    <table className="data-table" style={{ tableLayout: 'fixed' }}>
      <colgroup>
        <col style={{ width: '28%' }} />
        <col style={{ width: '72%' }} />
      </colgroup>
      <thead>
        <tr>
          <th>Object</th>
          <th>Prompt</th>
        </tr>
      </thead>
      <tbody>
        {asList(objectDescriptions).map((description) => (
          <tr key={description.object_id}>
            <td style={{ verticalAlign: 'top' }}>{formatObjectDisplayName(description.object_id, 'n/a')}</td>
            <td style={{ lineHeight: 1.6 }}>{safeText(description.prompt_text, 'n/a')}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function RelationGraphEditor({
  graph,
  objectCatalog,
  objectDescriptions,
  editable = false,
  sourceKey = '',
  saveBusy = false,
  saveError = null,
  hasSavedOverride = false,
  relationRetryStage = null,
  onSave = null,
  onClearSavedOverride = null,
  onSaveAndRetry = null,
}) {
  const [draftGraph, setDraftGraph] = useState(() => buildDraftGraph(graph, objectCatalog));
  const [draftDirty, setDraftDirty] = useState(false);
  const [selectedEdgeIndex, setSelectedEdgeIndex] = useState(0);
  const [nodePositions, setNodePositions] = useState({});
  const [draggingNodeId, setDraggingNodeId] = useState('');
  const [activeConnection, setActiveConnection] = useState(null);
  const svgRef = useRef(null);
  const draggingNodeIdRef = useRef('');
  const activeConnectionRef = useRef(null);

  useEffect(() => {
    const nextGraph = buildDraftGraph(graph, objectCatalog);
    const nextLayout = buildRenderableGraphLayout(nextGraph);
    setDraftGraph(nextGraph);
    setNodePositions(
      Object.fromEntries(nextLayout.nodes.map((node) => [node.id, { x: node.x, y: node.y }]))
    );
    setDraftDirty(false);
    setSelectedEdgeIndex(0);
  }, [sourceKey]);

  const validationIssues = useMemo(() => validateDraftGraph(draftGraph), [draftGraph]);
  useEffect(() => {
    const seededLayout = buildRenderableGraphLayout(draftGraph, nodePositions);
    setNodePositions((current) => {
      const next = { ...current };
      let changed = false;
      seededLayout.nodes.forEach((node) => {
        if (!next[node.id]) {
          next[node.id] = { x: node.x, y: node.y };
          changed = true;
        }
      });
      Object.keys(next).forEach((nodeId) => {
        if (!seededLayout.nodes.find((node) => node.id === nodeId)) {
          delete next[nodeId];
          changed = true;
        }
      });
      return changed ? next : current;
    });
  }, [draftGraph]);
  const graphLayout = useMemo(
    () => buildRenderableGraphLayout(draftGraph, nodePositions),
    [draftGraph, nodePositions],
  );
  const edgeMarkers = useMemo(() => {
    const markers = new Map();
    graphLayout.edges.forEach((edge) => {
      const syntheticRoot = Boolean(edge.relation?.synthetic_root);
      const selected = !syntheticRoot && selectedEdgeIndex === edge.index;
      const stroke = buildEdgeVisualStyle(edge.relation, { selected }).line.stroke;
      const markerId = markerIdForStrokeColor(stroke);
      if (!markers.has(markerId)) {
        markers.set(markerId, stroke);
      }
    });
    return Array.from(markers.entries()).map(([id, fill]) => ({ id, fill }));
  }, [graphLayout.edges, selectedEdgeIndex]);

  const graphNodesById = useMemo(() => {
    const nodeById = new Map();
    graphLayout.nodes.forEach((node) => {
      nodeById.set(node.id, node);
    });
    return nodeById;
  }, [graphLayout.nodes]);

  const activePort = useMemo(() => {
    if (!activeConnection?.pointer) return null;
    const kinds = activeConnection.mode === 'rebind-source' ? ['out'] : ['in'];
    return findPortAtPoint(graphLayout.nodes, activeConnection.pointer, kinds);
  }, [activeConnection, graphLayout.nodes]);

  const objectEntries = asList(draftGraph?.object_entries);
  const objectEntryById = useMemo(
    () => new Map(objectEntries.map((entry) => [entry.id, entry])),
    [objectEntries],
  );
  const canEdit = editable && objectEntries.length > 0;
  const saveDisabled = !canEdit || saveBusy || validationIssues.length > 0;
  const repeatedInstanceCount = objectEntries.filter((entry) => entry.isExpandedInstance).length;

  useEffect(() => {
    draggingNodeIdRef.current = draggingNodeId;
  }, [draggingNodeId]);

  useEffect(() => {
    activeConnectionRef.current = activeConnection;
  }, [activeConnection]);

  function objectLabel(objectId) {
    return objectEntryById.get(objectId)?.label || safeText(objectId, 'Unassigned');
  }

  function updateDraft(mutator) {
    setDraftGraph((current) => {
      const nextGraph = mutator(current);
      return {
        ...nextGraph,
        object_entries: current.object_entries,
      };
    });
    setDraftDirty(true);
  }

  function addEdge() {
    if (objectEntries.length < 2) return;
    const parent = objectEntries[0]?.id || '';
    const child = objectEntries.find((entry) => entry.id !== parent)?.id || '';
    updateDraft((current) => ({
      ...current,
      relations: [
        ...asList(current.relations),
        {
          parent_object_id: parent,
          relation_type: 'supports',
          child_object_id: child,
          confidence: null,
          evidence: '',
        },
      ],
    }));
    setSelectedEdgeIndex(asList(draftGraph?.relations).length);
  }

  function updateEdge(index, patch) {
    updateDraft((current) => ({
      ...current,
      relations: asList(current.relations).map((relation, relationIndex) => (
        relationIndex === index
          ? { ...relation, ...patch }
          : relation
      )),
    }));
  }

  function deleteEdge(index) {
    updateDraft((current) => ({
      ...current,
      relations: asList(current.relations).filter((_, relationIndex) => relationIndex !== index),
    }));
    setSelectedEdgeIndex((current) => Math.max(0, Math.min(current, asList(draftGraph?.relations).length - 2)));
  }

  function reverseEdge(index) {
    const relation = asList(draftGraph?.relations)[index];
    if (!relation) return;
    updateEdge(index, {
      parent_object_id: relation.child_object_id,
      child_object_id: relation.parent_object_id,
    });
  }

  async function handleSave() {
    if (!onSave) return;
    await onSave({
      scene_id: draftGraph.scene_id,
      root_object_ids: asList(draftGraph.root_object_ids),
      relations: asList(draftGraph.relations),
    });
    setDraftDirty(false);
  }

  async function handleSaveAndRetry() {
    if (!onSaveAndRetry || !relationRetryStage) return;
    await onSaveAndRetry({
      scene_id: draftGraph.scene_id,
      root_object_ids: asList(draftGraph.root_object_ids),
      relations: asList(draftGraph.relations),
    });
    setDraftDirty(false);
  }

  function eventPointInSvg(event) {
    const svg = svgRef.current;
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    const x = ((event.clientX - rect.left) / rect.width) * viewBox.width;
    const y = ((event.clientY - rect.top) / rect.height) * viewBox.height;
    return { x, y };
  }

  function startNodeDrag(nodeId, event) {
    if (!canEdit) return;
    if (graphNodesById.get(safeText(nodeId))?.isSyntheticGround) return;
    event.preventDefault();
    event.stopPropagation();
    draggingNodeIdRef.current = nodeId;
    setDraggingNodeId(nodeId);
  }

  function dragNode(event) {
    if (!draggingNodeId) return;
    const point = eventPointInSvg(event);
    if (!point) return;
    setNodePositions((current) => ({
      ...current,
      [draggingNodeId]: {
        x: clamp(point.x, 90, graphLayout.width - 90),
        y: clamp(point.y, 60, graphLayout.height - 60),
      },
    }));
  }

  function stopNodeDrag() {
    if (!draggingNodeId) return;
    draggingNodeIdRef.current = '';
    setDraggingNodeId('');
  }

  function stopConnectionDrag() {
    if (!activeConnection) return;
    activeConnectionRef.current = null;
    setActiveConnection(null);
  }

  function beginRebindSource(edgeIndex, event) {
    if (!canEdit) return;
    if (event.button !== 0) return;
    const relation = asList(draftGraph?.relations)[edgeIndex];
    if (!relation) return;
    event.preventDefault();
    event.stopPropagation();
    const sourceNode = graphNodesById.get(safeText(relation.parent_object_id));
    const anchor = sourceNode ? {
      x: sourceNode.x + NODE_CONNECT_OFFSET,
      y: sourceNode.y,
    } : eventPointInSvg(event);
    if (!anchor) return;
    activeConnectionRef.current = {
      mode: 'rebind-source',
      edgeIndex,
      anchorX: anchor.x,
      anchorY: anchor.y,
      pointer: anchor,
    };
    setActiveConnection({
      mode: 'rebind-source',
      edgeIndex,
      anchorX: anchor.x,
      anchorY: anchor.y,
      pointer: anchor,
    });
    setDraggingNodeId('');
    setSelectedEdgeIndex(edgeIndex);
  }

  function beginRebindTarget(edgeIndex, event) {
    if (!canEdit) return;
    if (event.button !== 0) return;
    const relation = asList(draftGraph?.relations)[edgeIndex];
    if (!relation) return;
    event.preventDefault();
    event.stopPropagation();
    const targetNode = graphNodesById.get(safeText(relation.child_object_id));
    const anchor = targetNode ? {
      x: targetNode.x - NODE_CONNECT_OFFSET,
      y: targetNode.y,
    } : eventPointInSvg(event);
    if (!anchor) return;
    activeConnectionRef.current = {
      mode: 'rebind-target',
      edgeIndex,
      anchorX: anchor.x,
      anchorY: anchor.y,
      pointer: anchor,
    };
    setActiveConnection({
      mode: 'rebind-target',
      edgeIndex,
      anchorX: anchor.x,
      anchorY: anchor.y,
      pointer: anchor,
    });
    setDraggingNodeId('');
    setSelectedEdgeIndex(edgeIndex);
  }

  function handleMouseMove(event) {
    dragNode(event);
    const point = eventPointInSvg(event);
    if (!point) return;
    if (!activeConnection) return;
    setActiveConnection((current) => (current ? { ...current, pointer: point } : current));
  }

  function handleMouseUp() {
    if (!activeConnection) return;

    const point = activeConnection.pointer;
    const hoveredPort = point ? findPortAtPoint(graphLayout.nodes, point, activeConnection.mode === 'rebind-source' ? ['out'] : ['in']) : null;
    const relationList = asList(draftGraph?.relations);
    if (hoveredPort) {
      if (activeConnection.mode === 'rebind-source' && Number.isInteger(activeConnection.edgeIndex)) {
        const index = activeConnection.edgeIndex;
        const relation = relationList[index];
        if (!relation) return;
        const nextParentId = hoveredPort.nodeId;
        if (nextParentId === relation.child_object_id || nextParentId === relation.parent_object_id) {
          stopConnectionDrag();
          return;
        }
        if (hasDuplicateRelation(relationList, nextParentId, relation.relation_type, relation.child_object_id, index)) {
          stopConnectionDrag();
          return;
        }
        updateDraft((current) => ({
          ...current,
          relations: asList(current.relations).map((relation, relationIndex) => (
            relationIndex === index
              ? {
                ...relation,
                parent_object_id: nextParentId,
              }
              : relation
          )),
        }));
        setSelectedEdgeIndex(index);
      } else if (activeConnection.mode === 'rebind-target' && Number.isInteger(activeConnection.edgeIndex)) {
        const index = activeConnection.edgeIndex;
        const relation = relationList[index];
        if (!relation) return;
        const nextChildId = hoveredPort.nodeId;
        if (nextChildId === relation.parent_object_id || nextChildId === relation.child_object_id) {
          stopConnectionDrag();
          return;
        }
        if (hasDuplicateRelation(relationList, relation.parent_object_id, relation.relation_type, nextChildId, index)) {
          stopConnectionDrag();
          return;
        }
        updateDraft((current) => ({
          ...current,
          relations: asList(current.relations).map((relation, relationIndex) => (
            relationIndex === index
              ? {
                ...relation,
                child_object_id: nextChildId,
              }
              : relation
          )),
        }));
        setSelectedEdgeIndex(index);
      }
    }

    stopConnectionDrag();
  }

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;

    const cancelInteractiveGesture = () => {
      if (draggingNodeIdRef.current) {
        draggingNodeIdRef.current = '';
        setDraggingNodeId('');
      }
      if (activeConnectionRef.current) {
        activeConnectionRef.current = null;
        setActiveConnection(null);
      }
    };

    window.addEventListener('mouseup', cancelInteractiveGesture);
    window.addEventListener('blur', cancelInteractiveGesture);
    return () => {
      window.removeEventListener('mouseup', cancelInteractiveGesture);
      window.removeEventListener('blur', cancelInteractiveGesture);
    };
  }, []);

  return (
    <div className="relation-editor-shell">
      <div className="relation-editor-toolbar">
        <div className="relation-editor-summary">
          <strong>Directed relation graph</strong>
          <span className="mono">{asList(draftGraph.relations).length} edge(s) · {objectEntries.length} object(s)</span>
        </div>
        <div className="relation-editor-actions">
          {canEdit ? (
            <>
              <button type="button" className="secondary-button" onClick={addEdge}>
                Add edge
              </button>
              <button
                type="button"
                className="secondary-button"
                onClick={() => {
                  setDraftGraph(buildDraftGraph(graph, objectCatalog));
                  setDraftDirty(false);
                  setSelectedEdgeIndex(0);
                }}
                disabled={saveBusy || !draftDirty}
              >
                Reset draft
              </button>
              {hasSavedOverride && onClearSavedOverride ? (
                <button
                  type="button"
                  className="secondary-button"
                  onClick={onClearSavedOverride}
                  disabled={saveBusy}
                >
                  Clear saved override
                </button>
              ) : null}
              <button type="button" className="primary-button" onClick={handleSave} disabled={saveDisabled}>
                {saveBusy ? 'Saving...' : 'Save graph'}
              </button>
              {relationRetryStage && onSaveAndRetry ? (
                <button type="button" className="primary-button" onClick={handleSaveAndRetry} disabled={saveDisabled}>
                  {saveBusy ? 'Saving...' : `Save + retry ${relationRetryStage.label}`}
                </button>
              ) : null}
            </>
          ) : null}
        </div>
      </div>
      {hasSavedOverride ? (
        <div className="note-box">A saved relation override is active for this job. Retry a downstream stage to materialize it into the runtime outputs.</div>
      ) : null}
      {repeatedInstanceCount > 0 ? (
        <div className="note-box">Repeated detections are expanded into instance-level nodes here, so you can edit directed edges for each individual object instance.</div>
      ) : null}
      {validationIssues.length ? (
        <div className="error-box">
          <strong>Graph validation</strong>
          <ul className="bullet-list">
            {validationIssues.map((issue) => <li key={issue}>{issue}</li>)}
          </ul>
        </div>
      ) : null}
      {saveError ? (
        <div className="error-box">
          <strong>Save failed</strong>
          <div>{safeText(saveError.userMessage, 'Could not save the relation graph.')}</div>
          {saveError.detail && saveError.detail !== saveError.userMessage ? (
            <div className="error-detail">{saveError.detail}</div>
          ) : null}
        </div>
      ) : null}
      <div className="relation-editor-grid">
        <div className="info-card relation-editor-pane relation-editor-pane-graph">
          <div className="card-heading">
            <div>
              <h3 className="card-title">Graph view</h3>
              <p className="card-subtle">Direct support and containment edges, with drag handles for rewiring.</p>
            </div>
            <span className="scroll-affordance">Scroll both directions</span>
          </div>
          <div className="relation-graph-frame scroll-surface scroll-surface-both">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${graphLayout.width} ${graphLayout.height}`}
              role="img"
              aria-label="Directed relation graph"
              onMouseMove={handleMouseMove}
              onMouseUp={() => {
                stopNodeDrag();
                handleMouseUp();
              }}
              onMouseLeave={() => {
                stopNodeDrag();
                stopConnectionDrag();
              }}
            >
              <defs>
                {edgeMarkers.map((marker) => (
                  <marker
                    key={marker.id}
                    id={marker.id}
                    markerWidth="10"
                    markerHeight="10"
                    refX="8"
                    refY="5"
                    orient="auto"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill={marker.fill} />
                  </marker>
                ))}
              </defs>
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
                const selected = !syntheticRoot && selectedEdgeIndex === edge.index;
                const visuals = buildEdgeVisualStyle(edge.relation, { selected });
                const markerId = markerIdForStrokeColor(visuals.line.stroke);
                return (
                  <g
                    key={`${edge.relation.parent_object_id}-${edge.relation.child_object_id}-${edge.index}`}
                    className={selected ? 'relation-edge-selected' : ''}
                    onClick={syntheticRoot ? undefined : () => setSelectedEdgeIndex(edge.index)}
                  >
                    {!syntheticRoot ? (
                      <path
                        d={edge.pathD}
                        className="relation-edge-hit-area"
                        style={{
                          stroke: RELATION_GRAPH_THEME.edgeHitStroke,
                          strokeWidth: EDGE_HIT_STROKE_WIDTH,
                          fill: 'none',
                        }}
                      />
                    ) : null}
                    <path
                      d={edge.pathD}
                      markerEnd={`url(#${markerId})`}
                      className="relation-edge-line"
                      style={{ ...visuals.line, fill: 'none' }}
                    />
                    {canEdit && !syntheticRoot ? (
                      <g className="relation-edge-endpoint-handles">
                        <circle
                          cx={edge.x1}
                          cy={edge.y1}
                          r={DRAG_HANDLE_HIT_RADIUS}
                          className="relation-edge-handle-hit-area"
                          style={{ fill: RELATION_GRAPH_THEME.handleHitFill }}
                          onMouseDown={(event) => beginRebindSource(edge.index, event)}
                          onClick={(event) => event.stopPropagation()}
                        />
                        <circle
                          cx={edge.x1}
                          cy={edge.y1}
                          r={DRAG_HANDLE_RADIUS}
                          className={`relation-edge-handle ${activeConnection?.edgeIndex === edge.index ? 'relation-edge-handle-active' : ''}`}
                          style={visuals.handle}
                          onMouseDown={(event) => beginRebindSource(edge.index, event)}
                          onClick={(event) => event.stopPropagation()}
                        />
                        <circle
                          cx={edge.x2}
                          cy={edge.y2}
                          r={DRAG_HANDLE_HIT_RADIUS}
                          className="relation-edge-handle-hit-area"
                          style={{ fill: RELATION_GRAPH_THEME.handleHitFill }}
                          onMouseDown={(event) => beginRebindTarget(edge.index, event)}
                          onClick={(event) => event.stopPropagation()}
                        />
                        <circle
                          cx={edge.x2}
                          cy={edge.y2}
                          r={DRAG_HANDLE_RADIUS}
                          className={`relation-edge-handle ${activeConnection?.edgeIndex === edge.index ? 'relation-edge-handle-active' : ''}`}
                          style={visuals.handle}
                          onMouseDown={(event) => beginRebindTarget(edge.index, event)}
                          onClick={(event) => event.stopPropagation()}
                        />
                      </g>
                    ) : null}
                    <text
                      x={edge.labelX}
                      y={edge.labelY}
                      textAnchor="middle"
                      className="relation-edge-label"
                      style={visuals.label}
                    >
                      {syntheticRoot ? 'root' : safeText(edge.relation.relation_type, 'supports')}
                    </text>
                  </g>
                );
              })}
              {graphLayout.nodes.map((node) => {
                const dragging = draggingNodeId === node.id;
                const visuals = buildNodeVisualStyle(node, { dragging });
                return (
                  <g
                    key={node.id}
                    transform={`translate(${node.x - NODE_WIDTH / 2}, ${node.y - NODE_CENTER_Y})`}
                    className={[
                      'relation-node-group',
                      dragging ? 'relation-node-dragging' : '',
                    ].filter(Boolean).join(' ')}
                    onMouseDown={node.isSyntheticGround ? undefined : (event) => startNodeDrag(node.id, event)}
                  >
                    <rect
                      x={-NODE_HIT_PADDING}
                      y={-NODE_HIT_PADDING}
                      width={NODE_WIDTH + (NODE_HIT_PADDING * 2)}
                      height={NODE_HEIGHT + (NODE_HIT_PADDING * 2)}
                      rx="22"
                      className="relation-node-hit-area"
                    />
                    <rect
                      width={NODE_WIDTH}
                      height={NODE_HEIGHT}
                      rx="16"
                      className="relation-node-box"
                      style={visuals.box}
                    />
                    <text
                      x={NODE_WIDTH / 2}
                      y={node.isSyntheticGround ? (NODE_CENTER_Y - 4) : (NODE_CENTER_Y + 1)}
                      textAnchor="middle"
                      dominantBaseline="central"
                      className="relation-node-label relation-node-label-main"
                      style={visuals.label}
                    >
                      {safeText(node.label, node.id)}
                    </text>
                    {node.isSyntheticGround ? (
                      <text
                        x={NODE_WIDTH / 2}
                        y={NODE_CENTER_Y + 14}
                        textAnchor="middle"
                        className="relation-node-subtitle"
                        style={visuals.subtitle}
                      >
                        {safeText(node.subtitle, 'scene root')}
                      </text>
                    ) : null}
                  </g>
                );
              })}
              {activePort ? (
                <circle
                  cx={activePort.x}
                  cy={activePort.y}
                  r="13"
                  className="relation-node-port-preview"
                  style={{
                    fill: RELATION_GRAPH_THEME.previewFill,
                    stroke: RELATION_GRAPH_THEME.previewStroke,
                    strokeWidth: 2,
                  }}
                />
              ) : null}
            </svg>
          </div>
          <div className="note-box">
            Drag nodes to arrange the direct support or containment graph.
            Add new edges from the toolbar or edge editor below. Drag edge endpoints (small circles) to rewire connections.
          </div>
        </div>
        <div className="info-card relation-editor-pane relation-editor-pane-fields">
          <div className="card-heading">
            <div>
              <h3 className="card-title">Edge editor</h3>
              <p className="card-subtle">Review each directed edge in a wide stack so parent, relation, and child stay visible together.</p>
            </div>
            <span className="scroll-affordance">Scroll vertically</span>
          </div>
          {asList(draftGraph.relations).length ? (
            <div className="relation-edge-list scroll-surface scroll-surface-vertical">
              {asList(draftGraph.relations).map((relation, index) => (
                <div
                  className={[
                    'relation-edge-row',
                    selectedEdgeIndex === index ? 'relation-edge-row-selected' : '',
                  ].filter(Boolean).join(' ')}
                  key={`${relation.parent_object_id}-${relation.child_object_id}-${index}`}
                  onClick={() => setSelectedEdgeIndex(index)}
                  style={buildEdgeRowStyle(relation, selectedEdgeIndex === index)}
                >
                  <div className="relation-edge-row-head">
                    <div className="relation-edge-row-copy">
                      <div className="relation-edge-row-meta">
                        <span className="relation-edge-chip" style={buildEdgeChipStyle()}>Edge {index + 1}</span>
                        <span className="relation-type-badge" style={buildRelationTypeBadgeStyle(relation)}>
                          {safeText(relation.relation_type, 'supports')}
                        </span>
                      </div>
                      <div className="relation-edge-summary">
                        {objectLabel(relation.parent_object_id)}
                        <span className="relation-edge-summary-arrow" aria-hidden="true">→</span>
                        {objectLabel(relation.child_object_id)}
                      </div>
                    </div>
                    <div className="relation-inline-actions">
                      <button type="button" className="secondary-button" onClick={() => reverseEdge(index)} disabled={!canEdit}>
                        Reverse
                      </button>
                      <button type="button" className="secondary-button" onClick={() => deleteEdge(index)} disabled={!canEdit}>
                        Delete
                      </button>
                    </div>
                  </div>
                  <div className="relation-edge-fields">
                    <label className="relation-field">
                      <span>Parent</span>
                      <select
                        className="select"
                        value={relation.parent_object_id}
                        onChange={(event) => updateEdge(index, { parent_object_id: event.target.value })}
                        disabled={!canEdit}
                      >
                        {objectEntries.map((entry) => (
                          <option key={`parent-${entry.id}`} value={entry.id}>{entry.label}</option>
                        ))}
                      </select>
                    </label>
                    <label className="relation-field">
                      <span>Relation</span>
                      <select
                        className="select"
                        value={relation.relation_type}
                        onChange={(event) => updateEdge(index, { relation_type: event.target.value })}
                        disabled={!canEdit}
                      >
                        {RELATION_TYPE_OPTIONS.map((option) => (
                          <option key={`${index}-${option}`} value={option}>{option}</option>
                        ))}
                      </select>
                    </label>
                    <label className="relation-field">
                      <span>Child</span>
                      <select
                        className="select"
                        value={relation.child_object_id}
                        onChange={(event) => updateEdge(index, { child_object_id: event.target.value })}
                        disabled={!canEdit}
                      >
                        {objectEntries.map((entry) => (
                          <option key={`child-${entry.id}`} value={entry.id}>{entry.label}</option>
                        ))}
                      </select>
                    </label>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty">No relation edges are present yet. Add one to define support or containment.</div>
          )}
        </div>
      </div>
      <div className="stage-grid-single stage-section-gap">
        <div className="info-card">
          <h3 className="card-title">Object descriptions</h3>
          <ObjectDescriptionTable objectDescriptions={objectDescriptions} />
        </div>
      </div>
    </div>
  );
}

export default RelationGraphEditor;
