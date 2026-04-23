import React from 'react';

const VIEWBOX_WIDTH = 720;
const VIEWBOX_HEIGHT = 360;
const CARD_GRADIENT = 'linear-gradient(180deg, rgba(22, 26, 33, 0.96) 0%, rgba(13, 17, 23, 0.98) 100%)';
const SURFACE_GRADIENT = 'linear-gradient(180deg, rgba(18, 24, 32, 0.96) 0%, rgba(10, 14, 20, 0.98) 100%)';
const PLOT_SURFACE = 'rgba(11, 16, 22, 0.76)';
const GRID_STROKE = 'rgba(120, 141, 171, 0.16)';
const AXIS_TEXT = '#98a5b7';
const AXIS_CAPSULE_FILL = 'rgba(9, 14, 20, 0.94)';
const AXIS_CAPSULE_STROKE = 'rgba(126, 148, 178, 0.28)';
const X_AXIS_ACCENT = '#8fb8ff';
const Z_AXIS_ACCENT = '#79d4c8';
const LABEL_SURFACE = 'rgba(8, 12, 18, 0.94)';
const PLOT_BOUNDS = {
  left: 40,
  right: 438,
  top: 52,
  bottom: 292,
};
const LEGEND_BOUNDS = {
  left: 476,
  right: 690,
  top: 64,
  bottom: 284,
};

const ROOT_STYLE = {
  display: 'grid',
  gap: '16px',
  padding: '18px',
  borderRadius: '28px',
  border: '1px solid var(--line)',
  background: CARD_GRADIENT,
  boxShadow: 'var(--shadow-md)',
  overflow: 'hidden',
};

const HEADER_STYLE = {
  display: 'flex',
  alignItems: 'flex-start',
  justifyContent: 'space-between',
  gap: '12px',
  flexWrap: 'wrap',
};

const TITLE_BLOCK_STYLE = {
  display: 'grid',
  gap: '6px',
};

const EYEBROW_STYLE = {
  fontSize: '11px',
  letterSpacing: '0.14em',
  textTransform: 'uppercase',
  color: 'var(--accent)',
  fontWeight: 700,
};

const TITLE_STYLE = {
  margin: 0,
  fontSize: '20px',
  lineHeight: 1.1,
  color: 'var(--ink)',
  fontWeight: 700,
};

const SUBTITLE_STYLE = {
  margin: 0,
  fontSize: '13px',
  lineHeight: 1.5,
  color: 'var(--ink-soft)',
  maxWidth: '64ch',
};

const CHIP_ROW_STYLE = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: '8px',
  justifyContent: 'flex-end',
};

const CHIP_STYLE = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '8px',
  minHeight: '32px',
  padding: '0 12px',
  borderRadius: '999px',
  border: '1px solid var(--line-strong)',
  background: 'rgba(255, 255, 255, 0.05)',
  color: 'var(--ink)',
  fontSize: '12px',
  fontWeight: 600,
  backdropFilter: 'blur(14px)',
};

const MAP_GUIDE_STYLE = {
  display: 'grid',
  gap: '12px',
  padding: '14px 16px',
  borderRadius: '18px',
  border: '1px solid var(--line)',
  background: 'linear-gradient(180deg, rgba(255, 255, 255, 0.035), rgba(255, 255, 255, 0.015))',
};

const MAP_GUIDE_HEAD_STYLE = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: '10px',
  flexWrap: 'wrap',
};

const MAP_GUIDE_META_STYLE = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: '8px',
};

const MAP_GUIDE_PILL_STYLE = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '8px',
  minHeight: '28px',
  padding: '0 12px',
  borderRadius: '999px',
  border: '1px solid var(--line)',
  background: 'rgba(255, 255, 255, 0.04)',
  color: 'var(--ink-soft)',
  fontSize: '11px',
  fontWeight: 600,
  letterSpacing: '0.02em',
};

const MAP_GUIDE_TEXT_STYLE = {
  margin: 0,
  color: 'var(--ink-soft)',
  fontSize: '12px',
  lineHeight: 1.6,
};

function asList(value) {
  return Array.isArray(value) ? value : [];
}

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function toNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function median(values) {
  if (!values.length) {
    return null;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) * 0.5;
  }
  return sorted[middle];
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function trimTrailingZeros(value) {
  return String(value).replace(/(\.\d*?[1-9])0+$/u, '$1').replace(/\.0+$/u, '');
}

function formatMeters(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 'n/a';
  }
  if (Math.abs(numeric) >= 10) {
    return trimTrailingZeros(numeric.toFixed(1));
  }
  if (Math.abs(numeric) >= 1) {
    return trimTrailingZeros(numeric.toFixed(2));
  }
  return trimTrailingZeros(numeric.toFixed(3));
}

function collapseRepeatedObjectSuffix(value) {
  const raw = String(value || '').trim();
  if (!raw) {
    return '';
  }
  const duplicatedTailMatch = raw.match(/(?:^|[_-])([a-z][a-z0-9]*\d*)(?:[_-]\1)+$/iu);
  return duplicatedTailMatch?.[1] || raw;
}

function formatObjectDisplayName(value, fallback = 'Object') {
  const raw = String(value || '').trim();
  if (!raw) {
    return fallback;
  }

  let display = raw.includes(':')
    ? raw.split(':').filter(Boolean).pop() || raw
    : raw;

  display = collapseRepeatedObjectSuffix(display);
  display = display.replace(/[_-]+/g, ' ').trim();
  display = display.replace(/\s+/g, ' ');
  display = display.replace(/(.+?)\s+(\d+)$/u, '$1$2');

  return display || fallback;
}

function buildObjectIdCandidates(value) {
  const raw = String(value || '').trim();
  if (!raw) {
    return [];
  }

  const results = new Set();
  const queue = [raw];

  while (queue.length) {
    const current = queue.shift();
    if (!current) {
      continue;
    }
    const normalized = current.toLowerCase();
    if (results.has(normalized)) {
      continue;
    }
    results.add(normalized);

    if (current.includes(':')) {
      const tail = current.split(':').filter(Boolean).pop();
      if (tail && tail !== current) {
        queue.push(tail);
      }
    }

    const collapsed = collapseRepeatedObjectSuffix(current);
    if (collapsed && collapsed !== current) {
      queue.push(collapsed);
    }

    const stripped = current.replace(/(?:(?:::)|[_-]|\b)(\d+)$/u, '').replace(/[:_-]+$/u, '');
    if (stripped && stripped !== current) {
      queue.push(stripped);
    }
  }

  return [...results];
}

function parseTriplet(value) {
  if (Array.isArray(value) && value.length >= 3) {
    const x = toNumber(value[0]);
    const y = toNumber(value[1]);
    const z = toNumber(value[2]);
    if (x !== null && y !== null && z !== null) {
      return [x, y, z];
    }
  }

  if (isRecord(value)) {
    const x = toNumber(value.x ?? value.width ?? value.min_x ?? value.max_x ?? value.cx);
    const y = toNumber(value.y ?? value.height ?? value.min_y ?? value.max_y ?? value.cy);
    const z = toNumber(value.z ?? value.depth ?? value.min_z ?? value.max_z ?? value.cz);
    if (x !== null && y !== null && z !== null) {
      return [x, y, z];
    }
  }

  return null;
}

function parsePair(value) {
  if (Array.isArray(value) && value.length >= 2) {
    const x = toNumber(value[0]);
    const z = toNumber(value[1]);
    if (x !== null && z !== null) {
      return [x, z];
    }
  }

  if (isRecord(value)) {
    const x = toNumber(value.x ?? value.min_x ?? value.max_x ?? value.cx ?? value.left ?? value.right);
    const z = toNumber(value.z ?? value.min_z ?? value.max_z ?? value.cz ?? value.top ?? value.bottom);
    if (x !== null && z !== null) {
      return [x, z];
    }
  }

  return null;
}

function parseBoundsXZ(value) {
  if (Array.isArray(value) && value.length >= 4) {
    const minX = toNumber(value[0]);
    const minZ = toNumber(value[1]);
    const maxX = toNumber(value[2]);
    const maxZ = toNumber(value[3]);
    if (minX !== null && minZ !== null && maxX !== null && maxZ !== null) {
      return {
        minX: Math.min(minX, maxX),
        minZ: Math.min(minZ, maxZ),
        maxX: Math.max(minX, maxX),
        maxZ: Math.max(minZ, maxZ),
      };
    }
  }

  if (Array.isArray(value) && value.length >= 2) {
    const min = parsePair(value[0]);
    const max = parsePair(value[1]);
    if (min && max) {
      return {
        minX: Math.min(min[0], max[0]),
        minZ: Math.min(min[1], max[1]),
        maxX: Math.max(min[0], max[0]),
        maxZ: Math.max(min[1], max[1]),
      };
    }
  }

  if (isRecord(value)) {
    const min = parsePair(value.min ?? value.minimum ?? value.low);
    const max = parsePair(value.max ?? value.maximum ?? value.high);
    if (min && max) {
      return {
        minX: Math.min(min[0], max[0]),
        minZ: Math.min(min[1], max[1]),
        maxX: Math.max(min[0], max[0]),
        maxZ: Math.max(min[1], max[1]),
      };
    }

    const minX = toNumber(value.minX ?? value.min_x ?? value.left);
    const minZ = toNumber(value.minZ ?? value.min_z ?? value.top);
    const maxX = toNumber(value.maxX ?? value.max_x ?? value.right);
    const maxZ = toNumber(value.maxZ ?? value.max_z ?? value.bottom);
    if (minX !== null && minZ !== null && maxX !== null && maxZ !== null) {
      return {
        minX: Math.min(minX, maxX),
        minZ: Math.min(minZ, maxZ),
        maxX: Math.max(minX, maxX),
        maxZ: Math.max(minZ, maxZ),
      };
    }
  }

  return null;
}

function parseBoundsXYZ(value) {
  if (Array.isArray(value) && value.length >= 2) {
    const min = parseTriplet(value[0]);
    const max = parseTriplet(value[1]);
    if (min && max) {
      return {
        min: [
          Math.min(min[0], max[0]),
          Math.min(min[1], max[1]),
          Math.min(min[2], max[2]),
        ],
        max: [
          Math.max(min[0], max[0]),
          Math.max(min[1], max[1]),
          Math.max(min[2], max[2]),
        ],
      };
    }
  }

  if (isRecord(value)) {
    const min = parseTriplet(
      value.min
      ?? value.minimum
      ?? value.low
      ?? value.min_xyz
      ?? value.bbox_min
      ?? value.bbox_min_xyz,
    );
    const max = parseTriplet(
      value.max
      ?? value.maximum
      ?? value.high
      ?? value.max_xyz
      ?? value.bbox_max
      ?? value.bbox_max_xyz,
    );
    if (min && max) {
      return {
        min: [
          Math.min(min[0], max[0]),
          Math.min(min[1], max[1]),
          Math.min(min[2], max[2]),
        ],
        max: [
          Math.max(min[0], max[0]),
          Math.max(min[1], max[1]),
          Math.max(min[2], max[2]),
        ],
      };
    }

    const center = parseTriplet(value.center ?? value.center_xyz ?? value.centroid ?? value.centroid_xyz);
    const extents = parseTriplet(value.extents ?? value.extents_xyz ?? value.size ?? value.size_xyz ?? value.dimensions_xyz);
    if (center && extents) {
      return {
        min: [
          center[0] - (extents[0] * 0.5),
          center[1] - (extents[1] * 0.5),
          center[2] - (extents[2] * 0.5),
        ],
        max: [
          center[0] + (extents[0] * 0.5),
          center[1] + (extents[1] * 0.5),
          center[2] + (extents[2] * 0.5),
        ],
      };
    }
  }

  return null;
}

function extractTranslationXZ(source) {
  const direct = parseTriplet(
    source?.translation_xyz
    ?? source?.translation
    ?? source?.position
    ?? source?.center_xyz
    ?? source?.center
    ?? source?.centroid_xyz
    ?? source?.centroid,
  );
  if (!direct) {
    return null;
  }
  return { x: direct[0], z: direct[2] };
}

function extractYawRadians(source) {
  const rotationType = String(source?.rotation_type ?? source?.rotationType ?? '').trim().toLowerCase();
  const rotationValue = source?.rotation_value ?? source?.rotation ?? source?.orientation ?? null;
  const values = Array.isArray(rotationValue) ? rotationValue.map((item) => toNumber(item)) : [];
  if (rotationType.includes('quat') && values.length >= 4 && values.every((item) => item !== null)) {
    const [w, x, y, z] = values;
    const yaw = Math.atan2(2 * ((w * y) + (x * z)), 1 - (2 * ((y * y) + (z * z))));
    return Number.isFinite(yaw) ? yaw : null;
  }
  if (rotationType.includes('euler') && values.length >= 3 && values[1] !== null) {
    return values[1];
  }
  return null;
}

function extractExplicitFootprintBounds(source) {
  const boundsXZCandidates = [
    source?.bounds_xz,
    source?.footprint_bounds_xz,
    source?.bounding_box_xz,
    source?.bbox_xz,
    source?.horizontal_bounds,
    source?.footprint?.bounds_xz,
    source?.footprint?.bbox_xz,
    source?.bounding_box?.bounds_xz,
    source?.bounding_box?.bbox_xz,
    source?.bbox?.bounds_xz,
    source?.bbox?.bbox_xz,
  ];

  for (const candidate of boundsXZCandidates) {
    const parsed = parseBoundsXZ(candidate);
    if (parsed) {
      return parsed;
    }
  }

  const boundsXYZCandidates = [
    source?.bounds,
    source?.bounds_xyz,
    source?.bounding_box,
    source?.bbox,
    source?.aabb,
    source?.axis_aligned_bounding_box,
    isRecord(source?.bbox_min_xyz) || Array.isArray(source?.bbox_min_xyz)
      ? { min_xyz: source.bbox_min_xyz, max_xyz: source.bbox_max_xyz }
      : null,
    isRecord(source?.bbox_min) || Array.isArray(source?.bbox_min)
      ? { bbox_min: source.bbox_min, bbox_max: source.bbox_max }
      : null,
    isRecord(source?.min_xyz) || Array.isArray(source?.min_xyz)
      ? { min_xyz: source.min_xyz, max_xyz: source.max_xyz }
      : null,
  ];

  for (const candidate of boundsXYZCandidates) {
    const parsed = parseBoundsXYZ(candidate);
    if (parsed) {
      return {
        minX: parsed.min[0],
        minZ: parsed.min[2],
        maxX: parsed.max[0],
        maxZ: parsed.max[2],
      };
    }
  }

  return null;
}

function extractDimensionFootprint(source) {
  const candidates = [
    source?.dimensions_m,
    source?.dimensions,
    source?.extents_xyz,
    source?.extents,
    source?.size_xyz,
    source?.size_m,
    source?.size,
    source?.footprint?.dimensions_m,
    source?.footprint?.size_xyz,
    source?.bounding_box?.dimensions_m,
    source?.bounding_box?.extents_xyz,
    source?.bounding_box?.extents,
    source?.bbox?.dimensions_m,
    source?.bbox?.extents_xyz,
  ];

  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    if (Array.isArray(candidate) && candidate.length >= 3) {
      const width = Math.abs(Number(candidate[0]));
      const depth = Math.abs(Number(candidate[2]));
      if (Number.isFinite(width) && width > 0 && Number.isFinite(depth) && depth > 0) {
        return { width, depth };
      }
    }
    if (Array.isArray(candidate) && candidate.length >= 2) {
      const width = Math.abs(Number(candidate[0]));
      const depth = Math.abs(Number(candidate[1]));
      if (Number.isFinite(width) && width > 0 && Number.isFinite(depth) && depth > 0) {
        return { width, depth };
      }
    }
    if (isRecord(candidate)) {
      const width = Math.abs(Number(candidate.x ?? candidate.width ?? candidate.w));
      const depth = Math.abs(Number(candidate.z ?? candidate.depth ?? candidate.d));
      if (Number.isFinite(width) && width > 0 && Number.isFinite(depth) && depth > 0) {
        return { width, depth };
      }
    }
  }

  return null;
}

function extractScaleFootprint(source) {
  const scale = source?.scale_xyz ?? source?.scale ?? null;
  const parsed = parseTriplet(scale);
  if (!parsed) {
    return null;
  }
  const width = Math.abs(parsed[0]);
  const depth = Math.abs(parsed[2]);
  if (!(width > 0 && depth > 0)) {
    return null;
  }
  return { width, depth };
}

function niceStep(rawStep) {
  if (!(rawStep > 0)) {
    return 1;
  }
  const exponent = Math.floor(Math.log10(rawStep));
  const magnitude = 10 ** exponent;
  const normalized = rawStep / magnitude;
  if (normalized <= 1) {
    return magnitude;
  }
  if (normalized <= 2) {
    return 2 * magnitude;
  }
  if (normalized <= 5) {
    return 5 * magnitude;
  }
  return 10 * magnitude;
}

function buildTicks(min, max, targetCount = 4) {
  const span = Math.max(Math.abs(max - min), 1e-6);
  const step = niceStep(span / Math.max(targetCount, 1));
  const start = Math.ceil(min / step) * step;
  const end = Math.floor(max / step) * step;
  const ticks = [];
  for (let value = start; value <= end + (step * 0.5); value += step) {
    const rounded = Math.abs(value) < 1e-9 ? 0 : Number(value.toFixed(8));
    ticks.push(rounded);
  }
  return ticks;
}

function colorForIndex(index) {
  const hueSeed = (index * 137.508) % 360;
  const hue = hueSeed >= 248 && hueSeed <= 322 ? (hueSeed + 82) % 360 : hueSeed;
  return {
    stroke: `hsl(${hue} 72% 42%)`,
    fill: `hsl(${hue} 78% 52% / 0.16)`,
    wash: `hsl(${hue} 86% 44% / 0.08)`,
    chip: `hsl(${hue} 78% 32%)`,
    accent: `hsl(${hue} 82% 56%)`,
  };
}

function appendMetadataItems(target, source, fallbackKeyPrefix = 'item') {
  if (!source) {
    return;
  }

  if (Array.isArray(source)) {
    source.forEach((item, index) => {
      if (!isRecord(item)) {
        return;
      }
      const objectKey = item.object_id ?? item.id ?? item.name ?? `${fallbackKeyPrefix}-${index + 1}`;
      target.push({ key: objectKey, data: item });
    });
    return;
  }

  if (isRecord(source) && Array.isArray(source.objects)) {
    appendMetadataItems(target, source.objects, fallbackKeyPrefix);
    return;
  }

  if (isRecord(source)) {
    Object.entries(source).forEach(([key, value]) => {
      if (!isRecord(value)) {
        return;
      }
      target.push({ key, data: value.object_id ? value : { object_id: key, ...value } });
    });
  }
}

function buildMetadataIndex({ layout, sizePriors, objectCatalog, objectMetadata }) {
  const entries = [];
  appendMetadataItems(entries, objectMetadata, 'meta');
  appendMetadataItems(entries, sizePriors ?? layout?.size_priors ?? layout?.sizePriors, 'size');
  appendMetadataItems(entries, objectCatalog ?? layout?.object_catalog ?? layout?.objectCatalog, 'catalog');
  appendMetadataItems(entries, layout?.objects, 'layout');

  let serial = 0;
  const index = new Map();
  entries.forEach((entry) => {
    const sourceIds = [
      entry.key,
      entry.data?.object_id,
      entry.data?.id,
      ...(Array.isArray(entry.data?.source_instance_ids) ? entry.data.source_instance_ids : []),
    ].filter(Boolean);
    sourceIds.forEach((sourceId) => {
      buildObjectIdCandidates(sourceId).forEach((candidate) => {
        const bucket = index.get(candidate) || [];
        bucket.push({
          id: `${candidate}:${serial += 1}`,
          data: entry.data,
        });
        index.set(candidate, bucket);
      });
    });
  });

  return index;
}

function resolveMetadataEntries(index, objectId) {
  const resolved = [];
  const seen = new Set();
  buildObjectIdCandidates(objectId).forEach((candidate) => {
    asList(index.get(candidate)).forEach((entry) => {
      if (!seen.has(entry.id)) {
        seen.add(entry.id);
        resolved.push(entry.data);
      }
    });
  });
  return resolved;
}

function resolveLabel(pose, metadataEntries, fallbackLabel) {
  const candidates = [
    pose?.object_id,
    ...metadataEntries.flatMap((entry) => [
      ...(Array.isArray(entry?.source_instance_ids) ? entry.source_instance_ids : []),
      entry?.object_id,
      entry?.id,
    ]),
    pose?.display_name,
    pose?.label,
    pose?.name,
    ...metadataEntries.flatMap((entry) => [
      entry?.display_name,
      entry?.label,
      entry?.name,
      entry?.canonical_name,
      entry?.object_name,
    ]),
  ];

  const label = candidates.find((value) => String(value || '').trim());
  return label ? formatObjectDisplayName(label, fallbackLabel) : fallbackLabel;
}

function preparePoseEntries(poses, metadataIndex) {
  return poses.map((pose, index) => {
    const objectId = String(pose?.object_id || `object-${index + 1}`);
    const metadataEntries = resolveMetadataEntries(metadataIndex, objectId);
    const translation = extractTranslationXZ(pose);
    const label = resolveLabel(pose, metadataEntries, `object${index + 1}`);
    const sources = [pose, ...metadataEntries];

    let bounds = null;
    for (const source of sources) {
      bounds = extractExplicitFootprintBounds(source);
      if (bounds) {
        break;
      }
    }

    let dimensions = null;
    if (!bounds) {
      for (const source of sources) {
        dimensions = extractDimensionFootprint(source);
        if (dimensions) {
          break;
        }
      }
    }

    let scaleFootprint = null;
    if (!bounds && !dimensions) {
      scaleFootprint = extractScaleFootprint(pose);
    }

    return {
      key: objectId || `object-${index + 1}`,
      objectId,
      label,
      translation,
      yaw: extractYawRadians(pose),
      explicitBounds: bounds,
      dimensions,
      scaleFootprint,
      rawPose: pose,
    };
  });
}

function finalizePoseEntries(entries) {
  const knownWidths = [];
  const knownDepths = [];

  entries.forEach((entry) => {
    if (entry.explicitBounds) {
      knownWidths.push(Math.max(entry.explicitBounds.maxX - entry.explicitBounds.minX, 0.04));
      knownDepths.push(Math.max(entry.explicitBounds.maxZ - entry.explicitBounds.minZ, 0.04));
      return;
    }
    if (entry.dimensions) {
      knownWidths.push(Math.max(entry.dimensions.width, 0.04));
      knownDepths.push(Math.max(entry.dimensions.depth, 0.04));
      return;
    }
    if (entry.scaleFootprint) {
      knownWidths.push(Math.max(entry.scaleFootprint.width, 0.04));
      knownDepths.push(Math.max(entry.scaleFootprint.depth, 0.04));
    }
  });

  const fallbackWidth = clamp(median(knownWidths) ?? 0.22, 0.12, 1.8);
  const fallbackDepth = clamp(median(knownDepths) ?? 0.22, 0.12, 1.8);

  return entries.map((entry, index) => {
    if (!entry.translation && !entry.explicitBounds) {
      return {
        ...entry,
        skipped: true,
      };
    }

    let minX;
    let maxX;
    let minZ;
    let maxZ;
    let footprintSource;

    if (entry.explicitBounds) {
      ({ minX, minZ, maxX, maxZ } = entry.explicitBounds);
      footprintSource = 'bbox';
    } else {
      const width = Math.max(
        entry.dimensions?.width ?? entry.scaleFootprint?.width ?? fallbackWidth,
        0.04,
      );
      const depth = Math.max(
        entry.dimensions?.depth ?? entry.scaleFootprint?.depth ?? fallbackDepth,
        0.04,
      );
      const centerX = entry.translation?.x ?? 0;
      const centerZ = entry.translation?.z ?? 0;
      minX = centerX - (width * 0.5);
      maxX = centerX + (width * 0.5);
      minZ = centerZ - (depth * 0.5);
      maxZ = centerZ + (depth * 0.5);
      footprintSource = entry.dimensions ? 'dimensions' : entry.scaleFootprint ? 'scale' : 'fallback';
    }

    const width = Math.max(maxX - minX, 0.04);
    const depth = Math.max(maxZ - minZ, 0.04);
    const centerX = entry.translation?.x ?? (minX + maxX) * 0.5;
    const centerZ = entry.translation?.z ?? (minZ + maxZ) * 0.5;
    const colors = colorForIndex(index);

    return {
      ...entry,
      skipped: false,
      minX,
      maxX,
      minZ,
      maxZ,
      width,
      depth,
      centerX,
      centerZ,
      footprintSource,
      colors,
    };
  });
}

function projectPoint(bounds, valueX, valueZ) {
  const plotWidth = PLOT_BOUNDS.right - PLOT_BOUNDS.left;
  const plotHeight = PLOT_BOUNDS.bottom - PLOT_BOUNDS.top;
  const x = PLOT_BOUNDS.left + ((valueX - bounds.minX) / bounds.spanX) * plotWidth;
  const y = PLOT_BOUNDS.bottom - ((valueZ - bounds.minZ) / bounds.spanZ) * plotHeight;
  return { x, y };
}

function buildOccupiedBounds(entries) {
  const minX = Math.min(...entries.map((entry) => entry.minX));
  const maxX = Math.max(...entries.map((entry) => entry.maxX));
  const minZ = Math.min(...entries.map((entry) => entry.minZ));
  const maxZ = Math.max(...entries.map((entry) => entry.maxZ));
  const rawSpanX = Math.max(maxX - minX, 0.2);
  const rawSpanZ = Math.max(maxZ - minZ, 0.2);
  const padX = Math.max(rawSpanX * 0.14, 0.12);
  const padZ = Math.max(rawSpanZ * 0.14, 0.12);
  const paddedMinX = minX - padX;
  const paddedMaxX = maxX + padX;
  const paddedMinZ = minZ - padZ;
  const paddedMaxZ = maxZ + padZ;
  return {
    minX: paddedMinX,
    maxX: paddedMaxX,
    minZ: paddedMinZ,
    maxZ: paddedMaxZ,
    spanX: Math.max(paddedMaxX - paddedMinX, 0.2),
    spanZ: Math.max(paddedMaxZ - paddedMinZ, 0.2),
  };
}

function buildRenderableEntries(entries, bounds) {
  const baseEntries = entries.map((entry) => {
    const topLeft = projectPoint(bounds, entry.minX, entry.maxZ);
    const bottomRight = projectPoint(bounds, entry.maxX, entry.minZ);
    const center = projectPoint(bounds, entry.centerX, entry.centerZ);
    const footprintWidth = Math.max(bottomRight.x - topLeft.x, 14);
    const footprintHeight = Math.max(bottomRight.y - topLeft.y, 14);
    const footprintX = center.x - (footprintWidth * 0.5);
    const footprintY = center.y - (footprintHeight * 0.5);
    return {
      ...entry,
      center,
      footprintWidth,
      footprintHeight,
      footprintX,
      footprintY,
    };
  });

  const labelHeight = 24;
  const labelX = LEGEND_BOUNDS.left + 18;
  const legendAnchorX = LEGEND_BOUNDS.left + 8;
  const labelSlots = [...baseEntries]
    .sort((left, right) => left.center.y - right.center.y);
  const availableTop = LEGEND_BOUNDS.top + 8;
  const availableBottom = LEGEND_BOUNDS.bottom - labelHeight - 8;
  const slotStep = labelSlots.length > 1
    ? (availableBottom - availableTop) / (labelSlots.length - 1)
    : 0;
  const positionsByKey = new Map(
    labelSlots.map((entry, index) => {
      const targetY = labelSlots.length > 1
        ? availableTop + (slotStep * index)
        : ((availableTop + availableBottom) * 0.5);
      const labelY = clamp(targetY, availableTop, availableBottom);
      return [entry.key, { labelX, labelY }];
    }),
  );

  return baseEntries.map((entry) => {
    const legendPosition = positionsByKey.get(entry.key) || { labelX, labelY: availableTop };
    const calloutY = legendPosition.labelY + 8;
    const calloutStartX = Math.min(PLOT_BOUNDS.right - 6, entry.footprintX + entry.footprintWidth + 6);
    const calloutStartY = entry.center.y;
    const calloutMidX = LEGEND_BOUNDS.left - 12;
    return {
      ...entry,
      labelX: legendPosition.labelX,
      labelY: legendPosition.labelY,
      calloutStartX,
      calloutStartY,
      calloutMidX,
      calloutEndX: legendAnchorX,
      calloutEndY: calloutY,
    };
  });
}

function renderEmptyState({ title, subtitle, message, className, style }) {
  return (
    <div className={className} style={{ ...ROOT_STYLE, ...style }}>
      <div style={TITLE_BLOCK_STYLE}>
        <div style={EYEBROW_STYLE}>Stage 05</div>
        <h3 style={TITLE_STYLE}>{title}</h3>
        {subtitle ? <p style={SUBTITLE_STYLE}>{subtitle}</p> : null}
      </div>
      <div
        style={{
          borderRadius: '22px',
          border: '1px dashed var(--line-strong)',
          background: SURFACE_GRADIENT,
          padding: '22px 18px',
          color: 'var(--ink-soft)',
          fontSize: '14px',
          lineHeight: 1.6,
        }}
      >
        {message}
      </div>
    </div>
  );
}

export default function LayoutPoseMap2D({
  layout,
  sizePriors = null,
  objectCatalog = null,
  objectMetadata = null,
  title = 'Pose map',
  subtitle = 'Top-down X/Z occupancy using per-object footprints and centered occupied bounds.',
  emptyMessage = 'No object pose data was recorded for this stage.',
  ariaLabel = 'Top-down layout pose map',
  showLegend = true,
  className = '',
  style = undefined,
}) {
  const poses = asList(layout?.object_poses).filter((pose) => isRecord(pose));
  if (!poses.length) {
    return renderEmptyState({
      title,
      subtitle,
      message: emptyMessage,
      className,
      style,
    });
  }

  const metadataIndex = buildMetadataIndex({
    layout,
    sizePriors,
    objectCatalog,
    objectMetadata,
  });
  const preparedEntries = preparePoseEntries(poses, metadataIndex);
  const entries = finalizePoseEntries(preparedEntries).filter((entry) => !entry.skipped);
  const skippedCount = preparedEntries.length - entries.length;

  if (!entries.length) {
    return renderEmptyState({
      title,
      subtitle,
      message: 'Object poses exist, but no usable X/Z positions or bounds were available for rendering.',
      className,
      style,
    });
  }

  const occupiedBounds = buildOccupiedBounds(entries);
  const xTicks = buildTicks(occupiedBounds.minX, occupiedBounds.maxX);
  const zTicks = buildTicks(occupiedBounds.minZ, occupiedBounds.maxZ);
  const inferredCount = entries.filter((entry) => entry.footprintSource !== 'bbox').length;
  const fallbackCount = entries.filter((entry) => entry.footprintSource === 'fallback').length;
  const sortedEntries = [...entries].sort((left, right) => (right.width * right.depth) - (left.width * left.depth));
  const renderableEntries = buildRenderableEntries(sortedEntries, occupiedBounds);
  const zeroPoint = projectPoint(occupiedBounds, 0, 0);
  const showZeroX = zeroPoint.x >= PLOT_BOUNDS.left && zeroPoint.x <= PLOT_BOUNDS.right;
  const showZeroZ = zeroPoint.y >= PLOT_BOUNDS.top && zeroPoint.y <= PLOT_BOUNDS.bottom;
  const plotCenterX = (PLOT_BOUNDS.left + PLOT_BOUNDS.right) * 0.5;
  const plotCenterY = (PLOT_BOUNDS.top + PLOT_BOUNDS.bottom) * 0.5;
  const xAxisRailY = PLOT_BOUNDS.bottom + 46;
  const xAxisRailLeft = PLOT_BOUNDS.left + 18;
  const xAxisRailRight = PLOT_BOUNDS.right - 18;
  const zAxisRailX = 14;
  const zAxisRailTop = PLOT_BOUNDS.top + 18;
  const zAxisRailBottom = PLOT_BOUNDS.bottom - 18;

  return (
    <div className={className} style={{ ...ROOT_STYLE, ...style }}>
      <div style={HEADER_STYLE}>
        <div style={TITLE_BLOCK_STYLE}>
          <div style={EYEBROW_STYLE}>Stage 05</div>
          <h3 style={TITLE_STYLE}>{title}</h3>
          {subtitle ? <p style={SUBTITLE_STYLE}>{subtitle}</p> : null}
        </div>
        <div style={CHIP_ROW_STYLE}>
          <div style={CHIP_STYLE}>{entries.length} objects</div>
          <div style={CHIP_STYLE}>
            {formatMeters(occupiedBounds.spanX)}m x {formatMeters(occupiedBounds.spanZ)}m occupied
          </div>
          {inferredCount ? <div style={CHIP_STYLE}>{inferredCount} inferred footprints</div> : null}
          {fallbackCount ? <div style={CHIP_STYLE}>{fallbackCount} clean fallbacks</div> : null}
          {skippedCount ? <div style={CHIP_STYLE}>{skippedCount} skipped</div> : null}
        </div>
      </div>

      <svg
        viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
        role="img"
        aria-label={ariaLabel}
        style={{
          width: '100%',
          height: 'auto',
          display: 'block',
          borderRadius: '24px',
          background: SURFACE_GRADIENT,
        }}
      >
        <defs>
          <linearGradient id="layout-pose-map-surface" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#17202a" stopOpacity="0.98" />
            <stop offset="100%" stopColor="#0d131b" stopOpacity="1" />
          </linearGradient>
          <filter id="layout-pose-map-shadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="18" stdDeviation="16" floodColor="#000000" floodOpacity="0.3" />
          </filter>
        </defs>

        <rect
          x="10"
          y="10"
          width={VIEWBOX_WIDTH - 20}
          height={VIEWBOX_HEIGHT - 20}
          rx="28"
          fill="url(#layout-pose-map-surface)"
          stroke="rgba(121, 142, 170, 0.28)"
        />

        <rect
          x={PLOT_BOUNDS.left}
          y={PLOT_BOUNDS.top}
          width={PLOT_BOUNDS.right - PLOT_BOUNDS.left}
          height={PLOT_BOUNDS.bottom - PLOT_BOUNDS.top}
          rx="22"
          fill={PLOT_SURFACE}
          stroke="rgba(121, 142, 170, 0.22)"
        />

        <line
          x1={PLOT_BOUNDS.left}
          y1={PLOT_BOUNDS.bottom}
          x2={PLOT_BOUNDS.right}
          y2={PLOT_BOUNDS.bottom}
          stroke="rgba(143, 184, 255, 0.24)"
          strokeWidth="1.4"
        />
        <line
          x1={PLOT_BOUNDS.left}
          y1={PLOT_BOUNDS.top}
          x2={PLOT_BOUNDS.left}
          y2={PLOT_BOUNDS.bottom}
          stroke="rgba(121, 212, 200, 0.24)"
          strokeWidth="1.4"
        />

        {xTicks.map((tick) => {
          const point = projectPoint(occupiedBounds, tick, occupiedBounds.minZ);
          return (
            <g key={`x-tick-${tick}`}>
              <line
                x1={point.x}
                y1={PLOT_BOUNDS.top}
                x2={point.x}
                y2={PLOT_BOUNDS.bottom}
                stroke={GRID_STROKE}
                strokeDasharray="5 7"
              />
              <text
                x={point.x}
                y={PLOT_BOUNDS.bottom + 18}
                textAnchor="middle"
                fontSize="11"
                fontFamily="'IBM Plex Mono', 'JetBrains Mono', monospace"
                fill={AXIS_TEXT}
              >
                {formatMeters(tick)}
              </text>
            </g>
          );
        })}

        {zTicks.map((tick) => {
          const point = projectPoint(occupiedBounds, occupiedBounds.minX, tick);
          return (
            <g key={`z-tick-${tick}`}>
              <line
                x1={PLOT_BOUNDS.left}
                y1={point.y}
                x2={PLOT_BOUNDS.right}
                y2={point.y}
                stroke={GRID_STROKE}
                strokeDasharray="5 7"
              />
              <text
                x={PLOT_BOUNDS.left + 10}
                y={point.y + 4}
                textAnchor="start"
                fontSize="11"
                fontFamily="'IBM Plex Mono', 'JetBrains Mono', monospace"
                fill={AXIS_TEXT}
              >
                {formatMeters(tick)}
              </text>
            </g>
          );
        })}

        {showZeroX ? (
          <line
            x1={zeroPoint.x}
            y1={PLOT_BOUNDS.top}
            x2={zeroPoint.x}
            y2={PLOT_BOUNDS.bottom}
            stroke="rgba(143, 154, 168, 0.24)"
            strokeDasharray="9 7"
          />
        ) : null}

        {showZeroZ ? (
          <line
            x1={PLOT_BOUNDS.left}
            y1={zeroPoint.y}
            x2={PLOT_BOUNDS.right}
            y2={zeroPoint.y}
            stroke="rgba(143, 154, 168, 0.24)"
            strokeDasharray="9 7"
          />
        ) : null}

        <g>
          <line
            x1={xAxisRailLeft}
            y1={xAxisRailY}
            x2={xAxisRailRight}
            y2={xAxisRailY}
            stroke={X_AXIS_ACCENT}
            strokeWidth="2"
            strokeLinecap="round"
            opacity="0.92"
          />
          <path
            d={`M ${xAxisRailRight - 8} ${xAxisRailY - 4} L ${xAxisRailRight} ${xAxisRailY} L ${xAxisRailRight - 8} ${xAxisRailY + 4}`}
            fill="none"
            stroke={X_AXIS_ACCENT}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <rect
            x={plotCenterX - 54}
            y={xAxisRailY - 14}
            width="108"
            height="26"
            rx="13"
            fill={AXIS_CAPSULE_FILL}
            stroke={AXIS_CAPSULE_STROKE}
          />
          <text
            x={plotCenterX}
            y={xAxisRailY + 4}
            textAnchor="middle"
            fontSize="11.5"
            fontWeight="700"
            fontFamily="'IBM Plex Mono', 'JetBrains Mono', monospace"
            fill={X_AXIS_ACCENT}
          >
            X axis · meters
          </text>
          <text
            x={xAxisRailRight + 6}
            y={xAxisRailY + 4}
            textAnchor="start"
            fontSize="10.5"
            fontWeight="700"
            fontFamily="'IBM Plex Mono', 'JetBrains Mono', monospace"
            fill={X_AXIS_ACCENT}
          >
            +X
          </text>
        </g>

        <g>
          <line
            x1={zAxisRailX}
            y1={zAxisRailBottom}
            x2={zAxisRailX}
            y2={zAxisRailTop}
            stroke={Z_AXIS_ACCENT}
            strokeWidth="2"
            strokeLinecap="round"
            opacity="0.92"
          />
          <path
            d={`M ${zAxisRailX - 4} ${zAxisRailTop + 8} L ${zAxisRailX} ${zAxisRailTop} L ${zAxisRailX + 4} ${zAxisRailTop + 8}`}
            fill="none"
            stroke={Z_AXIS_ACCENT}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <rect
            x={zAxisRailX - 12}
            y={plotCenterY - 56}
            width="24"
            height="112"
            rx="12"
            fill={AXIS_CAPSULE_FILL}
            stroke={AXIS_CAPSULE_STROKE}
          />
          <text
            x={zAxisRailX}
            y={plotCenterY}
            textAnchor="middle"
            fontSize="11.5"
            fontWeight="700"
            fontFamily="'IBM Plex Mono', 'JetBrains Mono', monospace"
            fill={Z_AXIS_ACCENT}
            transform={`rotate(-90 ${zAxisRailX} ${plotCenterY})`}
          >
            Z axis · meters
          </text>
          <text
            x={zAxisRailX}
            y={zAxisRailTop - 8}
            textAnchor="middle"
            fontSize="10.5"
            fontWeight="700"
            fontFamily="'IBM Plex Mono', 'JetBrains Mono', monospace"
            fill={Z_AXIS_ACCENT}
          >
            +Z
          </text>
        </g>

        {renderableEntries.map((entry) => {
          const labelText = entry.label;
          const {
            center,
            footprintWidth,
            footprintHeight,
            footprintX,
            footprintY,
            labelX,
            labelY,
            calloutStartX,
            calloutStartY,
            calloutMidX,
            calloutEndX,
            calloutEndY,
          } = entry;
          const headingLength = Math.min(Math.max(Math.min(footprintWidth, footprintHeight) * 0.26, 8), 22);
          const headingX = entry.yaw !== null
            ? center.x + (Math.sin(entry.yaw) * headingLength)
            : center.x;
          const headingY = entry.yaw !== null
            ? center.y - (Math.cos(entry.yaw) * headingLength)
            : center.y - headingLength;
          const strokeDasharray = entry.footprintSource === 'fallback' ? '7 5' : undefined;

          return (
            <g key={entry.key}>
              <title>
                {`${entry.label} | ${formatMeters(entry.width)}m x ${formatMeters(entry.depth)}m | ${entry.footprintSource}`}
              </title>

              <rect
                x={footprintX}
                y={footprintY + 5}
                width={footprintWidth}
                height={footprintHeight}
                rx={Math.min(Math.min(footprintWidth, footprintHeight) * 0.18, 18)}
                fill="rgba(0, 0, 0, 0.18)"
              />

              <rect
                x={footprintX}
                y={footprintY}
                width={footprintWidth}
                height={footprintHeight}
                rx={Math.min(Math.min(footprintWidth, footprintHeight) * 0.18, 18)}
                fill={entry.colors.fill}
                stroke={entry.colors.stroke}
                strokeWidth="2"
                strokeDasharray={strokeDasharray}
                filter="url(#layout-pose-map-shadow)"
              />

              <rect
                x={footprintX + 2}
                y={footprintY + 2}
                width={Math.max(footprintWidth - 4, 0)}
                height={Math.max(footprintHeight - 4, 0)}
                rx={Math.min(Math.min(footprintWidth, footprintHeight) * 0.14, 14)}
                fill={entry.colors.wash}
                stroke="rgba(255, 255, 255, 0.28)"
                strokeWidth="1"
              />

              <circle cx={center.x} cy={center.y} r="3.5" fill={entry.colors.stroke} />
              <line
                x1={center.x}
                y1={center.y}
                x2={headingX}
                y2={headingY}
                stroke={entry.colors.stroke}
                strokeWidth="2"
                strokeLinecap="round"
                opacity="0.8"
              />

              <path
                d={`M ${calloutStartX} ${calloutStartY} L ${calloutMidX} ${calloutStartY} L ${calloutEndX} ${calloutEndY}`}
                fill="none"
                stroke={entry.colors.accent}
                strokeWidth="1.6"
                strokeDasharray="5 6"
                strokeLinecap="round"
                opacity="0.88"
              />
              <circle
                cx={calloutEndX}
                cy={calloutEndY}
                r="4"
                fill={entry.colors.stroke}
                stroke="rgba(255, 255, 255, 0.7)"
                strokeWidth="1"
              />
              <text
                x={labelX}
                y={labelY + 12}
                fontSize="11.5"
                fontWeight="700"
                fontFamily="'IBM Plex Sans', 'Geist', system-ui, sans-serif"
                fill="#f4f7fb"
                dominantBaseline="middle"
              >
                {labelText}
              </text>
            </g>
          );
        })}

        <rect
          x={LEGEND_BOUNDS.left}
          y={LEGEND_BOUNDS.top}
          width={LEGEND_BOUNDS.right - LEGEND_BOUNDS.left}
          height={LEGEND_BOUNDS.bottom - LEGEND_BOUNDS.top}
          rx="18"
          fill="rgba(255, 255, 255, 0.018)"
          stroke="rgba(121, 142, 170, 0.2)"
        />
        <text
          x={LEGEND_BOUNDS.left + 16}
          y={LEGEND_BOUNDS.top - 10}
          fontSize="11"
          fontWeight="700"
          fontFamily="'IBM Plex Mono', 'JetBrains Mono', monospace"
          fill={AXIS_TEXT}
        >
          Object callouts
        </text>

      </svg>

      {showLegend ? (
        <div style={MAP_GUIDE_STYLE}>
          <div style={MAP_GUIDE_HEAD_STYLE}>
            <strong style={{ color: 'var(--ink)' }}>Map reading guide</strong>
            <div style={MAP_GUIDE_META_STYLE}>
              <span style={MAP_GUIDE_PILL_STYLE}>External dashed callouts</span>
              <span style={MAP_GUIDE_PILL_STYLE}>Hover footprints for size + source</span>
            </div>
          </div>
          <p style={MAP_GUIDE_TEXT_STYLE}>
            Object labels now sit outside the plot and connect back with dashed guide lines, so the footprint field stays readable even when objects are tightly packed.
            Colors remain consistent per object, while the top chips summarize inferred and fallback footprints.
          </p>
        </div>
      ) : null}
    </div>
  );
}
