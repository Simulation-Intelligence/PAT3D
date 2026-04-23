import { apiUrl } from './api';

export const stageDefinitions = [
  {
    id: 'reference-image',
    number: '01',
    label: 'Reference image',
    functionality: 'Generate or load the scene reference image that anchors all downstream reasoning.',
    paperIntent: 'This corresponds to the image-anchored scene setup step used to ground subsequent structure extraction and asset synthesis.',
  },
  {
    id: 'scene-understanding',
    number: '02',
    label: 'Scene understanding',
    functionality: 'Estimate depth, segment objects, and build the initial object inventory from the reference image.',
    paperIntent: 'This is the image-to-structure stage of the paper pipeline.',
  },
  {
    id: 'object-relation',
    number: '03',
    label: 'Object description and relations',
    functionality: 'Infer object descriptions, size priors, and support or spatial relations.',
    paperIntent: 'This is the structured reasoning stage that turns raw vision outputs into compositional scene constraints.',
  },
  {
    id: 'object-assets',
    number: '04',
    label: 'Object asset generation',
    functionality: 'Generate textured 3D assets for each object in the catalog.',
    paperIntent: 'This matches the paper’s object synthesis stage.',
  },
  {
    id: 'layout-initialization',
    number: '05',
    label: 'Layout initialization',
    functionality: 'Assemble objects into an initial global layout using support relations and estimated sizes.',
    paperIntent: 'This is the global scene assembly stage before mesh preparation and export.',
  },
  {
    id: 'simulation-preparation',
    number: '06',
    label: 'Simulation preparation',
    functionality: 'Convert object meshes into simulation-ready low-poly or tet representations and package collision settings.',
    paperIntent: 'This bridges pure geometry into the prepared representation consumed by the forward physical simulation stage.',
  },
  {
    id: 'physics-optimization',
    number: '07',
    label: 'Physics simulation',
    functionality: 'Run the paper-aligned forward Diff GIPC / IPC simulation, optionally preceded by diff-sim initialization.',
    paperIntent: 'This is the paper’s physical settling stage, with an optional backward diff-sim pass before the final forward rollout.',
  },
  {
    id: 'visualization',
    number: '08',
    label: 'Visualization and export',
    functionality: 'Render the final scene state and expose exported artifacts for inspection.',
    paperIntent: 'This is the final reporting and output stage.',
  },
];

const jobStateMeta = {
  queued: {
    label: 'Queued',
    tone: 'pending',
    description: 'The run is waiting for a worker slot.',
  },
  running: {
    label: 'Running',
    tone: 'running',
    description: 'The pipeline is actively executing stages.',
  },
  awaiting_mask_input: {
    label: 'Awaiting masks',
    tone: 'partial',
    description: 'The reference image is ready. Add hand-tuned masks to continue stage 2.',
  },
  awaiting_size_input: {
    label: 'Awaiting sizes',
    tone: 'partial',
    description: 'Stage 03 is waiting for manual size priors or a retry.',
  },
  completed: {
    label: 'Completed',
    tone: 'aligned',
    description: 'All stages finished successfully.',
  },
  failed: {
    label: 'Failed',
    tone: 'fallback',
    description: 'The run stopped on an error and needs attention.',
  },
  cancelled: {
    label: 'Cancelled',
    tone: 'partial',
    description: 'The run was stopped manually.',
  },
  idle: {
    label: 'Idle',
    tone: 'pending',
    description: 'No run is currently selected.',
  },
};

const stageExecutionMeta = {
  completed: {
    label: 'Completed',
    tone: 'aligned',
  },
  running: {
    label: 'Running',
    tone: 'running',
  },
  awaiting_input: {
    label: 'Needs input',
    tone: 'partial',
  },
  failed: {
    label: 'Failed',
    tone: 'fallback',
  },
  cancelled: {
    label: 'Cancelled',
    tone: 'partial',
  },
  queued: {
    label: 'Queued',
    tone: 'pending',
  },
  pending: {
    label: 'Pending',
    tone: 'pending',
  },
};

export function isLiveJobState(state) {
  return ['queued', 'running', 'awaiting_mask_input', 'awaiting_size_input'].includes(String(state || ''));
}

function asNumber(value) {
  const parsed = Number.parseFloat(value);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  return parsed;
}

function safeProgress(value) {
  const parsed = asNumber(value);
  if (parsed === null) {
    return null;
  }
  return Math.max(0, Math.min(100, Math.round(parsed)));
}

function parseStageProgressValue(row) {
  const direct = safeProgress(row?.progress);
  if (direct !== null) {
    return direct;
  }

  const completed = row?.progress_completed;
  const total = row?.progress_total;
  const parsedCompleted = asNumber(completed);
  const parsedTotal = asNumber(total);
  if (parsedCompleted === null || parsedTotal === null || parsedTotal <= 0) {
    return null;
  }
  return Math.max(0, Math.min(100, Math.round((parsedCompleted / parsedTotal) * 100)));
}

let artifactVersion = '';

export function asList(value) {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

export function safeText(value, fallback = 'Unavailable') {
  if (value === null || value === undefined || value === '') return fallback;
  return String(value);
}

function collapseRepeatedObjectSuffix(value) {
  const raw = String(value || '').trim();
  if (!raw) return '';
  const parts = raw.split(/[_-]+/u).filter(Boolean);
  for (let size = Math.floor(parts.length / 2); size >= 1; size -= 1) {
    const trailing = parts.slice(-size);
    const preceding = parts.slice(-2 * size, -size);
    if (trailing.length !== size || preceding.length !== size) {
      continue;
    }
    if (preceding.join('\0') !== trailing.join('\0')) {
      continue;
    }
    const collapsed = trailing.join('_');
    if (/^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/iu.test(collapsed) && /\d/u.test(collapsed)) {
      return collapsed;
    }
  }
  return raw;
}

export function formatObjectDisplayName(value, fallback = 'Unavailable') {
  const raw = String(value || '').trim();
  if (!raw) return fallback;

  let display = raw.includes(':')
    ? raw.split(':').filter(Boolean).pop() || raw
    : raw;

  display = collapseRepeatedObjectSuffix(display);
  display = display.replace(/[_-]+/g, ' ').trim();
  display = display.replace(/\s+/g, ' ');
  display = display.replace(/(.+?)\s+(\d+)$/u, '$1$2');

  return display || fallback;
}

export function formatObjectDisplayList(values, fallback = 'none') {
  const normalized = asList(values).filter(Boolean).map((value) => formatObjectDisplayName(value, '')).filter(Boolean);
  return normalized.length ? normalized.join(', ') : fallback;
}

export function setArtifactVersion(version) {
  artifactVersion = version ? String(version) : '';
}

export function artifactUrl(path) {
  if (!path) return null;
  const params = new URLSearchParams({ path });
  if (artifactVersion) {
    params.set('v', artifactVersion);
  }
  return apiUrl(`/api/artifact?${params.toString()}`);
}

function extractReference(runtime) {
  return runtime?.first_contract_slice?.reference_image_result || runtime?.layout_initialization?.reference_image_result || null;
}

function extractUnderstanding(runtime) {
  return runtime?.first_contract_slice?.scene_understanding || null;
}

function extractRelation(runtime) {
  return runtime?.first_contract_slice?.object_relation || null;
}

function extractObjectAssets(runtime) {
  return runtime?.object_assets?.object_assets || runtime?.layout_initialization?.object_assets || null;
}

function extractSceneBundleAssets(runtime) {
  return runtime?.layout_initialization?.object_assets
    || runtime?.simulation_preparation?.object_assets
    || runtime?.object_assets?.object_assets
    || null;
}

function extractPreferredSceneMeshes(runtime) {
  const physicsMeshes = asList(runtime?.physics_optimization?.optimization_result?.initial_scene?.simulation_meshes)
    .filter((artifact) => artifact?.path);
  if (physicsMeshes.length) {
    return {
      source: 'runtime_simulation_meshes',
      artifacts: physicsMeshes,
    };
  }

  const simulationMeshes = asList(runtime?.simulation_preparation?.physics_ready_scene?.simulation_meshes)
    .filter((artifact) => artifact?.path);
  if (simulationMeshes.length) {
    return {
      source: 'runtime_simulation_meshes',
      artifacts: simulationMeshes,
    };
  }

  const layoutMeshArtifacts = asList(runtime?.layout_initialization?.scene_layout?.artifacts)
    .filter((artifact) => artifact?.artifact_type === 'layout_mesh' && artifact?.path);
  if (layoutMeshArtifacts.length) {
    return {
      source: 'runtime_layout_fallback',
      artifacts: layoutMeshArtifacts,
    };
  }

  return {
    source: null,
    artifacts: [],
  };
}

export function extractLayout(runtime) {
  return runtime?.layout_initialization?.scene_layout || runtime?.simulation_preparation?.scene_layout || null;
}

function extractSimulation(runtime) {
  return runtime?.simulation_preparation?.physics_ready_scene || null;
}

function extractPhysics(runtime) {
  return runtime?.physics_optimization?.optimization_result || null;
}

export function extractPhysicsDebugReportPath(runtime) {
  const physics = extractPhysics(runtime);
  const debugArtifact = asList(physics?.artifacts).find((artifact) => artifact?.artifact_type === 'physics_debug_report' && artifact?.path);
  return debugArtifact?.path || null;
}

function extractRender(runtime) {
  return runtime?.visualization?.render_result || null;
}

function extractFinalObjectPoses(runtime) {
  const physics = extractPhysics(runtime);
  if (asList(physics?.optimized_object_poses).length) {
    return asList(physics.optimized_object_poses);
  }

  const simulation = extractSimulation(runtime);
  if (asList(simulation?.object_poses).length) {
    return asList(simulation.object_poses);
  }

  return asList(extractLayout(runtime)?.object_poses);
}

function poseTransformForBundle(pose) {
  if (!pose || !pose.object_id) return null;
  return {
    translation_xyz: asList(pose.translation_xyz || pose.position).slice(0, 3),
    rotation_type: pose.rotation_type || 'quaternion',
    rotation_value: asList(pose.rotation_value || pose.rotation || [1, 0, 0, 0]),
    scale_xyz: asList(pose.scale_xyz || pose.scale || [1, 1, 1]).slice(0, 3),
  };
}

function assetMetadataNotes(asset) {
  return asList(asset?.metadata?.notes).filter((note) => typeof note === 'string');
}

function assetNoteValue(asset, key) {
  const prefix = `${key}=`;
  const matchingNote = assetMetadataNotes(asset).find((note) => note.startsWith(prefix));
  return matchingNote ? matchingNote.slice(prefix.length) : '';
}

function normalizedObjectIdCandidates(value) {
  const raw = String(value || '').trim();
  if (!raw) {
    return [];
  }

  const candidates = new Set([raw.toLowerCase()]);
  if (raw.includes(':')) {
    const lastSegment = raw.split(':').filter(Boolean).pop();
    if (lastSegment) {
      candidates.add(lastSegment.toLowerCase());
    }
  }

  let current = raw;
  while (true) {
    const stripped = current.replace(/(?:(?:::)|[_-]|\b)(\d+)$/u, '').replace(/[:_-]+$/u, '');
    if (!stripped || stripped === current) {
      break;
    }
    current = stripped;
    candidates.add(current.toLowerCase());
    if (current.includes(':')) {
      const lastSegment = current.split(':').filter(Boolean).pop();
      if (lastSegment) {
        candidates.add(lastSegment.toLowerCase());
      }
    }
  }

  return Array.from(candidates);
}

function recordIdCandidates(record, extraIds = []) {
  return [
    ...normalizedObjectIdCandidates(record?.object_id),
    ...extraIds.flatMap((value) => normalizedObjectIdCandidates(value)),
  ];
}

function resolveRecordByObjectId(records, objectId, extraIdsSelector = () => []) {
  const exactId = String(objectId || '').trim();
  if (!exactId) {
    return null;
  }

  const exactMatch = records.find((record) => String(record?.object_id || '').trim() === exactId);
  if (exactMatch) {
    return exactMatch;
  }

  const objectCandidates = normalizedObjectIdCandidates(exactId);
  const matches = records.filter((record) => {
    const recordCandidates = recordIdCandidates(record, asList(extraIdsSelector(record)));
    return objectCandidates.some((candidate) => recordCandidates.includes(candidate));
  });
  return matches.length === 1 ? matches[0] : null;
}

function assetMeshAlreadyInSceneSpace(asset) {
  return assetMetadataNotes(asset).includes('mesh_pose_space=scene');
}

export function getSceneId(runtime) {
  return extractLayout(runtime)?.scene_id
    || extractSimulation(runtime)?.scene_id
    || extractPhysics(runtime)?.scene_id
    || extractRender(runtime)?.scene_id
    || extractObjectAssets(runtime)?.scene_id
    || 'unknown';
}

function resolveRuntimeGroundPlane(runtime) {
  const physics = extractPhysics(runtime);
  const simulation = extractSimulation(runtime);
  const simulationGroundY = asNumber(simulation?.collision_settings?.ground_y_value);

  if (physics) {
    const metrics = physics?.metrics || {};
    const requestedGroundY = asNumber(metrics.requested_ground_y_value);
    const appliedGroundY = asNumber(metrics.applied_ground_y_value);
    if (requestedGroundY !== null || appliedGroundY !== null) {
      const resolvedRequestedGroundY = requestedGroundY ?? simulationGroundY ?? -1.1;
      const resolvedAppliedGroundY = appliedGroundY ?? resolvedRequestedGroundY;
      return {
        requested_ground_plane_y: resolvedRequestedGroundY,
        applied_ground_plane_y: resolvedAppliedGroundY,
        ground_plane_source: 'physics_metrics',
      };
    }
    if (simulationGroundY !== null) {
      return {
        requested_ground_plane_y: simulationGroundY,
        applied_ground_plane_y: simulationGroundY,
        ground_plane_source: 'physics_collision_settings',
      };
    }
    return {
      requested_ground_plane_y: -1.1,
      applied_ground_plane_y: -1.1,
      ground_plane_source: 'physics_legacy_default',
    };
  }

  if (simulation) {
    if (simulationGroundY !== null) {
      return {
        requested_ground_plane_y: simulationGroundY,
        applied_ground_plane_y: simulationGroundY,
        ground_plane_source: 'physics_collision_settings',
      };
    }
    return {
      requested_ground_plane_y: -1.1,
      applied_ground_plane_y: -1.1,
      ground_plane_source: 'physics_legacy_default',
    };
  }

  return {
    requested_ground_plane_y: null,
    applied_ground_plane_y: null,
    ground_plane_source: null,
  };
}

export function objectInventoryRows(objectCatalog) {
  return asList(objectCatalog?.objects).map((object) => [
    formatObjectDisplayName(object.object_id),
    formatObjectDisplayName(object.display_name || object.canonical_name),
    safeText(object.count, '1'),
    asList(object.source_instance_ids).length
      ? asList(object.source_instance_ids).map((value) => formatObjectDisplayName(value)).join(', ')
      : 'n/a',
  ]);
}

function normalizedInstanceIdsForCatalogObject(object) {
  const count = Number.parseInt(String(object?.count ?? ''), 10);
  const normalizedCount = Number.isFinite(count) && count > 1 ? count : 1;
  const objectId = safeText(object?.object_id);
  const sourceIds = asList(object?.source_instance_ids).map((value) => safeText(value)).filter(Boolean);
  if (normalizedCount <= 1) {
    return objectId ? [objectId] : [];
  }
  return Array.from({ length: normalizedCount }, (_, index) => sourceIds[index] || `${objectId}::${index + 1}`);
}

function expandSizePriorsForDisplay(sizePriors, objectCatalog) {
  const priors = asList(sizePriors);
  const objects = asList(objectCatalog?.objects);
  if (!objects.some((object) => Number.parseInt(String(object?.count ?? ''), 10) > 1)) {
    return priors;
  }
  const priorById = new Map(priors.map((prior) => [safeText(prior?.object_id), prior]));
  const expanded = [];
  const consumed = new Set();
  objects.forEach((object) => {
    const objectId = safeText(object?.object_id);
    const prior = priorById.get(objectId);
    if (!prior) return;
    consumed.add(objectId);
    const instanceIds = normalizedInstanceIdsForCatalogObject(object);
    instanceIds.forEach((instanceId) => {
      expanded.push(instanceId === objectId ? prior : { ...prior, object_id: instanceId });
    });
  });
  priors.forEach((prior) => {
    if (!consumed.has(safeText(prior?.object_id))) {
      expanded.push(prior);
    }
  });
  return expanded;
}

export function sizePriorRows(sizePriors, objectCatalog = null) {
  return expandSizePriorsForDisplay(sizePriors, objectCatalog).map((prior) => {
    const dimensions = prior?.dimensions_m || {};
    return [
      formatObjectDisplayName(prior.object_id),
      safeText(dimensions.x, 'n/a'),
      safeText(dimensions.y, 'n/a'),
      safeText(dimensions.z, 'n/a'),
    ];
  });
}

export function collectArtifactPaths(runtime) {
  const results = new Set();

  function walk(node) {
    if (!node) return;
    if (Array.isArray(node)) {
      node.forEach(walk);
      return;
    }
    if (typeof node === 'object') {
      if (typeof node.path === 'string') {
        results.add(node.path);
      }
      Object.values(node).forEach(walk);
    }
  }

  walk(runtime);
  return [...results];
}

export function evaluateStageStatuses(runtime) {
  const layoutProvider = extractLayout(runtime)?.metadata?.provider_name || '';
  const physics = extractPhysics(runtime);
  const physicsProvider = physics?.metadata?.provider_name || '';
  const physicsNotes = asList(physics?.metadata?.notes).map((note) => safeText(note, ''));
  const physicsMetrics = physics?.metrics || {};
  const renderProvider = extractRender(runtime)?.metadata?.provider_name || '';
  const physicsIsFallback = physicsNotes.some((note) => (
    note.includes('identity_passthrough')
      || note.includes('fallback_reason=')
      || note.includes('backend_failure')
      || note.includes('forward_diff_simulator_failed')
      || note.includes('zero_frame_forward_result')
  )) || Number(physicsMetrics.forward_diff_simulator_failed || 0) > 0
    || Number(physicsMetrics.zero_frame_forward_result || 0) > 0;

  return {
    reference: 'aligned',
    understanding: 'aligned',
    relation: 'aligned',
    assets: 'aligned',
    layout: layoutProvider.includes('legacy') ? 'partial' : 'aligned',
    simulation: 'aligned',
    physics: physicsProvider === '' || physicsIsFallback ? 'partial' : 'aligned',
    visualization: renderProvider === 'legacy_renderer_fallback' ? 'partial' : 'aligned',
  };
}

export function getStageModels(runtime) {
  const statuses = evaluateStageStatuses(runtime);
  const reference = extractReference(runtime);
  const understanding = extractUnderstanding(runtime);
  const relation = extractRelation(runtime);
  const objectAssets = extractObjectAssets(runtime);
  const assetEntries = asList(objectAssets?.assets);
  const layout = extractLayout(runtime);
  const simulation = extractSimulation(runtime);
  const physics = extractPhysics(runtime);
  const render = extractRender(runtime);

  return {
    reference: {
      status: statuses.reference,
      providerName: reference?.metadata?.provider_name,
      notes: asList(reference?.metadata?.notes),
      data: reference,
    },
    understanding: {
      status: statuses.understanding,
      providerName: [
        understanding?.depth_result?.metadata?.provider_name,
        understanding?.segmentation_result?.metadata?.provider_name,
      ].filter(Boolean).join(' + '),
      notes: [
        ...asList(understanding?.depth_result?.metadata?.notes),
        ...asList(understanding?.segmentation_result?.metadata?.notes),
      ],
      data: understanding,
    },
    relation: {
      status: statuses.relation,
      providerName: relation?.relation_graph?.metadata?.provider_name || relation?.object_catalog?.metadata?.provider_name,
      notes: asList(relation?.relation_graph?.metadata?.notes),
      data: relation,
    },
    assets: {
      status: statuses.assets,
      providerName: assetEntries[0]?.metadata?.provider_name || objectAssets?.metadata?.provider_name,
      notes: assetEntries.flatMap((asset) => asList(asset?.metadata?.notes)),
      data: objectAssets,
    },
    layout: {
      status: statuses.layout,
      providerName: layout?.metadata?.provider_name,
      notes: asList(layout?.metadata?.notes),
      data: layout,
    },
    simulation: {
      status: statuses.simulation,
      providerName: simulation?.metadata?.provider_name,
      notes: asList(simulation?.metadata?.notes),
      data: simulation,
    },
    physics: {
      status: statuses.physics,
      providerName: physics?.metadata?.provider_name,
      notes: asList(physics?.metadata?.notes),
      data: physics,
    },
    visualization: {
      status: statuses.visualization,
      providerName: render?.metadata?.provider_name,
      notes: asList(render?.metadata?.notes),
      data: render,
    },
  };
}

function inferSiblingPath(path, extension) {
  if (!path || typeof path !== 'string' || !path.includes('.')) return null;
  return path.replace(/\.[^.]+$/, extension);
}

function inferObjectIdFromPath(path) {
  if (!path || typeof path !== 'string') return 'object';
  const fileName = path.split('/').filter(Boolean).pop() || path;
  return inferSiblingPath(fileName, '') || fileName || 'object';
}

function layoutArtifactObjects(layout) {
  return asList(layout?.artifacts)
    .filter((artifact) => artifact?.artifact_type === 'layout_mesh' && artifact?.path)
    .map((artifact) => ({
      object_id: artifact.role || inferObjectIdFromPath(artifact.path),
      mesh_obj_path: artifact.path,
      mesh_mtl_path: null,
      texture_image_path: null,
      already_transformed: true,
      transform: null,
    }));
}

export function buildSceneBundle(runtime, options = {}) {
  const layout = extractLayout(runtime);
  const sceneId = getSceneId(runtime);
  const render = extractRender(runtime);
  const meshSource = String(options?.meshSource || 'simplified').toLowerCase();
  const useOriginalMeshes = meshSource === 'original';
  const layoutObjects = layoutArtifactObjects(layout);
  const groundPlane = resolveRuntimeGroundPlane(runtime);
  const exportedSceneBundlePath = render?.camera_metadata?.role === 'scene_bundle' && render?.camera_metadata?.path
    ? render.camera_metadata.path
    : null;
  if (!useOriginalMeshes && exportedSceneBundlePath) {
    return {
      kind: 'artifact',
      path: exportedSceneBundlePath,
    };
  }

  if (!useOriginalMeshes && layoutObjects.length) {
    return {
      kind: 'inline',
      data: {
        scene_id: sceneId,
        source_scene_type: 'runtime_layout_artifacts',
        geometry_source_type: 'layout_mesh_artifacts',
        objects: layoutObjects,
      },
    };
  }

  const poses = extractFinalObjectPoses(runtime).filter((pose) => pose?.object_id);
  const sceneAssets = asList(extractSceneBundleAssets(runtime)?.assets)
    .filter((asset) => asset?.object_id && asset?.mesh_obj?.path)
    .filter(Boolean);
  const assetObjects = sceneAssets
    .map((asset) => {
      const alreadyTransformed = assetMeshAlreadyInSceneSpace(asset);
      const pose = resolveRecordByObjectId(poses, asset.object_id)
        || asList(asset.provider_asset_id).map((value) => resolveRecordByObjectId(poses, value)).find(Boolean)
        || null;
      const transform = alreadyTransformed ? null : poseTransformForBundle(pose);
      const meshObjPath = asset.mesh_obj.path;
      const canonicalMeshPath = assetNoteValue(asset, 'canonical_mesh_path');
      return {
        object_id: asset.object_id,
        mesh_obj_path: meshObjPath,
        mesh_mtl_path: asset.mesh_mtl?.path || null,
        texture_image_path: asset.texture_image?.path || null,
        canonical_mesh_path: canonicalMeshPath || null,
        already_transformed: alreadyTransformed || !transform,
        transform,
      };
    })
    .filter(Boolean);
  const preferredSceneMeshes = useOriginalMeshes
    ? { source: null, artifacts: [] }
    : extractPreferredSceneMeshes(runtime);
  const sceneMeshObjects = preferredSceneMeshes.artifacts
    .map((artifact, index) => {
      const object_id = artifact?.role || inferSiblingPath(artifact?.path, '') || `object_${index + 1}`;
      const asset = resolveRecordByObjectId(
        assetObjects,
        object_id,
        (candidate) => {
          const sourceAsset = sceneAssets.find((entry) => entry.object_id === candidate.object_id);
          return sourceAsset ? asList(sourceAsset.provider_asset_id) : [];
        },
      );
      const pose = resolveRecordByObjectId(poses, object_id)
        || (asset
          ? asList(sceneAssets.find((entry) => entry.object_id === asset.object_id)?.provider_asset_id)
            .map((value) => resolveRecordByObjectId(poses, value))
            .find(Boolean)
          : null)
        || null;
      const transform = poseTransformForBundle(pose);
      return {
        object_id,
        mesh_obj_path: artifact.path,
        mesh_mtl_path: asset?.mesh_mtl_path || null,
        texture_image_path: asset?.texture_image_path || null,
        canonical_mesh_path: asset?.canonical_mesh_path || null,
        already_transformed: (asset?.already_transformed || false) || !transform,
        transform,
      };
    })
    .filter((object) => typeof object.mesh_obj_path === 'string');

  const matchedObjectIds = new Set(sceneMeshObjects.map((object) => object.object_id));
  const assetObjectIds = new Set(assetObjects.map((object) => object.object_id));
  const fallbackLayoutObjects = layoutObjects.filter((object) => (
    !matchedObjectIds.has(object.object_id) && !assetObjectIds.has(object.object_id)
  ));

  if (useOriginalMeshes) {
    if (assetObjects.length || fallbackLayoutObjects.length) {
      return {
        kind: 'inline',
        data: {
          scene_id: sceneId,
          source_scene_type: assetObjects.length ? 'runtime_object_assets' : 'runtime_layout_fallback',
          geometry_source_type: assetObjects.length ? 'object_asset_meshes' : 'layout_mesh_artifacts',
          ...groundPlane,
          objects: [...assetObjects, ...fallbackLayoutObjects],
        },
      };
    }
    return null;
  }

  if (sceneMeshObjects.length || assetObjects.length || fallbackLayoutObjects.length) {
    const objects = sceneMeshObjects.length ? sceneMeshObjects : [...assetObjects, ...fallbackLayoutObjects];
    const geometrySourceType = sceneMeshObjects.length
      ? 'simulation_mesh_artifacts'
      : assetObjects.length
        ? 'object_asset_meshes'
        : 'layout_mesh_artifacts';
    return {
      kind: 'inline',
      data: {
        scene_id: sceneId,
        source_scene_type: sceneMeshObjects.length
          ? preferredSceneMeshes.source
          : (assetObjects.length ? 'runtime_object_assets' : 'runtime_layout_fallback'),
        geometry_source_type: geometrySourceType,
        ...groundPlane,
        objects,
      },
    };
  }

  if (exportedSceneBundlePath) {
    return {
      kind: 'artifact',
      path: exportedSceneBundlePath,
    };
  }

  if (!layout && !assetObjects.length) {
    return null;
  }
  return {
    kind: 'inline',
    data: {
      scene_id: sceneId,
      source_scene_type: 'runtime_layout_fallback',
      geometry_source_type: 'layout_mesh_artifacts',
      ...groundPlane,
      objects: fallbackLayoutObjects,
    },
  };
}

export function progressPercent(job) {
  const stages = asList(job?.stages);
  if (!stages.length) return 0;
  const fallbackRunningProgress = stages.some((stage) => stage.status === 'running' || stage.status === 'awaiting_input');

  let stagedProgress = 0;
  for (const stage of stages) {
    if (stage.status === 'completed') {
      stagedProgress += 100;
      continue;
    }

    if (stage.status === 'failed' || stage.status === 'cancelled') {
      const progress = safeProgress(stage.progress);
      stagedProgress += progress === null ? 0 : progress;
      continue;
    }

    const progress = parseStageProgressValue(stage);
    if (progress !== null) {
      stagedProgress += progress;
    } else if (stage.status === 'running' || stage.status === 'awaiting_input') {
      stagedProgress += fallbackRunningProgress ? 45 : 0;
    }
  }

  return Math.round(stagedProgress / stages.length);
}

function getJobStateMeta(state) {
  return jobStateMeta[state] || {
    label: safeText(state, 'Unknown'),
    tone: 'pending',
    description: 'The run reported an unknown state.',
  };
}

function getStageExecutionMeta(status) {
  return stageExecutionMeta[status] || {
    label: safeText(status, 'Unknown'),
    tone: 'pending',
  };
}

function parseDateValue(value) {
  const timestamp = Date.parse(value || '');
  return Number.isFinite(timestamp) ? timestamp : null;
}

export function formatTimestamp(value) {
  const timestamp = parseDateValue(value);
  if (timestamp === null) return 'n/a';
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(timestamp));
}

export function formatDuration(startValue, endValue) {
  const start = parseDateValue(startValue);
  const end = parseDateValue(endValue);
  if (start === null || end === null || end < start) return 'n/a';

  const totalSeconds = Math.round((end - start) / 1000);
  if (totalSeconds < 60) {
    return `${totalSeconds}s`;
  }

  const totalMinutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (totalMinutes < 60) {
    return seconds ? `${totalMinutes}m ${seconds}s` : `${totalMinutes}m`;
  }

  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return minutes ? `${hours}h ${minutes}m` : `${hours}h`;
}

export function summarizeJobProgress(job) {
  const stages = asList(job?.stages).map((stage, index) => {
    const definition = stageDefinitions.find((candidate) => candidate.id === stage.id) || stageDefinitions[index] || {};
    const execution = getStageExecutionMeta(stage.status);
    return {
      ...definition,
      ...stage,
      number: definition.number || String(index + 1).padStart(2, '0'),
      label: stage.label || definition.label || `Stage ${index + 1}`,
      functionality: definition.functionality || '',
      execution,
    };
  });

  const counts = {
    completed: stages.filter((stage) => stage.status === 'completed').length,
    running: stages.filter((stage) => stage.status === 'running').length,
    awaiting_input: stages.filter((stage) => stage.status === 'awaiting_input').length,
    failed: stages.filter((stage) => stage.status === 'failed').length,
    cancelled: stages.filter((stage) => stage.status === 'cancelled').length,
    queued: stages.filter((stage) => stage.status === 'queued').length,
    pending: stages.filter((stage) => stage.status === 'pending').length,
  };

  const activeStage = stages.find((stage) => stage.status === 'running')
    || stages.find((stage) => stage.status === 'awaiting_input')
    || (isLiveJobState(job?.state) ? stages.find((stage) => stage.status === 'queued') : null)
    || stages.find((stage) => stage.status === 'failed')
    || stages.find((stage) => stage.status === 'cancelled')
    || null;

  const lastCompletedStage = [...stages].reverse().find((stage) => stage.status === 'completed') || null;
  const startedAt = job?.started_at || stages.find((stage) => stage.started_at)?.started_at || null;
  const finishedAt = job?.finished_at || null;
  const updatedAt = job?.updated_at || finishedAt || startedAt || null;
  const total = stages.length;
  const inFlight = counts.running + counts.queued + counts.awaiting_input;
  const waiting = counts.pending;
  const percent = progressPercent({ stages });

  return {
    stages,
    counts,
    total,
    percent,
    inFlight,
    waiting,
    isLive: isLiveJobState(job?.state),
    activeStage,
    lastCompletedStage,
    startedAt,
    finishedAt,
    updatedAt,
    durationLabel: formatDuration(startedAt, finishedAt || updatedAt),
    stateMeta: getJobStateMeta(job?.state || 'idle'),
  };
}
