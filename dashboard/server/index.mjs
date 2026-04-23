import { spawn } from 'node:child_process';
import express from 'express';
import fs from 'node:fs/promises';
import { closeSync, existsSync, openSync, readFileSync, writeSync } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import crypto from 'node:crypto';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const appRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(appRoot, '..');
const envFilePath = path.join(repoRoot, '.env');
const loadDotenv = () => {
  if (!existsSync(envFilePath)) {
    return;
  }

  const content = readFileSync(envFilePath, 'utf8');
  for (const line of content.split(/\r?\n/)) {
    const trimmedLine = line.trim();
    if (!trimmedLine || trimmedLine.startsWith('#')) {
      continue;
    }
    const parsed = trimmedLine.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
    if (!parsed) {
      continue;
    }
    const [, key, rawValue] = parsed;
    if (!key || process.env[key] !== undefined) {
      continue;
    }

    let value = rawValue;
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (!value) {
      continue;
    }
    process.env[key] = value;
  }
};

loadDotenv();

const runtimeDir = path.join(repoRoot, 'results', 'runtime');
const jobDir = path.join(repoRoot, 'results', 'dashboard_jobs');
const metricsDir = path.join(repoRoot, 'results', 'metrics');
const workspaceResultsDir = path.join(repoRoot, 'results', 'workspaces');
const renderedImagesDir = path.join(repoRoot, 'results', 'rendered_images');
const rawResponsesDir = path.join(runtimeDir, 'raw_responses');
const tmpDir = path.join(repoRoot, 'tmp');
const pythonExecutable = process.env.PAT3D_DASHBOARD_PYTHON || path.join(repoRoot, '.conda', 'pat3d', 'bin', 'python');
const sam3PythonExecutable = resolveSam3PythonExecutable();
const jobRunnerScript = path.join(repoRoot, 'pat3d', 'scripts', 'dashboard', 'dashboard_run_job.py');
const trajectoryScript = path.join(repoRoot, 'pat3d', 'scripts', 'dashboard', 'extract_sim_trajectory.py');
const metricsScript = path.join(repoRoot, 'pat3d', 'metrics', 'semantics', 'dashboard_metrics.py');
const physicsMetricsScript = path.join(repoRoot, 'pat3d', 'scripts', 'dashboard', 'extract_physics_metrics.py');
const sceneExportScript = path.join(repoRoot, 'pat3d', 'scripts', 'physics', 'export_scene_bundle.py');
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
const REASONING_EFFORT_OPTIONS = ['auto', 'low', 'medium', 'high'];
const RELATION_TYPE_OPTIONS = ['supports', 'contains', 'on', 'in'];
const DEFAULT_PHYSICS_SETTINGS = {
  diffSimEnabled: false,
  endFrame: 300,
  groundYValue: -1.1,
  totalOptEpoch: 50,
  physLr: 0.001,
  contactDHat: 5e-4,
  contactEpsVelocity: 1e-5,
};
const STAGE_BACKEND_PROFILE_OPTIONS = ['default', 'host-compatible'];
const STAGE_BACKEND_PROFILE_PATHS = {
  default: path.join(appRoot, 'src', 'stageBackends.json'),
  'host-compatible': path.join(appRoot, 'src', 'stageBackends.host-compatible.json'),
};
const MAX_RUNTIME_FILES = 5;
const allowedArtifactRoots = new Set(['results', 'data', 'vis', 'visualize', '_phys_result']);
const isProduction = process.env.NODE_ENV === 'production';
const port = Number(process.env.PORT || 4173);
const host = process.env.HOST || '0.0.0.0';
const DEFAULT_STAGE_BACKENDS_PROFILE = sanitizeStageBackendsProfile(
  process.env.PAT3D_DASHBOARD_STAGE_BACKENDS_PROFILE || process.env.PAT3D_STAGE_BACKENDS_PROFILE,
  'default',
);
const FORCE_STAGE_BACKENDS_PROFILE = /^(1|true|yes|on)$/i.test(
  String(process.env.PAT3D_DASHBOARD_FORCE_STAGE_BACKENDS_PROFILE || '').trim(),
);
const runningChildren = new Map();
const JOB_STAGE_BLUEPRINTS = [
  { id: 'reference-image', label: 'Reference image' },
  { id: 'scene-understanding', label: 'Scene understanding' },
  { id: 'object-relation', label: 'Object description and relations' },
  { id: 'object-assets', label: 'Object asset generation' },
  { id: 'layout-initialization', label: 'Layout initialization' },
  { id: 'simulation-preparation', label: 'Simulation preparation' },
  { id: 'physics-optimization', label: 'Physics simulation' },
  { id: 'visualization', label: 'Visualization and export' },
];

const app = express();
app.disable('x-powered-by');
app.set('etag', false);
app.use(express.json({ limit: '20mb' }));
app.use('/api', (_req, res, next) => {
  res.set('Cache-Control', 'no-store, max-age=0');
  res.set('Pragma', 'no-cache');
  res.set('Expires', '0');
  next();
});

function resolveSam3PythonExecutable() {
  const configured = typeof process.env.PAT3D_SAM3_PYTHON === 'string'
    ? process.env.PAT3D_SAM3_PYTHON.trim()
    : '';
  if (configured) {
    return configured;
  }

  const homeDir = process.env.HOME || '';
  const candidates = [
    path.join(homeDir, 'anaconda3', 'envs', 'pat3d-sam3', 'bin', 'python'),
    path.join(homeDir, '.conda', 'envs', 'pat3d-sam3', 'bin', 'python'),
    path.join(homeDir, 'miniconda3', 'envs', 'pat3d-sam3', 'bin', 'python'),
    path.join(repoRoot, '.venv-sam3', 'bin', 'python'),
  ];
  for (const candidate of candidates) {
    if (candidate && existsSync(candidate)) {
      return candidate;
    }
  }
  return '';
}

function runPythonJsonScript(pythonPath, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(
      pythonPath,
      args,
      {
        cwd: repoRoot,
        env: {
          ...process.env,
          ...(options.env || {}),
        },
        stdio: ['ignore', 'pipe', 'pipe'],
      },
    );
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on('data', (chunk) => {
      stderr += String(chunk);
    });
    child.on('error', (error) => {
      reject(error);
    });
    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error((stderr || stdout || `exit code ${code}`).trim()));
        return;
      }
      try {
        resolve(parseJsonScriptOutput(stdout, path.basename(args[0] || 'python script')));
      } catch (error) {
        reject(error);
      }
    });
  });
}

function parseJsonScriptOutput(stdout, scriptName) {
  const trimmed = String(stdout || '').trim();
  if (!trimmed) {
    throw new Error(`invalid JSON from ${scriptName}: empty stdout`);
  }

  try {
    return JSON.parse(trimmed);
  } catch {
    const lines = trimmed.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      try {
        return JSON.parse(lines[index]);
      } catch {
        // Keep walking backwards until a valid JSON line is found.
      }
    }
  }

  const tail = trimmed.slice(-400);
  throw new Error(`invalid JSON from ${scriptName}: no parseable JSON payload in stdout tail: ${tail}`);
}

function normalizeRemoteAddress(remoteAddress) {
  if (!remoteAddress || typeof remoteAddress !== 'string') {
    return '';
  }

  if (remoteAddress === '::1') {
    return '127.0.0.1';
  }

  if (remoteAddress.startsWith('::ffff:')) {
    return remoteAddress.slice(7);
  }

  return remoteAddress;
}

function parseEnabledFlag(value) {
  const normalized = String(value || '').trim().toLowerCase();
  return normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on';
}

function isPrivateSubnetAddress(remoteAddress) {
  if (!remoteAddress || typeof remoteAddress !== 'string') {
    return false;
  }
  if (remoteAddress.startsWith('10.')) {
    return true;
  }
  if (remoteAddress.startsWith('192.168.')) {
    return true;
  }
  const match = remoteAddress.match(/^172\.(\d{1,3})\./);
  if (!match) {
    return false;
  }
  const octet = Number.parseInt(match[1], 10);
  return Number.isInteger(octet) && octet >= 16 && octet <= 31;
}

const allowPrivateSubnetAccess = parseEnabledFlag(process.env.PAT3D_DASHBOARD_ALLOW_PRIVATE_SUBNETS);

function isAllowedRemoteAddress(remoteAddress) {
  const normalized = normalizeRemoteAddress(remoteAddress);
  if (normalized === '127.0.0.1') {
    return true;
  }
  if (normalized.startsWith('192.168.')) {
    return true;
  }
  return allowPrivateSubnetAccess && isPrivateSubnetAddress(normalized);
}

function resolveArtifactPath(relativePath) {
  if (!relativePath || typeof relativePath !== 'string') {
    return null;
  }

  const normalizedInput = path.normalize(relativePath);
  const candidateAbsolutePath = path.isAbsolute(normalizedInput)
    ? path.resolve(normalizedInput)
    : path.resolve(repoRoot, normalizedInput.replace(/^([.][.][/\\])+/, ''));
  if (!candidateAbsolutePath.startsWith(repoRoot + path.sep) && candidateAbsolutePath !== repoRoot) {
    return null;
  }

  const repoRelativePath = path.relative(repoRoot, candidateAbsolutePath);
  const normalized = path.normalize(repoRelativePath);
  const firstSegment = normalized.split(path.sep)[0];
  if (!allowedArtifactRoots.has(firstSegment)) {
    return null;
  }
  return candidateAbsolutePath;
}

function repoRelativeArtifactPath(relativePath) {
  const absolutePath = resolveArtifactPath(relativePath);
  if (!absolutePath) {
    return null;
  }
  return path.relative(repoRoot, absolutePath);
}

function sanitizeNumericArray(value, length, fallbackValues) {
  const values = Array.isArray(value) ? value.slice(0, length) : fallbackValues;
  return values.map((entry, index) => {
    const numeric = Number.parseFloat(String(entry ?? fallbackValues[index] ?? 0));
    return Number.isFinite(numeric) ? numeric : fallbackValues[index] ?? 0;
  });
}

function sanitizeOptionalNumber(value) {
  if (value === null || value === undefined || value === '') {
    return null;
  }
  const numeric = Number.parseFloat(String(value));
  return Number.isFinite(numeric) ? numeric : null;
}

function sanitizeSceneBundleForExport(bundle) {
  const raw = bundle && typeof bundle === 'object' && !Array.isArray(bundle) ? bundle : null;
  if (!raw) {
    throw new ApiError(
      422,
      'scene_bundle_invalid',
      'The stage-8 scene bundle is missing or invalid.',
      'scene bundle must be an object',
    );
  }

  const sceneId = typeof raw.scene_id === 'string' && raw.scene_id.trim()
    ? raw.scene_id.trim()
    : 'scene';
  const objects = Array.isArray(raw.objects) ? raw.objects : [];
  if (!objects.length) {
    throw new ApiError(
      422,
      'scene_bundle_invalid',
      'The stage-8 scene bundle does not contain any objects to export.',
      'scene bundle objects must be a non-empty array',
    );
  }

  return {
    scene_id: sceneId,
    requested_ground_plane_y: sanitizeOptionalNumber(raw.requested_ground_plane_y),
    applied_ground_plane_y: sanitizeOptionalNumber(raw.applied_ground_plane_y),
    ground_plane_source: typeof raw.ground_plane_source === 'string' ? raw.ground_plane_source : null,
    objects: objects.map((item, index) => {
      const relativeMeshPath = repoRelativeArtifactPath(item?.mesh_obj_path);
      if (!relativeMeshPath) {
        throw new ApiError(
          422,
          'scene_bundle_invalid',
          'One or more scene objects reference an invalid mesh path.',
          `scene bundle object ${index + 1} has an invalid mesh path`,
        );
      }

      const transform = item?.transform && typeof item.transform === 'object' && !Array.isArray(item.transform)
        ? {
            translation_xyz: sanitizeNumericArray(item.transform.translation_xyz, 3, [0, 0, 0]),
            rotation_type: typeof item.transform.rotation_type === 'string' ? item.transform.rotation_type : 'quaternion',
            rotation_value: sanitizeNumericArray(item.transform.rotation_value, 4, [1, 0, 0, 0]),
            scale_xyz: sanitizeNumericArray(item.transform.scale_xyz, 3, [1, 1, 1]),
          }
        : null;

      return {
        object_id: typeof item?.object_id === 'string' && item.object_id.trim()
          ? item.object_id.trim()
          : `object_${index + 1}`,
        mesh_obj_path: relativeMeshPath,
        already_transformed: Boolean(item?.already_transformed),
        transform,
      };
    }),
  };
}

async function loadSceneBundleForExport(payload) {
  const bundlePath = typeof payload?.bundlePath === 'string' ? payload.bundlePath.trim() : '';
  if (bundlePath) {
    const absolutePath = resolveArtifactPath(bundlePath);
    if (!absolutePath) {
      throw new ApiError(
        422,
        'scene_bundle_invalid',
        'The recorded stage-8 scene bundle path is not allowed.',
        'scene bundle path is not allowed',
      );
    }
    return sanitizeSceneBundleForExport(await readJson(absolutePath));
  }
  return sanitizeSceneBundleForExport(payload?.bundle);
}

function parseSceneExportMode(value) {
  return value === 'separate' ? 'separate' : value === 'merged' ? 'merged' : '';
}

async function ensureDirectories() {
  await fs.mkdir(runtimeDir, { recursive: true });
  await fs.mkdir(jobDir, { recursive: true });
  await fs.mkdir(metricsDir, { recursive: true });
  await fs.mkdir(workspaceResultsDir, { recursive: true });
  await fs.mkdir(renderedImagesDir, { recursive: true });
  await fs.mkdir(rawResponsesDir, { recursive: true });
  await fs.mkdir(tmpDir, { recursive: true });
}

async function readJson(pathname) {
  return JSON.parse(await fs.readFile(pathname, 'utf8'));
}

async function writeJson(pathname, payload) {
  await fs.mkdir(path.dirname(pathname), { recursive: true });
  const tempPath = `${pathname}.${process.pid}.${crypto.randomUUID()}.tmp`;
  await fs.writeFile(tempPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
  await fs.rename(tempPath, pathname);
}

function isVisibleJsonEntry(entry, suffix = '.json') {
  return entry.isFile()
    && entry.name.endsWith(suffix)
    && !entry.name.startsWith('._')
    && !entry.name.startsWith('.');
}

function isVisibleGeneratedEntry(entry) {
  return !entry.name.startsWith('.');
}

function formatBytes(byteCount) {
  const numeric = Number(byteCount) || 0;
  if (numeric < 1024) {
    return `${numeric} B`;
  }
  if (numeric < 1024 * 1024) {
    return `${(numeric / 1024).toFixed(1)} KB`;
  }
  if (numeric < 1024 * 1024 * 1024) {
    return `${(numeric / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(numeric / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

async function collectPathSummary(targetPath, {
  recursive = true,
  includeEntry = isVisibleGeneratedEntry,
} = {}) {
  try {
    const stat = await fs.stat(targetPath);
    if (stat.isFile()) {
      return { entries: 1, bytes: stat.size };
    }
    if (!stat.isDirectory()) {
      return { entries: 0, bytes: 0 };
    }
  } catch {
    return { entries: 0, bytes: 0 };
  }

  const entries = await fs.readdir(targetPath, { withFileTypes: true });
  let totalEntries = 0;
  let totalBytes = 0;
  for (const entry of entries) {
    if (!includeEntry(entry)) {
      continue;
    }
    const absolutePath = path.join(targetPath, entry.name);
    if (entry.isFile()) {
      totalEntries += 1;
      try {
        const stat = await fs.stat(absolutePath);
        totalBytes += stat.size;
      } catch {
        // Ignore transient files.
      }
      continue;
    }
    if (entry.isDirectory()) {
      totalEntries += 1;
      if (recursive) {
        const nested = await collectPathSummary(absolutePath, { recursive, includeEntry });
        totalEntries += nested.entries;
        totalBytes += nested.bytes;
      }
    }
  }
  return { entries: totalEntries, bytes: totalBytes };
}

async function clearDirectoryContents(targetPath, { includeEntry = isVisibleGeneratedEntry } = {}) {
  try {
    const entries = await fs.readdir(targetPath, { withFileTypes: true });
    await Promise.all(entries.map(async (entry) => {
      if (!includeEntry(entry)) {
        return;
      }
      await fs.rm(path.join(targetPath, entry.name), { recursive: true, force: true });
    }));
  } catch {
    // Ignore missing directories.
  }
}

async function clearRuntimeJsonOutputs() {
  try {
    const entries = await fs.readdir(runtimeDir, { withFileTypes: true });
    await Promise.all(entries.map(async (entry) => {
      if (!isVisibleJsonEntry(entry)) {
        return;
      }
      await fs.rm(path.join(runtimeDir, entry.name), { recursive: true, force: true });
    }));
  } catch {
    // Ignore missing runtime dir.
  }
}

async function collectSceneExportSummary() {
  try {
    const entries = await fs.readdir(tmpDir, { withFileTypes: true });
    let totalEntries = 0;
    let totalBytes = 0;
    for (const entry of entries) {
      if (!entry.name.startsWith('scene-export-')) {
        continue;
      }
      totalEntries += 1;
      const nested = await collectPathSummary(path.join(tmpDir, entry.name), { recursive: true });
      totalEntries += nested.entries;
      totalBytes += nested.bytes;
    }
    return { entries: totalEntries, bytes: totalBytes };
  } catch {
    return { entries: 0, bytes: 0 };
  }
}

async function clearSceneExportCaches() {
  try {
    const entries = await fs.readdir(tmpDir, { withFileTypes: true });
    await Promise.all(entries.map(async (entry) => {
      if (!entry.name.startsWith('scene-export-')) {
        return;
      }
      await fs.rm(path.join(tmpDir, entry.name), { recursive: true, force: true });
    }));
  } catch {
    // Ignore missing tmp dir.
  }
}

async function hasActiveJobsOnDisk() {
  try {
    const entries = await fs.readdir(jobDir, { withFileTypes: true });
    for (const entry of entries) {
      if (!isVisibleJsonEntry(entry, '.status.json')) {
        continue;
      }
      try {
        const payload = await readJson(path.join(jobDir, entry.name));
        if (payload && ['queued', 'running', 'awaiting_mask_input', 'awaiting_size_input'].includes(payload.state)) {
          return true;
        }
      } catch {
        // Ignore unreadable status files.
      }
    }
  } catch {
    return false;
  }
  return false;
}

async function localCacheSummary() {
  const targets = [
    {
      id: 'runtime-json',
      label: 'Runtime JSON outputs',
      path: path.relative(repoRoot, runtimeDir),
      ...await collectPathSummary(runtimeDir, {
        recursive: false,
        includeEntry: (entry) => isVisibleJsonEntry(entry),
      }),
    },
    {
      id: 'runtime-raw',
      label: 'Runtime raw responses',
      path: path.relative(repoRoot, rawResponsesDir),
      ...await collectPathSummary(rawResponsesDir),
    },
    {
      id: 'dashboard-jobs',
      label: 'Dashboard job cache',
      path: path.relative(repoRoot, jobDir),
      ...await collectPathSummary(jobDir),
    },
    {
      id: 'scene-workspaces',
      label: 'Scene workspaces',
      path: path.relative(repoRoot, workspaceResultsDir),
      ...await collectPathSummary(workspaceResultsDir),
    },
    {
      id: 'rendered-images',
      label: 'Legacy rendered previews',
      path: path.relative(repoRoot, renderedImagesDir),
      ...await collectPathSummary(renderedImagesDir),
    },
    {
      id: 'scene-export-temp',
      label: 'Temporary scene exports',
      path: path.relative(repoRoot, tmpDir),
      ...await collectSceneExportSummary(),
    },
  ].map((target) => ({
    ...target,
    bytes_human: formatBytes(target.bytes),
  }));

  const totals = targets.reduce((accumulator, target) => ({
    entries: accumulator.entries + target.entries,
    bytes: accumulator.bytes + target.bytes,
  }), { entries: 0, bytes: 0 });

  const hasActiveJobs = runningChildren.size > 0 || await hasActiveJobsOnDisk();
  return {
    targets,
    totals: {
      ...totals,
      bytes_human: formatBytes(totals.bytes),
    },
    has_active_jobs: hasActiveJobs,
  };
}

async function clearLocalRunCaches() {
  if (runningChildren.size > 0 || await hasActiveJobsOnDisk()) {
    throw new ApiError(
      409,
      'local_cache_busy',
      'Local run cache cannot be cleared while a job is active.',
      'one or more dashboard jobs are still queued, running, or awaiting manual input',
      false,
    );
  }

  const before = await localCacheSummary();
  await clearRuntimeJsonOutputs();
  await clearDirectoryContents(rawResponsesDir);
  await clearDirectoryContents(jobDir);
  await clearDirectoryContents(workspaceResultsDir);
  await clearDirectoryContents(renderedImagesDir);
  await clearSceneExportCaches();
  const after = await localCacheSummary();
  return { before, after };
}

class ApiError extends Error {
  constructor(status, code, userMessage, detail, retryable = false, extra = {}) {
    super(detail || userMessage);
    this.status = status;
    this.code = code;
    this.userMessage = userMessage;
    this.detail = detail || userMessage;
    this.retryable = Boolean(retryable);
    this.extra = extra;
  }
}

function sendApiError(res, error, fallback) {
  const payload = error instanceof ApiError
    ? {
        status: error.status,
        code: error.code,
        userMessage: error.userMessage,
        detail: error.detail,
        retryable: error.retryable,
        ...error.extra,
      }
    : {
        status: fallback.status,
        code: fallback.code,
        userMessage: fallback.userMessage,
        detail: error?.message || fallback.userMessage,
        retryable: Boolean(fallback.retryable),
      };

  res.status(payload.status).json({
    ok: false,
    error: {
      code: payload.code,
      userMessage: payload.userMessage,
      detail: payload.detail,
      retryable: payload.retryable,
      ...Object.fromEntries(
        Object.entries(payload).filter(([key]) => !['status', 'code', 'userMessage', 'detail', 'retryable'].includes(key)),
      ),
    },
  });
}

function sanitizeRequestedObjects(value) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  if (typeof value === 'string') {
    return value
      .split(/[\n,]/g)
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return [];
}

function sanitizeStageBackends(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return {};
  }

  const result = {};
  for (const [key, selection] of Object.entries(value)) {
    if (typeof selection !== 'string') {
      continue;
    }
    const normalized = selection.trim();
    if (normalized) {
      result[String(key)] = normalized;
    }
  }
  return result;
}

function sanitizeStageBackendsProfile(value, fallbackValue = 'default') {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : '';
  return STAGE_BACKEND_PROFILE_OPTIONS.includes(normalized) ? normalized : fallbackValue;
}

function loadStageBackendCatalog(profile = DEFAULT_STAGE_BACKENDS_PROFILE) {
  const resolvedProfile = sanitizeStageBackendsProfile(profile, DEFAULT_STAGE_BACKENDS_PROFILE);
  const overrideCatalogPath = resolvedProfile === 'default'
    ? String(process.env.PAT3D_DASHBOARD_STAGE_BACKENDS_CATALOG || process.env.PAT3D_STAGE_BACKENDS_CATALOG || '').trim()
    : '';
  const catalogPath = overrideCatalogPath || STAGE_BACKEND_PROFILE_PATHS[resolvedProfile] || STAGE_BACKEND_PROFILE_PATHS.default;
  return JSON.parse(readFileSync(catalogPath, 'utf8'));
}

function resolveStageBackends(value, {
  profile = DEFAULT_STAGE_BACKENDS_PROFILE,
  forceProfileDefaults = FORCE_STAGE_BACKENDS_PROFILE,
} = {}) {
  const catalog = loadStageBackendCatalog(profile);
  const selected = sanitizeStageBackends(value);
  const resolved = {};

  for (const [stageId, entry] of Object.entries(catalog)) {
    const defaultValue = typeof entry?.default === 'string' ? entry.default.trim() : '';
    const candidate = forceProfileDefaults ? defaultValue : (selected[stageId] || defaultValue);
    const allowedValues = new Set(
      Array.isArray(entry?.options)
        ? entry.options
            .map((option) => (typeof option?.value === 'string' ? option.value.trim() : ''))
            .filter(Boolean)
        : [],
    );
    resolved[stageId] = candidate === 'disabled' || allowedValues.has(candidate) ? candidate : defaultValue;
  }

  return resolved;
}

function sanitizeSegmentationMode(value) {
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

function sanitizeModel(value, allowedValues, fallbackValue) {
  if (typeof value !== 'string') {
    return fallbackValue;
  }
  const normalized = value.trim();
  return allowedValues.includes(normalized) ? normalized : fallbackValue;
}

function sanitizeReasoningEffort(value, fallbackValue = REASONING_EFFORT_OPTIONS[0]) {
  return sanitizeModel(value, REASONING_EFFORT_OPTIONS, fallbackValue);
}

function sanitizeBoolean(value, fallbackValue) {
  return typeof value === 'boolean' ? value : fallbackValue;
}

function sanitizeInteger(value, fallbackValue, minimum, maximum) {
  const numeric = Number.parseInt(String(value ?? ''), 10);
  if (!Number.isFinite(numeric)) {
    return fallbackValue;
  }
  return Math.min(maximum, Math.max(minimum, numeric));
}

function sanitizeFloat(value, fallbackValue, minimum, maximum) {
  const numeric = Number.parseFloat(String(value ?? ''));
  if (!Number.isFinite(numeric)) {
    return fallbackValue;
  }
  return Math.min(maximum, Math.max(minimum, numeric));
}

function sanitizePhysicsSettings(value) {
  const raw = value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  return {
    diffSimEnabled: sanitizeBoolean(
      raw.diffSimEnabled ?? raw.diff_sim_enabled,
      DEFAULT_PHYSICS_SETTINGS.diffSimEnabled,
    ),
    endFrame: sanitizeInteger(
      raw.endFrame ?? raw.end_frame,
      DEFAULT_PHYSICS_SETTINGS.endFrame,
      1,
      5000,
    ),
    groundYValue: sanitizeFloat(
      raw.groundYValue ?? raw.ground_y_value,
      DEFAULT_PHYSICS_SETTINGS.groundYValue,
      -10.0,
      10.0,
    ),
    totalOptEpoch: sanitizeInteger(
      raw.totalOptEpoch ?? raw.total_opt_epoch,
      DEFAULT_PHYSICS_SETTINGS.totalOptEpoch,
      1,
      1000,
    ),
    physLr: sanitizeFloat(
      raw.physLr ?? raw.phys_lr,
      DEFAULT_PHYSICS_SETTINGS.physLr,
      1e-6,
      1.0,
    ),
    contactDHat: sanitizeFloat(
      raw.contactDHat ?? raw.contact_d_hat,
      DEFAULT_PHYSICS_SETTINGS.contactDHat,
      1e-7,
      1e-1,
    ),
    contactEpsVelocity: sanitizeFloat(
      raw.contactEpsVelocity ?? raw.contact_eps_velocity,
      DEFAULT_PHYSICS_SETTINGS.contactEpsVelocity,
      1e-8,
      1e-1,
    ),
  };
}

function slugify(value) {
  const slug = value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  return slug.slice(0, 40) || 'scene';
}

function metricCaseFileName(sceneId) {
  const cleaned = String(sceneId || 'unknown').replace(/[\\/]/g, '_').trim() || 'unknown';
  return `${cleaned}.json`;
}

function runtimeMetricsKey(runtimeName) {
  const base = path.basename(String(runtimeName || '').trim());
  return base.replace(/\.paper_core\.json$/i, '').replace(/\.json$/i, '') || 'unknown';
}

function nestedRuntimeValue(payload, pathKey) {
  let current = payload;
  for (const part of pathKey.split('.')) {
    if (!current || typeof current !== 'object' || !(part in current)) {
      return null;
    }
    current = current[part];
  }
  return current;
}

function extractSceneIdFromRuntimePayload(runtimePayload, fallbackName = '') {
  const scenePaths = [
    'layout_initialization.scene_layout.scene_id',
    'simulation_preparation.physics_ready_scene.scene_id',
    'physics_optimization.optimization_result.scene_id',
    'visualization.render_result.scene_id',
    'object_assets.object_assets.scene_id',
  ];
  for (const scenePath of scenePaths) {
    const value = nestedRuntimeValue(runtimePayload, scenePath);
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  const fallback = String(fallbackName || '').replace(/\.paper_core\.json$|\.json$/g, '');
  return fallback || 'unknown';
}

function buildOutputName(sceneId, jobId) {
  const stem = sceneId && sceneId.trim() ? sceneId.trim() : 'dashboard';
  return `${slugify(stem)}-${jobId.slice(-8)}.paper_core.json`;
}

function buildJobPaths(jobId, sceneId, runtimeOutputName = buildOutputName(sceneId, jobId)) {
  return {
    inputPath: path.join(jobDir, `${jobId}.input.json`),
    statusPath: path.join(jobDir, `${jobId}.status.json`),
    runtimeConfigPath: path.join(jobDir, `${jobId}.runtime.json`),
    logPath: path.join(jobDir, `${jobId}.log`),
    outputName: runtimeOutputName,
    outputPath: path.join(runtimeDir, runtimeOutputName),
    manualMaskDir: path.join(jobDir, jobId, 'manual_masks'),
  };
}

function randomAccessColor(index) {
  const palette = [
    '#ff6b6b',
    '#4dabf7',
    '#51cf66',
    '#fcc419',
    '#845ef7',
    '#ff922b',
    '#20c997',
    '#f06595',
    '#5c7cfa',
    '#94d82d',
  ];
  return palette[index % palette.length];
}

function normalizeHexColor(value, index) {
  if (typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value.trim())) {
    return value.trim();
  }
  return randomAccessColor(index);
}

function cloneJsonValue(value) {
  return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

function objectCatalogFromRuntime(runtimePayload) {
  return runtimePayload?.first_contract_slice?.object_relation?.object_catalog
    || runtimePayload?.first_contract_slice?.scene_understanding?.object_catalog
    || null;
}

function relationGraphFromRuntime(runtimePayload) {
  return runtimePayload?.first_contract_slice?.object_relation?.relation_graph || null;
}

function validateAcyclicRelationGraph(relations, rootObjectIds) {
  const edges = new Map();
  const allNodes = new Set(rootObjectIds);
  for (const relation of relations) {
    allNodes.add(relation.parent_object_id);
    allNodes.add(relation.child_object_id);
    const children = edges.get(relation.parent_object_id) || [];
    children.push(relation.child_object_id);
    edges.set(relation.parent_object_id, children);
  }

  const visiting = new Set();
  const visited = new Set();
  function visit(node) {
    if (visited.has(node)) return;
    if (visiting.has(node)) {
      throw new ApiError(
        422,
        'relation_graph_cycle',
        'Relation graph edits must remain acyclic.',
        'relation graph contains a cycle',
      );
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
}

function normalizeRelationGraphPayload(payload, {
  sceneId,
  objectIds,
  currentRootObjectIds = [],
} = {}) {
  const raw = payload && typeof payload === 'object' && !Array.isArray(payload) ? payload : {};
  if (raw.clear === true) {
    return null;
  }
  const graph = raw.relationGraph && typeof raw.relationGraph === 'object'
    ? raw.relationGraph
    : raw.relation_graph && typeof raw.relation_graph === 'object'
      ? raw.relation_graph
      : raw;
  const rawRelations = Array.isArray(graph?.relations) ? graph.relations : null;
  if (!rawRelations) {
    throw new ApiError(
      422,
      'relation_graph_invalid',
      'The relation graph payload is invalid.',
      'relations must be an array',
    );
  }

  const validObjectIds = new Set(objectIds || []);
  const normalizedRelations = rawRelations.map((entry, index) => {
    const parentObjectId = typeof entry?.parent_object_id === 'string' ? entry.parent_object_id.trim() : '';
    const childObjectId = typeof entry?.child_object_id === 'string' ? entry.child_object_id.trim() : '';
    const relationType = typeof entry?.relation_type === 'string' ? entry.relation_type.trim().toLowerCase() : '';
    if (!parentObjectId || !childObjectId) {
      throw new ApiError(
        422,
        'relation_graph_invalid',
        'Each edited relation must include a parent and child object.',
        `relation ${index + 1} is missing parent_object_id or child_object_id`,
      );
    }
    if (parentObjectId === childObjectId) {
      throw new ApiError(
        422,
        'relation_graph_invalid',
        'A relation cannot point from an object to itself.',
        `relation ${index + 1} is self-referential`,
      );
    }
    if (!RELATION_TYPE_OPTIONS.includes(relationType)) {
      throw new ApiError(
        422,
        'relation_graph_invalid',
        'The relation graph contains an unsupported edge type.',
        `relation ${index + 1} uses unsupported relation_type "${relationType || '<empty>'}"`,
      );
    }
    if (validObjectIds.size) {
      if (!validObjectIds.has(parentObjectId) || !validObjectIds.has(childObjectId)) {
        throw new ApiError(
          422,
          'relation_graph_invalid',
          'The relation graph references objects that are not part of this scene.',
          `relation ${index + 1} references unknown object ids`,
        );
      }
    }
    const confidence = entry?.confidence;
    const numericConfidence = confidence === null || confidence === undefined || confidence === ''
      ? null
      : Number.parseFloat(String(confidence));
    if (numericConfidence !== null && (!Number.isFinite(numericConfidence) || numericConfidence < 0 || numericConfidence > 1)) {
      throw new ApiError(
        422,
        'relation_graph_invalid',
        'Relation confidence must be between 0 and 1.',
        `relation ${index + 1} has invalid confidence`,
      );
    }
    return {
      parent_object_id: parentObjectId,
      relation_type: relationType,
      child_object_id: childObjectId,
      confidence: numericConfidence,
      evidence: typeof entry?.evidence === 'string' && entry.evidence.trim() ? entry.evidence.trim() : null,
    };
  });

  const rawRoots = Array.isArray(graph?.root_object_ids)
    ? graph.root_object_ids
    : Array.isArray(graph?.rootObjectIds)
      ? graph.rootObjectIds
      : currentRootObjectIds;
  const normalizedRootObjectIds = [...new Set(
    rawRoots
      .filter((value) => typeof value === 'string')
      .map((value) => value.trim())
      .filter((value) => value && (!validObjectIds.size || validObjectIds.has(value))),
  )];
  validateAcyclicRelationGraph(normalizedRelations, normalizedRootObjectIds);

  return {
    scene_id: sceneId,
    relations: normalizedRelations,
    root_object_ids: normalizedRootObjectIds,
  };
}

function referenceImageResultFromRuntime(runtimePayload) {
  return runtimePayload?.first_contract_slice?.reference_image_result || null;
}

function depthResultFromRuntime(runtimePayload) {
  return runtimePayload?.first_contract_slice?.scene_understanding?.depth_result
    || runtimePayload?.first_contract_slice?.object_relation?.depth_result
    || null;
}

function buildManualCheckpointStageRows(stageRows, updatedAt) {
  const existingRows = Array.isArray(stageRows) && stageRows.length
    ? stageRows
    : JOB_STAGE_BLUEPRINTS.map((stage) => ({ ...stage }));

  return existingRows.map((stage, index) => {
    const nextStage = {
      ...stage,
      error: null,
      error_detail: null,
      retryable: false,
    };

    if (index === 0) {
      return {
        ...nextStage,
        status: 'completed',
        progress: 100,
        progress_total: nextStage.progress_total || 100,
        progress_completed: nextStage.progress_total || 100,
        completed_at: nextStage.completed_at || updatedAt,
        last_log: nextStage.last_log || 'Reference image generation complete.',
      };
    }

    if (index === 1) {
      return {
        ...nextStage,
        status: 'awaiting_input',
        started_at: updatedAt,
        completed_at: null,
        progress: 0,
        progress_total: 1,
        progress_completed: 0,
        last_log: 'Reference image ready. Waiting for manual masks.',
      };
    }

    return {
      ...nextStage,
      status: 'pending',
      started_at: null,
      completed_at: null,
      progress: null,
      progress_total: null,
      progress_completed: null,
      last_log: null,
    };
  });
}

function parsePngDataUrl(dataUrl) {
  if (typeof dataUrl !== 'string') {
    throw new ApiError(422, 'invalid_mask_payload', 'Manual mask payload is invalid.', 'maskDataUrl must be a string.');
  }

  const match = dataUrl.match(/^data:image\/png;base64,([a-z0-9+/=]+)$/i);
  if (!match) {
    throw new ApiError(
      422,
      'invalid_mask_payload',
      'Manual mask payload is invalid.',
      'Each mask must be a PNG data URL.',
    );
  }

  return Buffer.from(match[1], 'base64');
}

function normalizeManualMaskInstances(value) {
  if (!Array.isArray(value)) {
    throw new ApiError(422, 'invalid_mask_payload', 'Manual mask payload is invalid.', 'instances must be an array.');
  }

  const usedIds = new Set();
  return value.map((entry, index) => {
    const label = typeof entry?.label === 'string' ? entry.label.trim() : '';
    if (!label) {
      throw new ApiError(422, 'invalid_mask_payload', 'Manual mask payload is invalid.', 'Each mask must include a label.');
    }

    const requestedId = typeof entry?.instanceId === 'string' ? entry.instanceId.trim() : '';
    let instanceId = slugify(requestedId || `${label}-${index + 1}`);
    while (usedIds.has(instanceId)) {
      instanceId = `${instanceId}-${usedIds.size + 1}`;
    }
    usedIds.add(instanceId);

    return {
      instance_id: instanceId,
      label,
      color: normalizeHexColor(entry?.color, index),
      width: Number.isFinite(Number(entry?.width)) ? Number(entry.width) : null,
      height: Number.isFinite(Number(entry?.height)) ? Number(entry.height) : null,
      maskBuffer: parsePngDataUrl(entry?.maskDataUrl),
    };
  });
}

function normalizeManualSizeEntries(value) {
  if (!Array.isArray(value)) {
    throw new ApiError(422, 'invalid_size_payload', 'Manual size payload is invalid.', 'entries must be an array.');
  }

  return value.map((entry, index) => {
    const objectId = typeof entry?.object_id === 'string'
      ? entry.object_id.trim()
      : typeof entry?.objectId === 'string'
        ? entry.objectId.trim()
        : '';
    if (!objectId) {
      throw new ApiError(422, 'invalid_size_payload', 'Manual size payload is invalid.', `Entry ${index + 1} is missing object_id.`);
    }
    const canonicalName = typeof entry?.canonical_name === 'string'
      ? entry.canonical_name.trim()
      : typeof entry?.canonicalName === 'string'
        ? entry.canonicalName.trim()
        : '';
    const displayName = typeof entry?.display_name === 'string'
      ? entry.display_name.trim()
      : typeof entry?.displayName === 'string'
        ? entry.displayName.trim()
        : '';
    const dimensionsSource = entry?.dimensions_m && typeof entry.dimensions_m === 'object' && !Array.isArray(entry.dimensions_m)
      ? entry.dimensions_m
      : entry?.dimensionsM && typeof entry.dimensionsM === 'object' && !Array.isArray(entry.dimensionsM)
        ? entry.dimensionsM
        : null;
    if (!dimensionsSource) {
      throw new ApiError(422, 'invalid_size_payload', 'Manual size payload is invalid.', `Entry ${objectId} is missing dimensions_m.`);
    }
    const dimensions = {};
    for (const axis of ['x', 'y', 'z']) {
      const numeric = Number.parseFloat(String(dimensionsSource?.[axis] ?? ''));
      if (!Number.isFinite(numeric) || numeric <= 0) {
        throw new ApiError(422, 'invalid_size_payload', 'Manual size payload is invalid.', `Entry ${objectId} must provide a positive ${axis} dimension.`);
      }
      dimensions[axis] = numeric;
    }
    return {
      object_id: objectId,
      canonical_name: canonicalName || objectId,
      display_name: displayName || canonicalName || objectId,
      dimensions_m: dimensions,
    };
  });
}

function isProcessAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false;
  }
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (error?.code === 'EPERM') {
      return true;
    }
    return false;
  }
}

async function reconcileJobStatus(payload, statusPath) {
  if (!payload) {
    return payload;
  }

  if (payload.state === 'failed') {
    const nextPayload = {
      ...payload,
      stages: (payload.stages || []).map((stage) => (
        stage.status === 'running'
          ? {
              ...stage,
              status: 'failed',
              completed_at: payload.finished_at || payload.updated_at || new Date().toISOString(),
              error: payload.error?.user_message || payload.error || 'dashboard worker process is no longer running',
              error_detail: payload.error?.technical_message || null,
              retryable: Boolean(payload.error?.retryable),
            }
          : stage
      )),
    };
    await writeJson(statusPath, nextPayload);
    return nextPayload;
  }

  if (payload.state === 'awaiting_mask_input' || payload.state === 'awaiting_size_input') {
    return payload;
  }

  if (payload.state !== 'queued' && payload.state !== 'running') {
    return payload;
  }

  const workerPid = Number(payload.worker_pid);
  if (runningChildren.has(payload.job_id) || isProcessAlive(workerPid)) {
    return payload;
  }

  const updatedAt = Date.parse(payload.updated_at || payload.started_at || '');
  if (Number.isFinite(updatedAt) && Date.now() - updatedAt < 30000) {
    return payload;
  }

  const nextPayload = {
    ...payload,
    state: 'failed',
    finished_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    error: payload.error || {
      code: 'worker_missing',
      user_message: 'The dashboard worker process is no longer running.',
      technical_message: 'dashboard worker process is no longer running',
      retryable: false,
      phase: 'dashboard_job',
      details: {},
    },
  };
  nextPayload.stages = (payload.stages || []).map((stage) => (
    stage.status === 'running'
      ? {
          ...stage,
          status: 'failed',
          completed_at: nextPayload.finished_at,
          error: nextPayload.error.user_message,
          error_detail: nextPayload.error.technical_message,
          retryable: Boolean(nextPayload.error.retryable),
        }
      : stage
  ));
  await writeJson(statusPath, nextPayload);
  return nextPayload;
}

async function startJobWorker({
  jobId,
  inputPath,
  statusPath,
  outputPath,
  runtimeConfigPath,
  logPath,
  resumeManual = false,
  resumeFrom = '',
}) {
  const args = [
    jobRunnerScript,
    '--job-id', jobId,
    '--input', inputPath,
    '--status', statusPath,
    '--output', outputPath,
    '--runtime-config-out', runtimeConfigPath,
  ];
  if (resumeManual) {
    args.push('--resume-manual');
  }
  if (resumeFrom) {
    args.push('--resume-from', resumeFrom);
  }

  const logFd = openSync(logPath, 'a');
  writeSync(logFd, `[dashboard-run-start] python=${pythonExecutable} args=${JSON.stringify(args)}\n`);
  let child;
  try {
    child = spawn(
      pythonExecutable,
      args,
      {
        cwd: repoRoot,
        env: {
          ...process.env,
          PAT3D_REPO_ROOT: repoRoot,
          ...(sam3PythonExecutable ? { PAT3D_SAM3_PYTHON: sam3PythonExecutable } : {}),
        },
        detached: false,
        stdio: ['ignore', logFd, logFd],
      },
    );
  } finally {
    closeSync(logFd);
  }

  const initialStatus = await readJson(statusPath);
  initialStatus.worker_pid = child.pid;
  initialStatus.updated_at = new Date().toISOString();
  initialStatus.interruption = null;
  await writeJson(statusPath, initialStatus);

  let finalized = false;
  const finalizeChild = async (exitCode, signal = null, extraLog = '') => {
    if (finalized) {
      return;
    }
    finalized = true;
    if (runningChildren.get(jobId) === child) {
      runningChildren.delete(jobId);
    }
    if (extraLog) {
      await fs.appendFile(logPath, extraLog);
    }
    await fs.appendFile(logPath, `[dashboard-run-close] code=${exitCode ?? 'null'} signal=${signal || 'null'}\n`);
    await patchJobOnProcessExit(jobId, exitCode ?? 1, signal, child.pid);
  };

  child.on('close', (code, signal) => {
    void finalizeChild(code ?? 1, signal);
  });

  child.on('error', (error) => {
    void finalizeChild(1, null, `\n[dashboard-run-error] ${error.stack || error.message}\n`);
  });

  runningChildren.set(jobId, child);
  child.unref();
}

async function listRuntimeFiles() {
  await ensureDirectories();
  const entries = await fs.readdir(runtimeDir, { withFileTypes: true });
  const jsonFiles = await Promise.all(
    entries
      .filter((entry) => isVisibleJsonEntry(entry))
      .map(async (entry) => {
        const absolutePath = path.join(runtimeDir, entry.name);
        const stat = await fs.stat(absolutePath);
        return {
          name: entry.name,
          updatedAt: stat.mtime.toISOString(),
          size: stat.size,
        };
      }),
  );

  const sortedFiles = jsonFiles.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  if (sortedFiles.length > MAX_RUNTIME_FILES) {
    const outdated = sortedFiles.slice(MAX_RUNTIME_FILES);
    await Promise.all(outdated.map(async (entry) => {
      try {
        await fs.unlink(path.join(runtimeDir, entry.name));
      } catch {
        // Ignore stale or already removed runtime artifacts.
      }
    }));
    return sortedFiles.slice(0, MAX_RUNTIME_FILES);
  }
  return sortedFiles;
}

async function loadRuntimeByName(fileName) {
  const safeName = path.basename(fileName || 'paper_core.real.json');
  const absolutePath = path.join(runtimeDir, safeName);
  const text = await fs.readFile(absolutePath, 'utf8');
  const stat = await fs.stat(absolutePath);
  return {
    name: safeName,
    updatedAt: stat.mtime.toISOString(),
    size: stat.size,
    data: JSON.parse(text),
  };
}

function extractJobRuntimeProgress(jobPayload) {
  const runningStage = (jobPayload?.stages || []).find((stage) => stage.status === 'running')
    || (jobPayload?.stages || []).find((stage) => stage.status === 'awaiting_input')
    || (jobPayload?.stages || []).find((stage) => stage.status === 'queued')
    || null;

  const stageName = runningStage?.label || jobPayload?.current_stage_id || null;
  const stageProgress = runningStage?.progress === null || runningStage?.progress === undefined ? null : runningStage.progress;

  return {
    stageName,
    stageProgress,
    updatedAt: jobPayload?.updated_at || null,
  };
}

async function findJobByRuntimeName(runtimeName) {
  const jobs = await listJobs();
  return jobs.find((job) => job?.runtime_output_name === runtimeName) || null;
}

async function listJobs() {
  await ensureDirectories();
  const entries = await fs.readdir(jobDir, { withFileTypes: true });
  const statusEntries = (await Promise.all(
    entries
      .filter((entry) => isVisibleJsonEntry(entry, '.status.json'))
      .map(async (entry) => {
        const absolutePath = path.join(jobDir, entry.name);
        try {
          const payload = await readJson(absolutePath);
          return reconcileJobStatus(payload, absolutePath);
        } catch (error) {
          if (error instanceof SyntaxError) {
            return null;
          }
          throw error;
        }
      }),
  )).filter(Boolean);

  return statusEntries.sort((a, b) => String(b.updated_at || '').localeCompare(String(a.updated_at || '')));
}

function isActiveJobState(state) {
  return ['queued', 'running', 'awaiting_mask_input', 'awaiting_size_input'].includes(String(state || ''));
}

async function findActiveJob({ excludeJobId = '' } = {}) {
  const jobs = await listJobs();
  return jobs.find((job) => isActiveJobState(job?.state) && job?.job_id !== excludeJobId) || null;
}

async function loadJob(jobId) {
  const safeId = path.basename(jobId);
  const statusPath = path.join(jobDir, `${safeId}.status.json`);
  const payload = await readJson(statusPath);
  return reconcileJobStatus(payload, statusPath);
}

async function readJobLog(jobId, tailLines = 200) {
  const safeId = path.basename(jobId);
  const logPath = path.join(jobDir, `${safeId}.log`);
  const text = await fs.readFile(logPath, 'utf8');
  const lines = text.split(/\r?\n/);
  return lines.slice(Math.max(0, lines.length - tailLines)).join('\n');
}

async function loadMetricsForRuntime(runtimeName) {
  const safeRuntimeName = path.basename(runtimeName || '');
  if (!safeRuntimeName) {
    throw new ApiError(422, 'metrics_runtime_missing', 'A runtime name is required to load metrics.', 'runtime query parameter is missing');
  }
  const runtimePath = path.join(runtimeDir, safeRuntimeName);
  if (!existsSync(runtimePath)) {
    throw new ApiError(404, 'metrics_runtime_not_found', 'The selected runtime output does not exist.', runtimePath);
  }
  const runtimePayload = await loadRuntimeByName(safeRuntimeName);
  const sceneId = extractSceneIdFromRuntimePayload(runtimePayload.data, safeRuntimeName);
  const metricsPathCandidates = [
    path.join(metricsDir, metricCaseFileName(runtimeMetricsKey(safeRuntimeName))),
    path.join(metricsDir, metricCaseFileName(sceneId)),
  ].filter((candidate, index, all) => all.indexOf(candidate) === index);
  const metricsPath = metricsPathCandidates.find((candidate) => existsSync(candidate)) || metricsPathCandidates[0];
  if (!existsSync(metricsPath)) {
    return {
      available: false,
      runtime: safeRuntimeName,
      scene_id: sceneId,
      metrics_path: path.relative(repoRoot, metricsPath),
      metrics: null,
    };
  }
  const metrics = await readJson(metricsPath);
  const runtimePathResolved = path.resolve(runtimePath);
  const metricsRuntimePath = typeof metrics?.runtime_path === 'string' ? path.resolve(metrics.runtime_path) : '';
  if (metricsRuntimePath && metricsRuntimePath !== runtimePathResolved) {
    return {
      available: false,
      runtime: safeRuntimeName,
      scene_id: sceneId,
      metrics_path: path.relative(repoRoot, metricsPath),
      metrics: null,
      stale_reason: 'runtime_mismatch',
    };
  }
  const generatedAtMs = Date.parse(typeof metrics?.generated_at === 'string' ? metrics.generated_at : '');
  const runtimeStat = await fs.stat(runtimePath);
  if (Number.isFinite(generatedAtMs) && generatedAtMs < runtimeStat.mtimeMs - 1) {
    return {
      available: false,
      runtime: safeRuntimeName,
      scene_id: sceneId,
      metrics_path: path.relative(repoRoot, metricsPath),
      metrics: null,
      stale_reason: 'runtime_newer_than_metrics',
    };
  }
  return {
    available: true,
    runtime: safeRuntimeName,
    scene_id: sceneId,
    metrics_path: path.relative(repoRoot, metricsPath),
    metrics,
  };
}

function metricsPythonEnv() {
  const t2vRoot = path.join(repoRoot, 'extern', 't2v_metrics');
  const homeDir = process.env.HOME || '';
  const t2vCandidates = [
    process.env.PAT3D_VQA_PYTHON || '',
    path.join(repoRoot, '.venv-t2v', 'bin', 'python'),
    path.join(homeDir, 'anaconda3', 'envs', 'sam3d-objects', 'bin', 'python'),
    path.join(homeDir, '.conda', 'envs', 'sam3d-objects', 'bin', 'python'),
    path.join(homeDir, 'miniconda3', 'envs', 'sam3d-objects', 'bin', 'python'),
  ];
  const t2vPython = t2vCandidates.find((candidate) => candidate && existsSync(candidate)) || '';
  const t2vCacheDir = process.env.PAT3D_T2V_HF_CACHE_DIR || path.join(repoRoot, 'data', 'metrics', 'hf_cache');
  const pythonPathEntries = [
    existsSync(t2vRoot) ? t2vRoot : '',
    process.env.PYTHONPATH || '',
  ].filter(Boolean);
  const blenderCandidates = [
    process.env.PAT3D_BLENDER_BIN || '',
    '/home/eleven/guying/blender-4.3.0-linux-x64/blender',
    '/usr/local/blender/blender',
  ];
  const blenderBin = blenderCandidates.find((candidate) => candidate && existsSync(candidate)) || '';
  return {
    ...(pythonPathEntries.length ? { PYTHONPATH: pythonPathEntries.join(path.delimiter) } : {}),
    ...(blenderBin ? { PAT3D_BLENDER_BIN: blenderBin } : {}),
    ...(existsSync(t2vPython) ? { PAT3D_VQA_PYTHON: t2vPython } : {}),
    PAT3D_VQA_MODEL: process.env.PAT3D_VQA_MODEL || 'clip-flant5-xl',
    PAT3D_T2V_HF_CACHE_DIR: t2vCacheDir,
  };
}

async function computeMetricsForRuntime(payload) {
  const runtimeName = path.basename(typeof payload?.runtime === 'string' ? payload.runtime : '');
  if (!runtimeName) {
    throw new ApiError(422, 'metrics_runtime_missing', 'A runtime name is required to compute metrics.', 'runtime field is missing');
  }
  const runtimePath = path.join(runtimeDir, runtimeName);
  if (!existsSync(runtimePath)) {
    throw new ApiError(404, 'metrics_runtime_not_found', 'The selected runtime output does not exist.', runtimePath);
  }

  const args = [
    metricsScript,
    '--runtime', runtimePath,
    '--output-root', metricsDir,
  ];
  if (payload?.force) {
    args.push('--force-render', '--force-metrics');
  }
  if (payload?.skipClip) {
    args.push('--skip-clip');
  }
  if (payload?.skipVqa) {
    args.push('--skip-vqa');
  }
  if (payload?.skipPhysical) {
    args.push('--skip-physical');
  }

  return runPythonJsonScript(
    pythonExecutable,
    args,
    { env: metricsPythonEnv() },
  );
}

async function patchJobOnProcessExit(jobId, exitCode, signal = null, workerPid = null) {
  const statusPath = path.join(jobDir, `${jobId}.status.json`);
  try {
    const payload = await readJson(statusPath);
    const expectedExitPid = Number.parseInt(String(payload?.interruption?.expected_exit_pid || ''), 10);
    if (Number.isInteger(workerPid) && Number.isInteger(expectedExitPid) && expectedExitPid === workerPid) {
      return;
    }
    const trackedWorkerPid = Number.parseInt(String(payload?.worker_pid || ''), 10);
    if (Number.isInteger(workerPid) && Number.isInteger(trackedWorkerPid) && trackedWorkerPid !== workerPid) {
      return;
    }
    if (payload.state === 'completed' || payload.state === 'failed' || payload.state === 'awaiting_mask_input' || payload.state === 'awaiting_size_input' || payload.state === 'cancelled') {
      return;
    }
    payload.state = exitCode === 0 ? 'completed' : 'failed';
    payload.updated_at = new Date().toISOString();
    payload.finished_at = payload.updated_at;
    payload.worker_pid = null;
    payload.interruption = null;
    if (exitCode !== 0) {
      const failureCode = signal ? 'worker_signal' : 'worker_exit';
      const failureUserMessage = signal
        ? `Dashboard job process terminated with signal ${signal}.`
        : `Dashboard job process exited with code ${exitCode}.`;
      const failureTechnicalMessage = signal
        ? `dashboard job process terminated with signal ${signal}`
        : `dashboard job process exited with code ${exitCode}`;
      payload.error = payload.error || {
        code: failureCode,
        user_message: failureUserMessage,
        technical_message: failureTechnicalMessage,
        retryable: false,
        phase: 'dashboard_job',
        details: {
          exit_code: exitCode,
          signal,
        },
      };
      payload.stages = (payload.stages || []).map((stage) => (
        stage.status === 'running'
          ? {
              ...stage,
              status: 'failed',
              completed_at: payload.finished_at,
              error: payload.error.user_message,
              error_detail: payload.error.technical_message,
              retryable: Boolean(payload.error.retryable),
            }
          : stage
      ));
    }
    await writeJson(statusPath, payload);
  } catch (error) {
    console.error(`Could not patch dashboard job status for ${jobId}:`, error);
  }
}

async function createJob(payload) {
  await ensureDirectories();

  const prompt = typeof payload?.prompt === 'string' ? payload.prompt.trim() : '';
  if (!prompt) {
    throw new Error('prompt must be a non-empty string');
  }
  const activeJob = await findActiveJob();
  if (activeJob) {
    throw new ApiError(
      409,
      'active_job_exists',
      'Finish or retry the current run before starting a new one.',
      `job "${activeJob.scene_id}" (${activeJob.job_id}) is currently ${activeJob.state}`,
      true,
      {
        active_job_id: activeJob.job_id,
        active_job_state: activeJob.state,
        active_scene_id: activeJob.scene_id,
      },
    );
  }

  const jobId = crypto.randomUUID();
  const sceneId = typeof payload?.sceneId === 'string' && payload.sceneId.trim()
    ? payload.sceneId.trim()
    : `${slugify(prompt)}-${jobId.slice(-8)}`;
  const requestedObjects = sanitizeRequestedObjects(payload?.requestedObjects);
  const requestedObjectsInferred = false;
  const stageBackendsProfile = sanitizeStageBackendsProfile(
    payload?.stageBackendsProfile ?? payload?.stage_backends_profile,
    DEFAULT_STAGE_BACKENDS_PROFILE,
  );
  const stageBackends = resolveStageBackends(payload?.stageBackends, { profile: stageBackendsProfile });
  const segmentationMode = sanitizeSegmentationMode(payload?.segmentationMode);
  const previewAngleCount = sanitizePreviewAngleCount(payload?.previewAngleCount);
  const requestedObjectInferenceBudget = sanitizeRequestedObjectInferenceBudget(
    payload?.requestedObjectInferenceBudget ?? payload?.requested_object_inference_budget,
  );
  const physicsSettings = sanitizePhysicsSettings(payload?.physicsSettings ?? payload?.physics_settings);
  const llmModel = sanitizeModel(payload?.llmModel || payload?.llm_model, CHAT_MODEL_OPTIONS, 'gpt-5.4');
  const imageModel = sanitizeModel(payload?.imageModel || payload?.image_model, IMAGE_MODEL_OPTIONS, 'gpt-image-1.5');
  const objectCropCompletionEnabled = Boolean(
    payload?.objectCropCompletionEnabled ?? payload?.object_crop_completion_enabled ?? true,
  );
  const structuredLlmMaxAttempts = sanitizeStructuredLlmMaxAttempts(
    payload?.structuredLlmMaxAttempts ?? payload?.structured_llm_max_attempts,
  );
  const structuredLlmReasoningBudget = sanitizeStructuredLlmReasoningBudget(
    payload?.structuredLlmReasoningBudget ?? payload?.structured_llm_reasoning_budget,
  );
  const objectCropCompletionModel = sanitizeModel(
    payload?.objectCropCompletionModel || payload?.object_crop_completion_model,
    IMAGE_MODEL_OPTIONS,
    'gpt-image-1.5',
  );
  const reasoningEffort = sanitizeReasoningEffort(payload?.reasoningEffort || payload?.reasoning_effort, 'high');
  if (segmentationMode === 'manual' && stageBackends['scene-understanding-segmentation'] === 'sam3_segmenter') {
    stageBackends['scene-understanding-segmentation'] = 'current_segmenter';
  }
  const {
    inputPath,
    statusPath,
    runtimeConfigPath,
    logPath,
    outputName,
    outputPath,
  } = buildJobPaths(jobId, sceneId);

  const inputPayload = {
    prompt,
    scene_id: sceneId,
    requested_objects: requestedObjects,
    requested_objects_inferred: requestedObjectsInferred,
    stage_backends: stageBackends,
    stage_backends_profile: stageBackendsProfile,
    segmentation_mode: segmentationMode,
    preview_angle_count: previewAngleCount,
    requested_object_inference_budget: requestedObjectInferenceBudget,
    physics_settings: {
      diff_sim_enabled: physicsSettings.diffSimEnabled,
      end_frame: physicsSettings.endFrame,
      ground_y_value: physicsSettings.groundYValue,
      total_opt_epoch: physicsSettings.totalOptEpoch,
      phys_lr: physicsSettings.physLr,
      contact_d_hat: physicsSettings.contactDHat,
      contact_eps_velocity: physicsSettings.contactEpsVelocity,
    },
    llm_model: llmModel,
    image_model: imageModel,
    object_crop_completion_enabled: objectCropCompletionEnabled,
    object_crop_completion_model: objectCropCompletionModel,
    structured_llm_max_attempts: structuredLlmMaxAttempts,
    structured_llm_reasoning_budget: structuredLlmReasoningBudget,
    reasoning_effort: reasoningEffort,
    relation_graph_override: null,
    started_at: new Date().toISOString(),
  };

  await writeJson(inputPath, inputPayload);
  await writeJson(statusPath, {
    job_id: jobId,
    state: 'queued',
    prompt,
    scene_id: sceneId,
    requested_objects: requestedObjects,
    requested_objects_inferred: requestedObjectsInferred,
    stage_backends: stageBackends,
    stage_backends_profile: stageBackendsProfile,
    segmentation_mode: segmentationMode,
    preview_angle_count: previewAngleCount,
    requested_object_inference_budget: requestedObjectInferenceBudget,
    physics_settings: {
      diff_sim_enabled: physicsSettings.diffSimEnabled,
      end_frame: physicsSettings.endFrame,
      ground_y_value: physicsSettings.groundYValue,
      total_opt_epoch: physicsSettings.totalOptEpoch,
      phys_lr: physicsSettings.physLr,
      contact_d_hat: physicsSettings.contactDHat,
      contact_eps_velocity: physicsSettings.contactEpsVelocity,
    },
    llm_model: llmModel,
    image_model: imageModel,
    object_crop_completion_enabled: objectCropCompletionEnabled,
    object_crop_completion_model: objectCropCompletionModel,
    structured_llm_max_attempts: structuredLlmMaxAttempts,
    structured_llm_reasoning_budget: structuredLlmReasoningBudget,
    reasoning_effort: reasoningEffort,
    runtime_output_name: outputName,
    current_stage_id: null,
    stages: [],
    started_at: inputPayload.started_at,
    updated_at: inputPayload.started_at,
    finished_at: null,
    error: null,
    log_name: path.basename(logPath),
    manual_segmentation: null,
    manual_size_priors: null,
    relation_graph_override: null,
  });

  await startJobWorker({
    jobId,
    inputPath,
    statusPath,
    outputPath,
    runtimeConfigPath,
    logPath,
  });
  return loadJob(jobId);
}

async function stopTrackedJobWorker(jobId, workerPid) {
  const trackedChild = runningChildren.get(jobId);
  if (trackedChild && typeof trackedChild.kill === 'function') {
    if (trackedChild.exitCode !== null || trackedChild.signalCode !== null) {
      return;
    }
    await new Promise((resolve) => {
      let resolved = false;
      const finish = () => {
        if (resolved) return;
        resolved = true;
        resolve();
      };
      trackedChild.once('close', finish);
      trackedChild.once('error', finish);
      try {
        trackedChild.kill('SIGTERM');
      } catch {
        finish();
        return;
      }
      globalThis.setTimeout(() => {
        if (trackedChild.exitCode !== null || trackedChild.signalCode !== null) {
          finish();
          return;
        }
        try {
          trackedChild.kill('SIGKILL');
        } catch {
          // ignore hard-kill failures
        }
      }, 5000);
    });
    return;
  }

  if (!Number.isInteger(workerPid) || workerPid <= 0 || !isProcessAlive(workerPid)) {
    return;
  }
  try {
    process.kill(workerPid, 'SIGTERM');
  } catch {
    return;
  }
  const startedAt = Date.now();
  while (isProcessAlive(workerPid) && Date.now() - startedAt < 5000) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  if (isProcessAlive(workerPid)) {
    try {
      process.kill(workerPid, 'SIGKILL');
    } catch {
      // ignore hard-kill failures
    }
  }
}

async function cancelJob(jobId) {
  await ensureDirectories();
  const safeId = path.basename(jobId);
  const statusPath = path.join(jobDir, `${safeId}.status.json`);
  const statusPayload = await readJson(statusPath);

  if (!isActiveJobState(statusPayload.state)) {
    throw new ApiError(
      409,
      'job_not_cancellable',
      'Only the current queued, running, or manual-input run can be cancelled.',
      `job is currently in state "${statusPayload.state}"`,
    );
  }

  const trackedChild = runningChildren.get(safeId);
  const trackedChildPid = trackedChild?.pid ?? null;
  const currentWorkerPid = Number.parseInt(String(statusPayload.worker_pid || trackedChildPid || ''), 10);
  const updatedAt = new Date().toISOString();

  statusPayload.state = 'cancelled';
  statusPayload.updated_at = updatedAt;
  statusPayload.finished_at = updatedAt;
  statusPayload.error = null;
  statusPayload.worker_pid = null;
  statusPayload.interruption = Number.isInteger(currentWorkerPid) && currentWorkerPid > 0
    ? {
        expected_exit_pid: currentWorkerPid,
        reason: 'cancelled',
        requested_at: updatedAt,
      }
    : null;
  statusPayload.stages = Array.isArray(statusPayload.stages)
    ? statusPayload.stages.map((stage) => (
        ['queued', 'running', 'awaiting_input'].includes(stage.status)
          ? {
              ...stage,
              status: 'cancelled',
              completed_at: updatedAt,
              error: null,
              error_detail: null,
              retryable: false,
              last_log: 'Cancelled by user.',
            }
          : stage
      ))
    : [];

  await writeJson(statusPath, statusPayload);
  await stopTrackedJobWorker(safeId, currentWorkerPid);
  return loadJob(safeId);
}

async function persistRelationGraph(jobId, payload) {
  await ensureDirectories();
  const safeId = path.basename(jobId);
  const statusPath = path.join(jobDir, `${safeId}.status.json`);
  const statusPayload = await readJson(statusPath);

  const {
    inputPath,
    outputPath,
  } = buildJobPaths(safeId, statusPayload.scene_id, statusPayload.runtime_output_name);

  if (!existsSync(outputPath)) {
    throw new ApiError(
      409,
      'resume_reference_missing',
      'No saved runtime output is available for relation edits.',
      'runtime output file does not exist yet',
    );
  }

  let runtimePayload;
  try {
    runtimePayload = await readJson(outputPath);
  } catch (error) {
    throw new ApiError(
      409,
      'resume_runtime_invalid',
      'The saved runtime output could not be read for relation edits.',
      error?.message || 'runtime output is invalid',
    );
  }

  const objectCatalog = objectCatalogFromRuntime(runtimePayload);
  const currentGraph = relationGraphFromRuntime(runtimePayload);
  if (!objectCatalog || !Array.isArray(objectCatalog.objects)) {
    throw new ApiError(
      409,
      'relation_graph_source_missing',
      'No stage-3 object catalog is available for relation editing.',
      'first_contract_slice.scene_understanding.object_catalog is missing from the runtime output',
    );
  }
  if (!currentGraph || !Array.isArray(currentGraph.relations)) {
    throw new ApiError(
      409,
      'relation_graph_source_missing',
      'No stage-3 relation graph is available for editing.',
      'first_contract_slice.object_relation.relation_graph is missing from the runtime output',
    );
  }

  const nextRelationGraph = normalizeRelationGraphPayload(payload, {
    sceneId: statusPayload.scene_id,
    objectIds: objectCatalog.objects.map((object) => object.object_id),
    currentRootObjectIds: Array.isArray(currentGraph.root_object_ids) ? currentGraph.root_object_ids : [],
  });

  const inputPayload = await readJson(inputPath);
  const updatedAt = new Date().toISOString();
  const relationGraphOverride = nextRelationGraph
    ? {
        relation_graph: nextRelationGraph,
        updated_at: updatedAt,
      }
    : null;

  inputPayload.relation_graph_override = relationGraphOverride;
  statusPayload.relation_graph_override = relationGraphOverride;
  statusPayload.updated_at = updatedAt;

  await writeJson(inputPath, inputPayload);
  await writeJson(statusPath, statusPayload);
  return loadJob(safeId);
}

async function persistManualMasks(jobId, payload) {
  await ensureDirectories();
  const safeId = path.basename(jobId);
  const statusPath = path.join(jobDir, `${safeId}.status.json`);
  const statusPayload = await readJson(statusPath);

  if (statusPayload.segmentation_mode !== 'manual') {
    throw new ApiError(
      409,
      'manual_masks_not_enabled',
      'This job is not configured for hand-tuned masking.',
      'manual masks can only be saved for jobs created with segmentationMode="manual"',
    );
  }

  if (statusPayload.state !== 'awaiting_mask_input') {
    throw new ApiError(
      409,
      'job_not_waiting_for_masks',
      'This job is not ready for manual masks.',
      `job is currently in state "${statusPayload.state}"`,
    );
  }

  const instances = normalizeManualMaskInstances(payload?.instances || []);
  const {
    inputPath,
    manualMaskDir,
  } = buildJobPaths(safeId, statusPayload.scene_id, statusPayload.runtime_output_name);

  await fs.rm(manualMaskDir, { recursive: true, force: true });
  await fs.mkdir(manualMaskDir, { recursive: true });

  const savedInstances = [];
  for (const instance of instances) {
    const filePath = path.join(manualMaskDir, `${instance.instance_id}.png`);
    await fs.writeFile(filePath, instance.maskBuffer);
    savedInstances.push({
      instance_id: instance.instance_id,
      label: instance.label,
      color: instance.color,
      width: instance.width,
      height: instance.height,
      mask_path: path.relative(repoRoot, filePath),
    });
  }

  const manualSegmentation = {
    state: savedInstances.length ? 'saved' : 'pending',
    reference_image_path: statusPayload.manual_segmentation?.reference_image_path || null,
    preserved_depth_result: cloneJsonValue(statusPayload.manual_segmentation?.preserved_depth_result || null),
    updated_at: new Date().toISOString(),
    instances: savedInstances,
  };

  const inputPayload = await readJson(inputPath);
  inputPayload.manual_segmentation = manualSegmentation;
  statusPayload.manual_segmentation = manualSegmentation;
  statusPayload.updated_at = manualSegmentation.updated_at;

  await writeJson(inputPath, inputPayload);
  await writeJson(statusPath, statusPayload);
  return loadJob(safeId);
}

async function continueSizePriorsJob(jobId, payload) {
  await ensureDirectories();
  const safeId = path.basename(jobId);
  const statusPath = path.join(jobDir, `${safeId}.status.json`);
  const statusPayload = await readJson(statusPath);

  if (statusPayload.state !== 'awaiting_size_input') {
    throw new ApiError(
      409,
      'job_not_waiting_for_sizes',
      'This job is not waiting for manual size priors.',
      `job is currently in state "${statusPayload.state}"`,
    );
  }

  const {
    inputPath,
    runtimeConfigPath,
    logPath,
    outputPath,
  } = buildJobPaths(safeId, statusPayload.scene_id, statusPayload.runtime_output_name);
  const inputPayload = await readJson(inputPath);
  const existingManualSizes = statusPayload.manual_size_priors || inputPayload.manual_size_priors;
  const existingEntries = Array.isArray(existingManualSizes?.entries) ? existingManualSizes.entries : [];
  const submittedEntries = normalizeManualSizeEntries(payload?.entries || existingEntries);
  if (!submittedEntries.length) {
    throw new ApiError(
      422,
      'manual_sizes_required',
      'At least one manual size prior is required before continuing.',
      'manual size entry list is empty',
    );
  }

  const updatedAt = new Date().toISOString();
  const manualSizePriors = {
    state: 'submitted',
    updated_at: updatedAt,
    entries: submittedEntries,
    error: existingManualSizes?.error || statusPayload.error || null,
  };
  inputPayload.manual_size_priors = manualSizePriors;
  statusPayload.manual_size_priors = manualSizePriors;
  statusPayload.state = 'queued';
  statusPayload.current_stage_id = 'object-assets';
  statusPayload.updated_at = updatedAt;
  statusPayload.finished_at = null;
  statusPayload.error = null;
  statusPayload.worker_pid = null;
  statusPayload.interruption = null;
  statusPayload.stages = prepareStageRowsForResume(statusPayload.stages, 'object-assets');

  await writeJson(inputPath, inputPayload);
  await writeJson(statusPath, statusPayload);
  await startJobWorker({
    jobId: safeId,
    inputPath,
    statusPath,
    outputPath,
    runtimeConfigPath,
    logPath,
    resumeFrom: 'object-assets',
  });
  return loadJob(safeId);
}

async function continueManualJob(jobId, payload) {
  await ensureDirectories();
  const safeId = path.basename(jobId);
  const statusPath = path.join(jobDir, `${safeId}.status.json`);
  const statusPayload = await readJson(statusPath);

  if (statusPayload.segmentation_mode !== 'manual') {
    throw new ApiError(
      409,
      'manual_masks_not_enabled',
      'This job is not configured for hand-tuned masking.',
      'manual resume can only be used for jobs created with segmentationMode="manual"',
    );
  }

  if (statusPayload.state !== 'awaiting_mask_input') {
    throw new ApiError(
      409,
      'job_not_waiting_for_masks',
      'This job is not waiting for manual masks.',
      `job is currently in state "${statusPayload.state}"`,
    );
  }

  if (Array.isArray(payload?.instances)) {
    await persistManualMasks(safeId, payload);
  }

  const {
    inputPath,
    runtimeConfigPath,
    logPath,
    outputPath,
  } = buildJobPaths(safeId, statusPayload.scene_id, statusPayload.runtime_output_name);
  const inputPayload = await readJson(inputPath);
  const manualSegmentation = inputPayload.manual_segmentation;
  if (!manualSegmentation || !Array.isArray(manualSegmentation.instances) || !manualSegmentation.instances.length) {
    throw new ApiError(
      422,
      'manual_masks_required',
      'At least one manual mask is required before continuing.',
      'manual mask list is empty',
    );
  }

  const stages = Array.isArray(statusPayload.stages) ? statusPayload.stages.map((stage) => (
    stage.id === 'scene-understanding'
      ? { ...stage, status: 'queued', error: null, error_detail: null, retryable: false }
      : stage
  )) : [];
  const updatedAt = new Date().toISOString();
  statusPayload.state = 'queued';
  statusPayload.current_stage_id = 'scene-understanding';
  statusPayload.updated_at = updatedAt;
  statusPayload.finished_at = null;
  statusPayload.error = null;
  statusPayload.stages = stages;
  statusPayload.manual_segmentation = {
    ...manualSegmentation,
    state: 'submitted',
    updated_at: updatedAt,
  };
  inputPayload.manual_segmentation = statusPayload.manual_segmentation;

  await writeJson(inputPath, inputPayload);
  await writeJson(statusPath, statusPayload);
  await startJobWorker({
    jobId: safeId,
    inputPath,
    statusPath,
    outputPath,
    runtimeConfigPath,
    logPath,
    resumeManual: true,
  });
  return loadJob(safeId);
}

async function redoSegmentationManually(jobId) {
  await ensureDirectories();
  const safeId = path.basename(jobId);
  const statusPath = path.join(jobDir, `${safeId}.status.json`);
  const statusPayload = await readJson(statusPath);
  const activeJob = await findActiveJob({ excludeJobId: safeId });
  if (activeJob) {
    throw new ApiError(
      409,
      'active_job_exists',
      'Finish or retry the current run before reopening a different run for manual segmentation.',
      `job "${activeJob.scene_id}" (${activeJob.job_id}) is currently ${activeJob.state}`,
      true,
    );
  }

  if (!['completed', 'failed'].includes(statusPayload.state)) {
    throw new ApiError(
      409,
      'job_not_redoable',
      'This job cannot be reopened for manual segmentation right now.',
      `job is currently in state "${statusPayload.state}"`,
    );
  }

  const {
    inputPath,
    outputPath,
    manualMaskDir,
  } = buildJobPaths(safeId, statusPayload.scene_id, statusPayload.runtime_output_name);

  if (!existsSync(outputPath)) {
    throw new ApiError(
      409,
      'resume_reference_missing',
      'No saved reference image result is available for this job.',
      'runtime output file does not exist yet',
    );
  }

  let runtimePayload;
  try {
    runtimePayload = await readJson(outputPath);
  } catch (error) {
    throw new ApiError(
      409,
      'resume_runtime_invalid',
      'The saved runtime output could not be read for manual segmentation.',
      error?.message || 'runtime output is invalid',
    );
  }

  const referenceImageResult = referenceImageResultFromRuntime(runtimePayload);
  const referenceImagePath = referenceImageResult?.image?.path || null;
  const depthResult = depthResultFromRuntime(runtimePayload);
  if (!referenceImagePath) {
    throw new ApiError(
      409,
      'resume_reference_missing',
      'No saved reference image result is available for this job.',
      'first_contract_slice.reference_image_result is missing from the runtime output',
    );
  }

  const inputPayload = await readJson(inputPath);
  const updatedAt = new Date().toISOString();
  const requestedObjects = Array.isArray(statusPayload.requested_objects)
    ? statusPayload.requested_objects
    : Array.isArray(inputPayload.requested_objects)
      ? inputPayload.requested_objects
      : [];
  const requestedObjectsInferred = typeof statusPayload.requested_objects_inferred === 'boolean'
    ? statusPayload.requested_objects_inferred
    : Boolean(inputPayload.requested_objects_inferred);
  const manualSegmentation = {
    state: 'pending',
    reference_image_path: referenceImagePath,
    preserved_depth_result: cloneJsonValue(depthResult),
    updated_at: updatedAt,
    instances: [],
  };

  inputPayload.segmentation_mode = 'manual';
  inputPayload.requested_objects = requestedObjects;
  inputPayload.requested_objects_inferred = requestedObjectsInferred;
  inputPayload.manual_segmentation = manualSegmentation;

  statusPayload.state = 'awaiting_mask_input';
  statusPayload.current_stage_id = 'scene-understanding';
  statusPayload.updated_at = updatedAt;
  statusPayload.finished_at = null;
  statusPayload.error = null;
  statusPayload.worker_pid = null;
  statusPayload.segmentation_mode = 'manual';
  statusPayload.requested_objects = requestedObjects;
  statusPayload.requested_objects_inferred = requestedObjectsInferred;
  statusPayload.manual_segmentation = manualSegmentation;
  statusPayload.stages = buildManualCheckpointStageRows(statusPayload.stages, updatedAt);

  await fs.rm(manualMaskDir, { recursive: true, force: true });
  await writeJson(outputPath, {
    first_contract_slice: {
      reference_image_result: cloneJsonValue(referenceImageResult),
      preserved_depth_result: cloneJsonValue(depthResult),
    },
  });
  await writeJson(inputPath, inputPayload);
  await writeJson(statusPath, statusPayload);
  return loadJob(safeId);
}

function prepareStageRowsForResume(stageRows, resumeFrom) {
  if (!Array.isArray(stageRows) || !stageRows.length) {
    return [];
  }
  const resumeIndex = stageRows.findIndex((stage) => stage?.id === resumeFrom);
  if (resumeIndex < 0) {
    throw new ApiError(
      422,
      'resume_stage_unknown',
      'The requested resume stage is not available for this job.',
      `unknown resume stage "${resumeFrom}"`,
    );
  }

  const now = new Date().toISOString();
  return stageRows.map((stage, index) => {
    const nextStage = { ...stage, error: null, error_detail: null, retryable: false };
    if (index < resumeIndex) {
      return {
        ...nextStage,
        status: 'completed',
        progress: 100,
        progress_total: nextStage.progress_total || 100,
        progress_completed: nextStage.progress_total || 100,
        completed_at: nextStage.completed_at || now,
        last_log: nextStage.last_log || 'Completed.',
      };
    }
    if (index === resumeIndex) {
      return {
        ...nextStage,
        status: 'queued',
        started_at: null,
        completed_at: null,
        progress: null,
        progress_total: null,
        progress_completed: null,
        last_log: `Queued to retry stage '${resumeFrom}'.`,
      };
    }
    return {
      ...nextStage,
      status: 'pending',
      started_at: null,
      completed_at: null,
      progress: null,
      progress_total: null,
      progress_completed: null,
      last_log: null,
    };
  });
}

function validateRuntimeForRetryStage(runtimePayload, retryStage) {
  if (retryStage === 'reference-image') {
    return;
  }
  const requirementsByStage = {
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
  const requirements = requirementsByStage[retryStage] || [];
  const missing = requirements.filter((pathKey) => {
    const parts = pathKey.split('.');
    let current = runtimePayload;
    for (const part of parts) {
      if (!current || typeof current !== 'object' || !(part in current)) {
        return true;
      }
      current = current[part];
    }
    return current == null;
  });
  if (!missing.length) {
    return;
  }
  throw new ApiError(
    409,
    'resume_prerequisite_missing',
    'The saved runtime does not contain the upstream outputs required to retry that stage.',
    `missing prerequisite outputs for "${retryStage}": ${missing.join(', ')}`,
  );
}

async function resumeJobFromStage(jobId, payload) {
  await ensureDirectories();
  const safeId = path.basename(jobId);
  const statusPath = path.join(jobDir, `${safeId}.status.json`);
  const statusPayload = await readJson(statusPath);
  const resumeFrom = typeof payload?.resumeFrom === 'string' ? payload.resumeFrom.trim() : '';
  const activeJob = await findActiveJob({ excludeJobId: safeId });
  if (activeJob) {
    throw new ApiError(
      409,
      'active_job_exists',
      'Finish or retry the current run before restarting a different run.',
      `job "${activeJob.scene_id}" (${activeJob.job_id}) is currently ${activeJob.state}`,
      true,
    );
  }

  const knownStageIds = new Set(JOB_STAGE_BLUEPRINTS.map((stage) => stage.id));
  if (!knownStageIds.has(resumeFrom)) {
    throw new ApiError(
      422,
      'resume_stage_unknown',
      'The requested retry stage is not available for this job.',
      `unknown retry stage "${resumeFrom || '<empty>'}"`,
    );
  }

  if (!['failed', 'completed', 'cancelled', 'queued', 'running', 'awaiting_mask_input', 'awaiting_size_input'].includes(statusPayload.state)) {
    throw new ApiError(
      409,
      'job_not_resumable',
      'This job cannot be resumed right now.',
      `job is currently in state "${statusPayload.state}"`,
    );
  }

  const {
    inputPath,
    runtimeConfigPath,
    logPath,
    outputPath,
  } = buildJobPaths(safeId, statusPayload.scene_id, statusPayload.runtime_output_name);

  const outputExists = existsSync(outputPath);
  if (resumeFrom !== 'reference-image' && !outputExists) {
    throw new ApiError(
      409,
      'resume_reference_missing',
      'No saved runtime output is available for that retry stage.',
      'runtime output file does not exist yet',
    );
  }

  let runtimePayload;
  if (resumeFrom !== 'reference-image') {
    try {
      runtimePayload = await readJson(outputPath);
    } catch (error) {
      throw new ApiError(
        409,
        'resume_runtime_invalid',
        'The saved runtime output could not be read for retry.',
        error?.message || 'runtime output is invalid',
      );
    }
    validateRuntimeForRetryStage(runtimePayload, resumeFrom);
  }

  const inputPayload = await readJson(inputPath);
  const hasRetryPhysicsSettings = payload?.physicsSettings || payload?.physics_settings;
  if (hasRetryPhysicsSettings) {
    const physicsSettings = sanitizePhysicsSettings(payload?.physicsSettings ?? payload?.physics_settings);
    const persistedPhysicsSettings = {
      diff_sim_enabled: physicsSettings.diffSimEnabled,
      end_frame: physicsSettings.endFrame,
      ground_y_value: physicsSettings.groundYValue,
      total_opt_epoch: physicsSettings.totalOptEpoch,
      phys_lr: physicsSettings.physLr,
      contact_d_hat: physicsSettings.contactDHat,
      contact_eps_velocity: physicsSettings.contactEpsVelocity,
    };
    inputPayload.physics_settings = persistedPhysicsSettings;
    statusPayload.physics_settings = persistedPhysicsSettings;
  }
  const updatedAt = new Date().toISOString();
  const currentWorkerPid = Number.parseInt(String(statusPayload.worker_pid || ''), 10);
  statusPayload.state = 'queued';
  statusPayload.current_stage_id = resumeFrom;
  statusPayload.updated_at = updatedAt;
  statusPayload.finished_at = null;
  statusPayload.error = null;
  statusPayload.interruption = Number.isInteger(currentWorkerPid) && currentWorkerPid > 0
    ? {
        expected_exit_pid: currentWorkerPid,
        reason: 'stage_retry',
        requested_at: updatedAt,
        resume_from: resumeFrom,
      }
    : null;
  statusPayload.worker_pid = null;
  statusPayload.stages = prepareStageRowsForResume(statusPayload.stages, resumeFrom);
  const resumeStageIndex = JOB_STAGE_BLUEPRINTS.findIndex((stage) => stage.id === resumeFrom);

  await writeJson(inputPath, inputPayload);
  await writeJson(statusPath, statusPayload);
  if (Number.isInteger(currentWorkerPid) && currentWorkerPid > 0) {
    await stopTrackedJobWorker(safeId, currentWorkerPid);
  }
  await startJobWorker({
    jobId: safeId,
    inputPath,
    statusPath,
    outputPath,
    runtimeConfigPath,
    logPath,
    resumeFrom,
  });
  return loadJob(safeId);
}

app.use((req, res, next) => {
  const remoteAddress = normalizeRemoteAddress(req.socket?.remoteAddress);
  if (!isAllowedRemoteAddress(remoteAddress)) {
    res.status(403).json({
      error: allowPrivateSubnetAccess
        ? 'Dashboard access is restricted to localhost and approved private subnets.'
        : 'Dashboard access is restricted to localhost and 192.168.0.0/16.',
    });
    return;
  }

  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-site');
  if (req.path.startsWith('/api/')) {
    res.setHeader('Cache-Control', 'no-store, max-age=0, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
  }
  next();
});

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, mode: isProduction ? 'production' : 'development', host, port });
});

app.get('/api/runtimes', async (_req, res) => {
  try {
    const runtimes = await listRuntimeFiles();
    res.json({ runtimes });
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'runtime_list_failed',
      userMessage: 'Could not list runtime outputs.',
    });
  }
});

app.get('/api/runtime', async (req, res) => {
  try {
    const payload = await loadRuntimeByName(req.query.name);
    res.json(payload);
  } catch (error) {
    if (error?.code === 'ENOENT') {
      const runtimeName = path.basename(typeof req.query?.name === 'string' ? req.query.name : '');
      const activeJob = runtimeName ? await findJobByRuntimeName(runtimeName) : null;
      const progress = extractJobRuntimeProgress(activeJob || {});
      const runtimeMessage = activeJob?.state === 'running' || activeJob?.state === 'queued' || activeJob?.state === 'awaiting_mask_input' || activeJob?.state === 'awaiting_size_input'
        ? 'Runtime output is still being produced while the job is running.'
        : 'The selected runtime output has not been written yet.';
      res.status(202).json({
        ok: false,
        name: runtimeName,
        data: null,
        pending: true,
        job_id: activeJob?.job_id || null,
        job_state: activeJob?.state || null,
        active_stage: progress.stageName || null,
        active_stage_progress: progress.stageProgress ?? null,
        runtime_updated_at: progress.updatedAt,
        error: {
          code: 'runtime_pending',
          userMessage: runtimeMessage,
          detail: error.message,
          retryable: true,
        },
      });
      return;
    }
    sendApiError(res, error, {
      status: error instanceof SyntaxError ? 500 : 404,
      code: error instanceof SyntaxError ? 'runtime_invalid_json' : 'runtime_load_failed',
      userMessage: error instanceof SyntaxError
        ? 'The selected runtime JSON is malformed.'
        : 'Could not load the selected runtime output.',
    });
  }
});

app.get('/api/trajectory', async (req, res) => {
  try {
    const runtimeName = typeof req.query?.runtime === 'string' ? req.query.runtime.trim() : '';
    if (!runtimeName) {
      throw new ApiError(422, 'trajectory_runtime_missing', 'A runtime name is required to load trajectory data.', 'runtime query parameter is missing');
    }
    const runtimePath = path.join(runtimeDir, path.basename(runtimeName));
    if (!existsSync(runtimePath)) {
      throw new ApiError(404, 'trajectory_runtime_missing', 'The selected runtime output does not exist.', runtimePath);
    }
    const payload = await runPythonJsonScript(
      pythonExecutable,
      [trajectoryScript, '--runtime', runtimePath],
    );
    res.json(payload);
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'trajectory_load_failed',
      userMessage: 'Could not load simulation trajectory data for this runtime.',
    });
  }
});

app.get('/api/metrics', async (req, res) => {
  try {
    const runtimeName = typeof req.query?.runtime === 'string' ? req.query.runtime.trim() : '';
    const payload = await loadMetricsForRuntime(runtimeName);
    res.json(payload);
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'metrics_load_failed',
      userMessage: 'Could not load metrics for this runtime.',
    });
  }
});

app.get('/api/physics-metrics', async (req, res) => {
  try {
    const runtimeName = typeof req.query?.runtime === 'string' ? req.query.runtime.trim() : '';
    if (!runtimeName) {
      throw new ApiError(422, 'physics_metrics_runtime_missing', 'A runtime name is required to load physics metrics.', 'runtime query parameter is missing');
    }
    const runtimePath = path.join(runtimeDir, path.basename(runtimeName));
    if (!existsSync(runtimePath)) {
      throw new ApiError(404, 'physics_metrics_runtime_missing', 'The selected runtime output does not exist.', runtimePath);
    }
    const payload = await runPythonJsonScript(
      pythonExecutable,
      [physicsMetricsScript, '--runtime', runtimePath],
    );
    res.json(payload);
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'physics_metrics_load_failed',
      userMessage: 'Could not load physics loss history for this runtime.',
    });
  }
});

app.post('/api/metrics', async (req, res) => {
  try {
    const payload = await computeMetricsForRuntime(req.body || {});
    res.json({
      available: true,
      scene_id: payload.case_id || null,
      metrics_path: payload.metrics_result_path ? path.relative(repoRoot, payload.metrics_result_path) : null,
      metrics: payload,
    });
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'metrics_compute_failed',
      userMessage: 'Could not compute metrics for this runtime.',
    });
  }
});

app.get('/api/jobs/:jobId/physics-metrics', async (req, res) => {
  try {
    const safeId = path.basename(req.params.jobId);
    const statusPath = path.join(jobDir, `${safeId}.status.json`);
    if (!existsSync(statusPath)) {
      throw new ApiError(404, 'job_missing', 'The requested dashboard job does not exist.', statusPath);
    }
    const statusPayload = await readJson(statusPath);
    const sceneId = typeof statusPayload?.scene_id === 'string' ? statusPayload.scene_id.trim() : '';
    if (!sceneId) {
      throw new ApiError(422, 'physics_metrics_scene_missing', 'The job does not have a scene id yet.', 'scene_id missing from job status');
    }
    const runtimeOutputName = typeof statusPayload?.runtime_output_name === 'string'
      ? statusPayload.runtime_output_name.trim()
      : '';
    const runtimePath = runtimeOutputName
      ? path.join(runtimeDir, path.basename(runtimeOutputName))
      : '';
    const scriptArgs = runtimePath && existsSync(runtimePath)
      ? [physicsMetricsScript, '--runtime', runtimePath]
      : [physicsMetricsScript, '--scene-id', sceneId];
    const payload = await runPythonJsonScript(
      pythonExecutable,
      scriptArgs,
    );
    res.json(payload);
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'job_physics_metrics_load_failed',
      userMessage: 'Could not load physics loss history for this dashboard job.',
    });
  }
});

app.get('/api/jobs', async (_req, res) => {
  try {
    const jobs = await listJobs();
    res.json({ jobs });
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'job_list_failed',
      userMessage: 'Could not list dashboard jobs.',
    });
  }
});

app.get('/api/jobs/:jobId', async (req, res) => {
  try {
    const job = await loadJob(req.params.jobId);
    res.json(job);
  } catch (error) {
    sendApiError(res, error, {
      status: 404,
      code: 'job_load_failed',
      userMessage: 'Could not load the selected dashboard job.',
    });
  }
});

app.get('/api/jobs/:jobId/log', async (req, res) => {
  try {
    const tail = Number.parseInt(String(req.query.tail || '200'), 10);
    const log = await readJobLog(req.params.jobId, Number.isFinite(tail) ? tail : 200);
    res.json({ job_id: path.basename(req.params.jobId), log });
  } catch (error) {
    sendApiError(res, error, {
      status: 404,
      code: 'job_log_load_failed',
      userMessage: 'Could not load the dashboard job log.',
    });
  }
});

app.get('/api/local-cache/summary', async (_req, res) => {
  try {
    const summary = await localCacheSummary();
    res.json({ ok: true, summary });
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'local_cache_summary_failed',
      userMessage: 'Could not inspect the local dashboard cache.',
    });
  }
});

app.post('/api/local-cache/cleanup', async (_req, res) => {
  try {
    const result = await clearLocalRunCaches();
    res.json({ ok: true, result });
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'local_cache_cleanup_failed',
      userMessage: 'Could not clear the local dashboard cache.',
    });
  }
});

app.post('/api/run', async (req, res) => {
  try {
    const job = await createJob(req.body || {});
    res.status(202).json(job);
  } catch (error) {
    const detail = error?.message || 'Could not create dashboard job.';
    const isValidationError = detail.includes('prompt must be a non-empty string') || detail.includes('Invalid backend');
    sendApiError(
      res,
      isValidationError
        ? new ApiError(422, 'invalid_request', 'The dashboard job request is invalid.', detail, false)
        : error,
      {
        status: 500,
        code: 'job_create_failed',
        userMessage: 'Could not create the dashboard job.',
      },
    );
  }
});

app.post('/api/jobs/:jobId/manual-masks', async (req, res) => {
  try {
    const job = await persistManualMasks(req.params.jobId, req.body || {});
    res.json(job);
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'manual_mask_save_failed',
      userMessage: 'Could not save the manual masks for this job.',
    });
  }
});

app.post('/api/jobs/:jobId/relation-graph', async (req, res) => {
  try {
    const job = await persistRelationGraph(req.params.jobId, req.body || {});
    res.json(job);
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'relation_graph_save_failed',
      userMessage: 'Could not save the edited relation graph for this job.',
    });
  }
});

app.post('/api/jobs/:jobId/continue', async (req, res) => {
  try {
    const job = await continueManualJob(req.params.jobId, req.body || {});
    res.json(job);
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'manual_job_continue_failed',
      userMessage: 'Could not continue the manual-masking job.',
    });
  }
});

app.post('/api/jobs/:jobId/continue-size-priors', async (req, res) => {
  try {
    const job = await continueSizePriorsJob(req.params.jobId, req.body || {});
    res.json(job);
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'manual_size_continue_failed',
      userMessage: 'Could not continue the run with manual size priors.',
    });
  }
});

app.post('/api/jobs/:jobId/redo-segmentation-manually', async (req, res) => {
  try {
    const job = await redoSegmentationManually(req.params.jobId);
    res.json(job);
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'job_manual_redo_failed',
      userMessage: 'Could not reopen the job for manual segmentation.',
    });
  }
});

app.post('/api/jobs/:jobId/resume', async (req, res) => {
  try {
    const job = await resumeJobFromStage(req.params.jobId, req.body || {});
    res.json(job);
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'job_resume_failed',
      userMessage: 'Could not resume the dashboard job from the requested stage.',
    });
  }
});

app.post('/api/jobs/:jobId/cancel', async (req, res) => {
  try {
    const job = await cancelJob(req.params.jobId);
    res.json(job);
  } catch (error) {
    sendApiError(res, error, {
      status: 500,
      code: 'job_cancel_failed',
      userMessage: 'Could not cancel the current run.',
    });
  }
});

app.get('/api/artifact', async (req, res) => {
  try {
    const absolutePath = resolveArtifactPath(req.query.path);
    if (!absolutePath) {
      throw new ApiError(400, 'artifact_not_allowed', 'Artifact path is not allowed.', 'Artifact path is not allowed.', false);
    }

    const stat = await fs.stat(absolutePath);
    if (!stat.isFile()) {
      throw new ApiError(404, 'artifact_not_found', 'Artifact is not a file.', 'Artifact is not a file.', false);
    }

    res.sendFile(absolutePath);
  } catch (error) {
    sendApiError(res, error, {
      status: 404,
      code: 'artifact_load_failed',
      userMessage: 'Could not serve the requested artifact.',
    });
  }
});

app.post('/api/scene-export', async (req, res) => {
  let tempDir = '';
  const cleanup = async () => {
    if (!tempDir) {
      return;
    }
    const target = tempDir;
    tempDir = '';
    await fs.rm(target, { recursive: true, force: true }).catch(() => {});
  };

  try {
    const mode = parseSceneExportMode(req.body?.mode);
    if (!mode) {
      throw new ApiError(
        422,
        'scene_export_invalid_mode',
        'The requested export mode is invalid.',
        'scene export mode must be either "merged" or "separate"',
      );
    }

    const bundle = await loadSceneBundleForExport(req.body || {});
    const tempRoot = path.join(repoRoot, 'tmp');
    await fs.mkdir(tempRoot, { recursive: true });
    tempDir = await fs.mkdtemp(path.join(tempRoot, 'scene-export-'));

    const bundlePath = path.join(tempDir, 'scene_bundle.json');
    const zipPath = path.join(tempDir, `${bundle.scene_id}-${mode}.zip`);
    await writeJson(bundlePath, bundle);
    const exportInfo = await runPythonJsonScript(
      pythonExecutable,
      [
        sceneExportScript,
        '--bundle',
        bundlePath,
        '--mode',
        mode,
        '--output',
        zipPath,
      ],
    );

    const resolvedZipPath = typeof exportInfo?.zip_path === 'string' ? exportInfo.zip_path : zipPath;
    const downloadName = typeof exportInfo?.download_name === 'string'
      ? exportInfo.download_name
      : `${bundle.scene_id}-${mode}.zip`;

    res.on('finish', () => {
      void cleanup();
    });
    res.on('close', () => {
      void cleanup();
    });
    res.download(resolvedZipPath, downloadName);
  } catch (error) {
    await cleanup();
    sendApiError(res, error, {
      status: 500,
      code: 'scene_export_failed',
      userMessage: 'Could not export the stage-8 scene bundle.',
    });
  }
});

async function bootstrap() {
  await ensureDirectories();

  if (isProduction) {
    const distDir = path.join(appRoot, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const httpServer = http.createServer(app);
    const vite = await createViteServer({
      root: appRoot,
      appType: 'custom',
      server: {
        middlewareMode: true,
        hmr: {
          server: httpServer,
        },
      },
    });

    app.use(vite.middlewares);
    app.use('*', async (req, res, next) => {
      try {
        const url = req.originalUrl;
        let template = await fs.readFile(path.join(appRoot, 'index.html'), 'utf8');
        template = await vite.transformIndexHtml(url, template);
        res.status(200).set({ 'Content-Type': 'text/html' }).end(template);
      } catch (error) {
        vite.ssrFixStacktrace(error);
        next(error);
      }
    });

    httpServer.listen(port, host, () => {
      console.log(`PAT3D dashboard server listening on http://${host}:${port}`);
    });
    return;
  }

  app.listen(port, host, () => {
    console.log(`PAT3D dashboard server listening on http://${host}:${port}`);
  });
}

bootstrap().catch((error) => {
  console.error(error);
  process.exit(1);
});
