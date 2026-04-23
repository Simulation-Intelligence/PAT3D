const DEFAULT_DASHBOARD_API_PORT = '4173';
const LOCAL_DEV_HOST_PATTERN = /^(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3})$/;
const VITE_DEV_PORT_MIN = 5173;
const VITE_DEV_PORT_MAX = 5199;

function normalizeConfiguredBaseUrl(value) {
  if (typeof value !== 'string') {
    return '';
  }
  return value.trim().replace(/\/+$/, '');
}

function isAbsoluteUrl(value) {
  return typeof value === 'string' && /^[a-z][a-z0-9+.-]*:\/\//i.test(value);
}

export function resolveApiBaseUrl(
  locationLike = typeof window !== 'undefined' ? window.location : null,
  env = typeof import.meta !== 'undefined' ? import.meta.env : undefined,
) {
  const configuredBaseUrl = normalizeConfiguredBaseUrl(env?.VITE_PAT3D_API_BASE_URL);
  if (configuredBaseUrl) {
    return configuredBaseUrl;
  }

  const hostname = typeof locationLike?.hostname === 'string' ? locationLike.hostname : '';
  const port = typeof locationLike?.port === 'string' ? locationLike.port : '';
  const protocol = typeof locationLike?.protocol === 'string' ? locationLike.protocol : 'http:';
  const numericPort = Number.parseInt(port, 10);

  if (!hostname || !LOCAL_DEV_HOST_PATTERN.test(hostname) || !Number.isFinite(numericPort)) {
    return '';
  }

  if (numericPort < VITE_DEV_PORT_MIN || numericPort > VITE_DEV_PORT_MAX) {
    return '';
  }

  return `${protocol}//${hostname}:${DEFAULT_DASHBOARD_API_PORT}`;
}

export function apiUrl(path, options = {}) {
  if (typeof path !== 'string' || !path) {
    return path;
  }
  if (isAbsoluteUrl(path)) {
    return path;
  }

  const baseUrl = normalizeConfiguredBaseUrl(
    options.baseUrl ?? resolveApiBaseUrl(options.location, options.env),
  );
  if (!baseUrl) {
    return path;
  }

  return new URL(path.replace(/^\/+/, ''), `${baseUrl}/`).toString();
}

function buildReachabilityDetail(requestUrl, error) {
  const originalMessage = error instanceof Error
    ? error.message
    : String(error || 'Unknown network error');
  return `Request to ${requestUrl} failed before the server returned a response. ${originalMessage}. `
    + 'Start the dashboard server on port 4173 or point the UI at the correct API origin.';
}

export class ApiRequestError extends Error {
  constructor(requestUrl, cause) {
    super('Could not reach the dashboard API.');
    this.name = 'ApiRequestError';
    this.userMessage = 'Could not reach the dashboard API.';
    this.detail = buildReachabilityDetail(requestUrl, cause);
    this.retryable = true;
    this.code = 'api_unreachable';
    this.cause = cause;
  }
}

export async function apiFetch(path, init, options = {}) {
  const requestUrl = apiUrl(path, options);
  const method = typeof init?.method === 'string' ? init.method.toUpperCase() : 'GET';
  const requestInit = method === 'GET' || method === 'HEAD'
    ? { ...init, cache: init?.cache || 'no-store' }
    : init;
  try {
    return await fetch(requestUrl, requestInit);
  } catch (error) {
    throw new ApiRequestError(requestUrl, error);
  }
}
