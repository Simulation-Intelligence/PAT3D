import { useEffect, useMemo, useRef, useState } from 'react';
import { ApiRequestError, apiFetch } from './api';
import { artifactUrl } from './runtimeViewModel';

export function meshFormatForPath(path) {
  const normalized = String(path || '').split('?')[0].toLowerCase();
  if (normalized.endsWith('.obj')) return 'obj';
  if (normalized.endsWith('.glb') || normalized.endsWith('.gltf')) return 'gltf';
  return 'unknown';
}

function bundleIdentity(bundle, version) {
  if (!bundle) return 'none';
  if (bundle.kind === 'artifact') {
    return `artifact:${bundle.path || ''}:${version}`;
  }
  const sceneId = bundle.data?.scene_id || 'scene';
  const objectKey = (bundle.data?.objects || [])
    .map((object) => (
      `${object.object_id || 'object'}:`
      + `${object.mesh_obj_path || ''}:`
      + `${object.mesh_mtl_path || ''}:`
      + `${object.texture_image_path || ''}:`
      + `${JSON.stringify(object.transform || null)}`
    ))
    .join('|');
  return `inline:${sceneId}:${objectKey}:${version}`;
}

function disposeMaterial(material) {
  if (!material) return;
  material.map?.dispose?.();
  material.dispose?.();
}

function disposeSceneObject(object) {
  object.traverse((child) => {
    child.geometry?.dispose?.();
    if (Array.isArray(child.material)) {
      child.material.forEach(disposeMaterial);
      return;
    }
    disposeMaterial(child.material);
  });
}

async function fetchJson(url, signal) {
  const response = await apiFetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Scene bundle fetch failed with HTTP ${response.status}`);
  }
  return response.json();
}

async function fetchText(url, signal) {
  const response = await apiFetch(url, { signal });
  if (!response.ok) {
    throw new Error(`OBJ fetch failed with HTTP ${response.status}`);
  }
  return response.text();
}

export function isArtifactApiUnreachableError(error) {
  return error instanceof ApiRequestError || error?.code === 'api_unreachable';
}

export function formatArtifactFetchError(error) {
  if (error && typeof error === 'object') {
    if (typeof error.detail === 'string' && error.detail.trim()) {
      return error.detail.trim();
    }
    if (typeof error.userMessage === 'string' && error.userMessage.trim()) {
      return error.userMessage.trim();
    }
    if (typeof error.message === 'string' && error.message.trim()) {
      return error.message.trim();
    }
  }
  return String(error || 'Unknown artifact load error');
}

function artifactDirectoryPath(path) {
  const normalized = String(path || '').replace(/\\/g, '/');
  const slashIndex = normalized.lastIndexOf('/');
  if (slashIndex < 0) {
    return '';
  }
  return normalized.slice(0, slashIndex + 1);
}

function isAbsoluteTextureReference(value) {
  return /^(?:[a-z]+:|\/\/)/iu.test(String(value || ''));
}

export function rewriteMtlTexturePaths(mtlSource, mtlPath) {
  const basePath = artifactDirectoryPath(mtlPath);
  if (!basePath || typeof mtlSource !== 'string' || !mtlSource.trim()) {
    return mtlSource;
  }

  return mtlSource
    .split(/\r?\n/u)
    .map((line) => {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) {
        return line;
      }
      const mapDirective = trimmed.match(/^(map_[A-Za-z0-9]+|bump|disp|decal|norm|refl)\s+(.+)$/u);
      if (!mapDirective) {
        return line;
      }

      const [, directive, remainder] = mapDirective;
      const parts = remainder.trim().split(/\s+/u);
      if (!parts.length) {
        return line;
      }
      const textureRef = parts.at(-1);
      if (!textureRef || isAbsoluteTextureReference(textureRef) || textureRef.startsWith('-')) {
        return line;
      }
      parts[parts.length - 1] = artifactUrl(`${basePath}${textureRef}`);
      return `${directive} ${parts.join(' ')}`;
    })
    .join('\n');
}

function meshFormatIsGltf(path) {
  const lower = String(path || '').toLowerCase();
  return lower.endsWith('.glb') || lower.endsWith('.gltf');
}

function prepareColorTexture(texture, THREE, renderer) {
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.max(1, renderer.capabilities.getMaxAnisotropy?.() || 1);
  texture.needsUpdate = true;
}

function prepareMaterialMaps(material, THREE, renderer) {
  if (!material) return;
  if (Array.isArray(material)) {
    material.forEach((entry) => prepareMaterialMaps(entry, THREE, renderer));
    return;
  }
  if (material.map) {
    prepareColorTexture(material.map, THREE, renderer);
  }
}

export function prepareMeshForDisplay(mesh, THREE, renderer) {
  if (!mesh?.isMesh) return;
  if (mesh.geometry?.computeVertexNormals) {
    mesh.geometry.computeVertexNormals();
  }
  if (mesh.geometry?.computeBoundingSphere) {
    mesh.geometry.computeBoundingSphere();
  }
  if (Array.isArray(mesh.material)) {
    mesh.material.forEach((material) => {
      if (!material) return;
      material.side = THREE.DoubleSide;
      material.shadowSide = THREE.DoubleSide;
      material.needsUpdate = true;
    });
  } else if (mesh.material) {
    mesh.material.side = THREE.DoubleSide;
    mesh.material.shadowSide = THREE.DoubleSide;
    mesh.material.needsUpdate = true;
  }
  mesh.castShadow = true;
  mesh.receiveShadow = true;
}

function parseGroundY(value) {
  const numeric = Number.parseFloat(String(value ?? ''));
  return Number.isFinite(numeric) ? numeric : null;
}

function bundleGroundInfo(sceneBundle) {
  if (!sceneBundle || typeof sceneBundle !== 'object') {
    return {
      requestedGroundY: null,
      appliedGroundY: null,
      source: null,
    };
  }
  return {
    requestedGroundY: parseGroundY(sceneBundle.requested_ground_plane_y),
    appliedGroundY: parseGroundY(sceneBundle.applied_ground_plane_y),
    source: typeof sceneBundle.ground_plane_source === 'string' ? sceneBundle.ground_plane_source : null,
  };
}

function formatGroundY(value) {
  if (!Number.isFinite(value)) {
    return 'n/a';
  }
  return value.toFixed(3).replace(/\.?0+$/, '');
}

export function sceneFloorY(bounds, explicitGroundY = null) {
  if (Number.isFinite(explicitGroundY)) {
    return explicitGroundY;
  }
  if (!bounds || typeof bounds.isEmpty !== 'function' || bounds.isEmpty()) {
    return -0.001;
  }
  return bounds.min.y;
}

export default function SceneViewer({ bundle, version = '' }) {
  const containerRef = useRef(null);
  const [error, setError] = useState('');
  const [warnings, setWarnings] = useState([]);
  const [loading, setLoading] = useState(false);
  const [groundInfo, setGroundInfo] = useState({
    requestedGroundY: null,
    appliedGroundY: null,
    source: null,
  });
  const bundleKey = useMemo(() => bundleIdentity(bundle, version), [bundle, version]);

  useEffect(() => {
    let disposed = false;
    let renderer;
    let controls;
    let resizeObserver;
    const abortController = new AbortController();
    const mountedObjects = [];

    async function loadBundle() {
      if (!bundle) return null;
      if (bundle.kind === 'inline') return bundle.data;
      if (bundle.kind === 'artifact' && bundle.path) {
        return fetchJson(artifactUrl(bundle.path), abortController.signal);
      }
      return null;
    }

    async function mount() {
      if (!containerRef.current) return;
      setLoading(true);
      setWarnings([]);
      try {
        const sceneBundle = await loadBundle();
        if (abortController.signal.aborted || disposed || !containerRef.current) return;
        const sceneGroundInfo = bundleGroundInfo(sceneBundle);
        setGroundInfo(sceneGroundInfo);
        if (!sceneBundle || !sceneBundle.objects?.length) {
          setError('No scene bundle data is available for the final 3D scene.');
          return;
        }

        const [THREE, { OBJLoader }, { MTLLoader }, { GLTFLoader }, { DRACOLoader }, { OrbitControls }] = await Promise.all([
          import('three'),
          import('three/examples/jsm/loaders/OBJLoader.js'),
          import('three/examples/jsm/loaders/MTLLoader.js'),
          import('three/examples/jsm/loaders/GLTFLoader.js'),
          import('three/examples/jsm/loaders/DRACOLoader.js'),
          import('three/examples/jsm/controls/OrbitControls.js'),
        ]);
        if (abortController.signal.aborted || disposed || !containerRef.current) return;

        const scene = new THREE.Scene();
        scene.background = new THREE.Color('#2c3040');

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.18;
        containerRef.current.innerHTML = '';
        containerRef.current.appendChild(renderer.domElement);

        const setRendererSize = () => {
          if (!containerRef.current || !renderer) return;
          const width = containerRef.current.clientWidth || 640;
          const height = Math.max(420, Math.round(width * 0.62));
          renderer.setSize(width, height);
          if (camera) {
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
          }
        };

        const camera = new THREE.PerspectiveCamera(45, 640 / 420, 0.1, 500);
        camera.up.set(0, 1, 0);
        camera.position.set(6, 6, -9);

        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = false;
        controls.target.set(0, 0.8, 0);

        scene.add(new THREE.HemisphereLight('#f4f7ff', '#5d6878', 1.45));

        const keyLight = new THREE.DirectionalLight('#fff1d5', 2.35);
        keyLight.position.set(9, 14, 11);
        keyLight.castShadow = true;
        scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight('#9edbff', 1.1);
        fillLight.position.set(-10, 7, -6);
        scene.add(fillLight);

        const rimLight = new THREE.DirectionalLight('#d6c9ff', 0.72);
        rimLight.position.set(6, 4, -10);
        scene.add(rimLight);

        const grid = new THREE.GridHelper(12, 24, 0x44475a, 0x6272a4);
        grid.position.y = -0.001;
        scene.add(grid);
        const floor = new THREE.Mesh(
          new THREE.PlaneGeometry(1, 1),
          new THREE.MeshStandardMaterial({
            color: '#d7cfbf',
            roughness: 0.96,
            metalness: 0.0,
            side: THREE.DoubleSide,
            polygonOffset: true,
            polygonOffsetFactor: 1,
            polygonOffsetUnits: 1,
          }),
        );
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        scene.add(floor);
        mountedObjects.push(floor);

        const objLoader = new OBJLoader();
        const mtlLoader = new MTLLoader();
        const dracoLoader = new DRACOLoader();
        dracoLoader.setDecoderPath('https://www.gstatic.com/draco/v1/decoders/');
        const gltfLoader = new GLTFLoader();
        gltfLoader.setDRACOLoader(dracoLoader);
        const textureLoader = new THREE.TextureLoader();
        const bounds = new THREE.Box3();
        const nextWarnings = [];

        try {
          for (const object of sceneBundle.objects) {
            if (abortController.signal.aborted) return;
            if (!object?.mesh_obj_path) continue;
            try {
              const meshUrl = artifactUrl(object.mesh_obj_path);
              const meshFormat = meshFormatForPath(object.mesh_obj_path);
              let group;
              if (meshFormat === 'obj') {
                let loadedMaterials = null;
                if (object.mesh_mtl_path) {
                  try {
                    const mtlSource = await fetchText(artifactUrl(object.mesh_mtl_path), abortController.signal);
                    loadedMaterials = mtlLoader.parse(
                      rewriteMtlTexturePaths(mtlSource, object.mesh_mtl_path),
                      '',
                    );
                    loadedMaterials.preload();
                    objLoader.setMaterials(loadedMaterials);
                  } catch (mtlError) {
                    if (isArtifactApiUnreachableError(mtlError)) {
                      throw mtlError;
                    }
                    loadedMaterials = null;
                    objLoader.setMaterials(null);
                    nextWarnings.push(`Material file unavailable for ${object.object_id || 'object'}: ${formatArtifactFetchError(mtlError)}`);
                  }
                } else {
                  objLoader.setMaterials(null);
                }

                const objSource = await fetchText(meshUrl, abortController.signal);
                group = objLoader.parse(objSource);
                let texture = null;
                if (!loadedMaterials && object.texture_image_path) {
                  try {
                    texture = await textureLoader.loadAsync(artifactUrl(object.texture_image_path));
                    prepareColorTexture(texture, THREE, renderer);
                  } catch (textureError) {
                    if (isArtifactApiUnreachableError(textureError)) {
                      throw textureError;
                    }
                    nextWarnings.push(`Texture unavailable for ${object.object_id || 'object'}: ${formatArtifactFetchError(textureError)}`);
                  }
                }

                group.traverse((child) => {
                  if (!child.isMesh) return;
                  if (loadedMaterials) {
                    prepareMaterialMaps(child.material, THREE, renderer);
                  } else {
                    child.material = new THREE.MeshStandardMaterial({
                      color: texture ? '#ffffff' : '#f3dcc4',
                      map: texture,
                      roughness: 0.85,
                      metalness: 0.04,
                    });
                  }
                  prepareMeshForDisplay(child, THREE, renderer);
                });
              } else if (meshFormat === 'gltf') {
                const gltf = await gltfLoader.loadAsync(meshUrl);
                group = gltf.scene || gltf.scenes?.[0];
                if (!group) {
                  throw new Error('GLTF scene is empty');
                }
                group.traverse((child) => {
                  if (!child.isMesh) return;
                  prepareMaterialMaps(child.material, THREE, renderer);
                  prepareMeshForDisplay(child, THREE, renderer);
                });
              } else {
                throw new Error(`Unsupported mesh format: ${object.mesh_obj_path}`);
              }

              if (!object.already_transformed && object.transform) {
                const translation = object.transform.translation_xyz || [0, 0, 0];
                group.position.set(translation[0] || 0, translation[1] || 0, translation[2] || 0);
                const scale = object.transform.scale_xyz || [1, 1, 1];
                group.scale.set(scale[0] || 1, scale[1] || 1, scale[2] || 1);
                if (object.transform.rotation_type === 'quaternion') {
                  const [w = 1, x = 0, y = 0, z = 0] = object.transform.rotation_value || [];
                  group.quaternion.set(x, y, z, w);
                }
              }

              scene.add(group);
              mountedObjects.push(group);
              bounds.expandByObject(group);
            } catch (objectError) {
              if (isArtifactApiUnreachableError(objectError)) {
                throw objectError;
              }
              nextWarnings.push(`Could not load ${object.object_id || 'object'}: ${formatArtifactFetchError(objectError)}`);
            }
          }
        } finally {
          dracoLoader.dispose();
        }

        if (!mountedObjects.length) {
          setWarnings(nextWarnings);
          setError('No mesh objects could be loaded for the final 3D scene.');
          return;
        }

        if (!bounds.isEmpty()) {
          const center = bounds.getCenter(new THREE.Vector3());
          const size = bounds.getSize(new THREE.Vector3());
          const radius = Math.max(size.x, size.y, size.z, 1);
          const floorY = sceneFloorY(bounds, sceneGroundInfo.appliedGroundY);
          const floorExtent = Math.max(size.x, size.z, 1) * 2.4;
          grid.position.y = floorY;
          grid.scale.setScalar(floorExtent / 12);
          floor.position.set(center.x, floorY, center.z);
          floor.scale.set(floorExtent, floorExtent, 1);
          controls.target.copy(center);
          camera.position.set(center.x + radius * 1.2, center.y + radius * 0.95, center.z - radius * 1.65);
          camera.near = 0.01;
          camera.far = radius * 20;
          camera.updateProjectionMatrix();
        }

        resizeObserver = new ResizeObserver(() => setRendererSize());
        const renderScene = () => {
          if (disposed || abortController.signal.aborted || !renderer) return;
          renderer.render(scene, camera);
        };

        controls.addEventListener('change', renderScene);
        resizeObserver = new ResizeObserver(() => {
          setRendererSize();
          renderScene();
        });
        resizeObserver.observe(containerRef.current);
        setRendererSize();
        renderScene();
        setWarnings(nextWarnings);
        setError('');
      } catch (mountError) {
        if (!abortController.signal.aborted) {
          setGroundInfo({
            requestedGroundY: null,
            appliedGroundY: null,
            source: null,
          });
          setError(formatArtifactFetchError(mountError));
        }
      } finally {
        if (!abortController.signal.aborted) {
          setLoading(false);
        }
      }
    }

    mount();

    return () => {
      disposed = true;
      abortController.abort();
      resizeObserver?.disconnect();
      controls?.dispose();
      mountedObjects.forEach(disposeSceneObject);
      renderer?.dispose();
    };
  }, [bundleKey]);

  if (!bundle) {
    return <div className="empty">No scene bundle is available for the final-stage viewer.</div>;
  }

  return (
    <div className="scene-viewer-shell">
      <div className="scene-viewer-toolbar">
        <span className="meta-label">{loading ? 'Loading scene…' : 'Interactive scene viewer'}</span>
        {groundInfo.appliedGroundY !== null ? (
          <span className="meta-label">
            Ground Y {formatGroundY(groundInfo.appliedGroundY)}
            {groundInfo.requestedGroundY !== null && groundInfo.requestedGroundY !== groundInfo.appliedGroundY
              ? ` (requested ${formatGroundY(groundInfo.requestedGroundY)})`
              : ''}
          </span>
        ) : null}
      </div>
      <div className="scene-viewer-frame" ref={containerRef} />
      {error ? <div className="note-box warning-box">{error}</div> : null}
      {warnings.length ? (
        <div className="note-box warning-box">
          {warnings.map((warning) => (
            <div key={warning}>{warning}</div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
