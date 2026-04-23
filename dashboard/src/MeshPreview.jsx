import { useEffect, useRef, useState } from 'react';
import { artifactUrl } from './runtimeViewModel';

function disposeSceneObject(object) {
  object?.traverse((child) => {
    if (!child.isMesh) return;
    child.geometry?.dispose?.();
    if (Array.isArray(child.material)) {
      child.material.forEach((material) => {
        material.map?.dispose?.();
        material.dispose?.();
      });
      return;
    }
    child.material?.map?.dispose?.();
    child.material?.dispose?.();
  });
}

function prepareColorTexture(texture, THREE, renderer) {
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.max(1, renderer.capabilities.getMaxAnisotropy?.() || 1);
  texture.needsUpdate = true;
}

export default function MeshPreview({ objPath, texturePath, previewImagePath }) {
  const containerRef = useRef(null);
  const [error, setError] = useState('');
  const [warning, setWarning] = useState('');
  const [loading, setLoading] = useState(false);
  const [showImageFallback, setShowImageFallback] = useState(false);
  const meshPathLower = objPath ? objPath.toLowerCase() : '';
  const meshFormatIsGltf =
    meshPathLower.endsWith('.gltf') ||
    meshPathLower.endsWith('.glb');

  useEffect(() => {
    let disposed = false;
    let renderer;
    let controls;
    let resizeObserver;
    let mountedObject = null;
    const abortController = new AbortController();

    async function mount() {
      if (!objPath || !containerRef.current) {
        return;
      }

      setLoading(true);
      setError('');
      setWarning('');
      setShowImageFallback(false);

      try {
        const [THREE, { OBJLoader }, { GLTFLoader }, { DRACOLoader }, { OrbitControls }] = await Promise.all([
          import('three'),
          import('three/examples/jsm/loaders/OBJLoader.js'),
          import('three/examples/jsm/loaders/GLTFLoader.js'),
          import('three/examples/jsm/loaders/DRACOLoader.js'),
          import('three/examples/jsm/controls/OrbitControls.js'),
        ]);

        if (disposed || abortController.signal.aborted || !containerRef.current) {
          return;
        }

        const scene = new THREE.Scene();
        scene.background = new THREE.Color('#282a36');

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        containerRef.current.innerHTML = '';
        containerRef.current.appendChild(renderer.domElement);

        const setRendererSize = () => {
          if (!containerRef.current || !renderer) return;
          const width = containerRef.current.clientWidth || 320;
          const height = 250;
          renderer.setSize(width, height);
          if (camera) {
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
          }
        };

        const camera = new THREE.PerspectiveCamera(42, 320 / 250, 0.1, 100);
        camera.position.set(1.8, 1.8, 2.2);

        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = false;
        controls.enablePan = false;

        scene.add(new THREE.AmbientLight('#ffffff', 1.35));
        const keyLight = new THREE.DirectionalLight('#fff2dc', 1.4);
        keyLight.position.set(2.5, 4, 3.5);
        scene.add(keyLight);

        const meshUrl = artifactUrl(objPath);
        let object = null;
        if (meshFormatIsGltf) {
          const dracoLoader = new DRACOLoader();
          dracoLoader.setDecoderPath('https://www.gstatic.com/draco/v1/decoders/');
          const gltfLoader = new GLTFLoader();
          gltfLoader.setDRACOLoader(dracoLoader);
          try {
            object = await new Promise((resolve, reject) => {
              gltfLoader.load(
                meshUrl,
                (gltf) => resolve(gltf.scene),
                undefined,
                (gltfError) => reject(gltfError),
              );
            });
          } finally {
            dracoLoader?.dispose();
          }
        } else {
          const response = await fetch(meshUrl, { signal: abortController.signal });
          if (!response.ok) {
            throw new Error(`OBJ fetch failed with HTTP ${response.status}`);
          }
          const objSource = await response.text();
          if (!objSource || !objSource.trim()) {
            throw new Error('OBJ payload is empty.');
          }
          const objLoader = new OBJLoader();
          object = objLoader.parse(objSource);
        }
        mountedObject = object;

        let texture = null;
        if (texturePath && !meshFormatIsGltf) {
          try {
            const loader = new THREE.TextureLoader();
            texture = await loader.loadAsync(artifactUrl(texturePath));
            prepareColorTexture(texture, THREE, renderer);
          } catch (textureError) {
            setWarning(`Texture unavailable: ${textureError.message}`);
          }
        }

        if (!meshFormatIsGltf) {
          object.traverse((child) => {
            if (!child.isMesh) return;
            child.material = new THREE.MeshStandardMaterial({
              color: texture ? '#ffffff' : '#44475a',
              map: texture,
              roughness: 0.88,
              metalness: 0.04,
            });
          });
        }

        const bounds = new THREE.Box3().setFromObject(object);
        if (bounds.isEmpty()) {
          throw new Error('Mesh contains no renderable geometry.');
        }

        const center = bounds.getCenter(new THREE.Vector3());
        const size = bounds.getSize(new THREE.Vector3());
        object.position.sub(center);
        const scale = 1.9 / Math.max(size.x, size.y, size.z, 0.001);
        object.scale.setScalar(scale);
        scene.add(object);

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
      } catch (mountError) {
        if (!abortController.signal.aborted) {
          setError(mountError.message);
          if (previewImagePath) {
            setShowImageFallback(true);
          }
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
      disposeSceneObject(mountedObject);
      renderer?.dispose();
    };
  }, [objPath, texturePath, previewImagePath]);

  if (!objPath) {
    return (
      <div className="mesh-preview-shell">
        <div className="mesh-preview-toolbar">
          <span className="meta-label">Mesh preview</span>
        </div>
        <div className="mesh-fallback">
          {error
            ? `Mesh preview failed: ${error}. Showing preview image instead.`
            : previewImagePath
              ? 'Mesh artifact not available. Showing preview image instead.'
              : 'No mesh artifact recorded for this object.'
          }
        </div>
        {previewImagePath ? (
          <div className="media-frame compact-frame">
            <img src={artifactUrl(previewImagePath)} alt="Mesh preview fallback" loading="lazy" />
          </div>
        ) : null}
      </div>
    );
  }

  if (showImageFallback && previewImagePath) {
    return (
      <div className="mesh-preview-shell">
        <div className="mesh-preview-toolbar">
          <span className="meta-label">Mesh preview fallback</span>
        </div>
        <div className="note-box warning-box">{`Mesh preview failed: ${error}`}</div>
        <div className="media-frame compact-frame">
          <img src={artifactUrl(previewImagePath)} alt="Mesh preview fallback" loading="lazy" />
        </div>
      </div>
    );
  }

  return (
    <div className="mesh-preview-shell">
      <div className="mesh-preview-toolbar">
        <span className="meta-label">{loading ? 'Loading mesh...' : 'Mesh preview'}</span>
      </div>
      <div className="mesh-frame" ref={containerRef} />
      {warning ? <div className="note-box warning-box">{warning}</div> : null}
      {error ? <div className="note-box warning-box">{error}</div> : null}
    </div>
  );
}
