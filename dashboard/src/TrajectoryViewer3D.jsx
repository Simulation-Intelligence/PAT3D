import { useEffect, useMemo, useRef, useState } from 'react';
import { asList, formatObjectDisplayName, safeText } from './runtimeViewModel';

const TRAJECTORY_COLORS = [
  '#5cabff',
  '#46d19a',
  '#f1b157',
  '#ff8b7a',
  '#b494ff',
  '#54c5f8',
  '#f285c1',
  '#8ccf68',
];

function disposeMaterial(material) {
  if (!material) return;
  if (Array.isArray(material)) {
    material.forEach(disposeMaterial);
    return;
  }
  material.map?.dispose?.();
  material.dispose?.();
}

function disposeObject(object) {
  object?.traverse?.((child) => {
    child.geometry?.dispose?.();
    disposeMaterial(child.material);
  });
}

function formatAxisValue(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 'n/a';
  }
  if (Math.abs(numeric) >= 10) {
    return numeric.toFixed(1);
  }
  return numeric.toFixed(3).replace(/\.?0+$/u, '');
}

function formatDisplacement(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 'n/a';
  }
  const sign = numeric > 0 ? '+' : '';
  return `${sign}${formatAxisValue(numeric)}`;
}

function interpolatePoint(points, progress, THREE) {
  if (!points.length) {
    return new THREE.Vector3();
  }
  if (points.length === 1) {
    return points[0].clone();
  }
  const bounded = Math.max(0, Math.min(0.999999, progress));
  const scaled = bounded * (points.length - 1);
  const startIndex = Math.floor(scaled);
  const localT = scaled - startIndex;
  const start = points[startIndex];
  const end = points[Math.min(points.length - 1, startIndex + 1)];
  return start.clone().lerp(end, localT);
}

export default function TrajectoryViewer3D({ trajectory }) {
  const containerRef = useRef(null);
  const [loading, setLoading] = useState(false);
  const [renderError, setRenderError] = useState('');

  const tracks = useMemo(() => (
    asList(trajectory?.tracks)
      .map((track, index) => {
        const points = asList(track?.points)
          .map((point, pointIndex) => ({
            id: `${track?.object_id || `object-${index + 1}`}-${pointIndex}`,
            frameIndex: Number(point?.frame_index ?? pointIndex),
            x: Number(point?.translation_xyz?.[0] ?? 0),
            y: Number(point?.translation_xyz?.[1] ?? 0),
            z: Number(point?.translation_xyz?.[2] ?? 0),
          }))
          .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z));
        if (!points.length) {
          return null;
        }
        const start = points[0];
        const end = points[points.length - 1];
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const dz = end.z - start.z;
        return {
          id: track?.object_id || `object-${index + 1}`,
          label: formatObjectDisplayName(track?.object_id, `object${index + 1}`),
          color: TRAJECTORY_COLORS[index % TRAJECTORY_COLORS.length],
          pointCount: points.length,
          points,
          start,
          end,
          delta: { x: dx, y: dy, z: dz },
          distance: Math.sqrt((dx ** 2) + (dy ** 2) + (dz ** 2)),
        };
      })
      .filter(Boolean)
  ), [trajectory]);

  const bounds = useMemo(() => {
    const points = tracks.flatMap((track) => track.points);
    if (!points.length) {
      return null;
    }
    const xs = points.map((point) => point.x);
    const ys = points.map((point) => point.y);
    const zs = points.map((point) => point.z);
    const min = {
      x: Math.min(...xs),
      y: Math.min(...ys),
      z: Math.min(...zs),
    };
    const max = {
      x: Math.max(...xs),
      y: Math.max(...ys),
      z: Math.max(...zs),
    };
    const center = {
      x: (min.x + max.x) / 2,
      y: (min.y + max.y) / 2,
      z: (min.z + max.z) / 2,
    };
    const size = {
      x: Math.max(1e-4, max.x - min.x),
      y: Math.max(1e-4, max.y - min.y),
      z: Math.max(1e-4, max.z - min.z),
    };
    return {
      min,
      max,
      center,
      size,
      span: Math.max(size.x, size.y, size.z, 0.18),
    };
  }, [tracks]);

  useEffect(() => {
    if (!tracks.length || !bounds || !containerRef.current) {
      return undefined;
    }

    let disposed = false;
    let renderer;
    let controls;
    let animationFrame = 0;
    let mountedRoot = null;
    const cleanupCallbacks = [];

    async function mount() {
      setLoading(true);
      setRenderError('');
      try {
        const [THREE, { OrbitControls }] = await Promise.all([
          import('three'),
          import('three/examples/jsm/controls/OrbitControls.js'),
        ]);
        if (disposed || !containerRef.current) {
          return;
        }

        const scene = new THREE.Scene();
        scene.background = new THREE.Color('#101418');
        scene.fog = new THREE.Fog('#101418', bounds.span * 2.6, bounds.span * 7.4);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.15;
        containerRef.current.innerHTML = '';
        containerRef.current.appendChild(renderer.domElement);

        const camera = new THREE.PerspectiveCamera(42, 1, 0.01, Math.max(40, bounds.span * 20));
        const cameraDistance = Math.max(0.95, bounds.span * 2.3);
        camera.position.set(
          bounds.center.x + (cameraDistance * 0.95),
          bounds.center.y + (cameraDistance * 0.82),
          bounds.center.z + (cameraDistance * 1.18),
        );

        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.065;
        controls.target.set(bounds.center.x, bounds.center.y, bounds.center.z);
        controls.minDistance = Math.max(0.2, bounds.span * 0.55);
        controls.maxDistance = Math.max(3.5, bounds.span * 7.5);
        controls.update();

        scene.add(new THREE.AmbientLight('#f8fbff', 0.95));
        const keyLight = new THREE.DirectionalLight('#e8f0ff', 1.35);
        keyLight.position.set(bounds.center.x + 2.6, bounds.center.y + 4.2, bounds.center.z + 3.8);
        scene.add(keyLight);
        const rimLight = new THREE.DirectionalLight('#ffcba3', 0.48);
        rimLight.position.set(bounds.center.x - 3.5, bounds.center.y + 1.2, bounds.center.z - 2.4);
        scene.add(rimLight);

        const root = new THREE.Group();
        mountedRoot = root;
        scene.add(root);

        const grid = new THREE.GridHelper(bounds.span * 2.8, 12, '#505763', '#1b212a');
        grid.position.set(bounds.center.x, bounds.min.y - Math.max(bounds.span * 0.1, 0.03), bounds.center.z);
        if (Array.isArray(grid.material)) {
          grid.material.forEach((material) => {
            material.opacity = 0.54;
            material.transparent = true;
          });
        } else {
          grid.material.opacity = 0.54;
          grid.material.transparent = true;
        }
        scene.add(grid);

        const axes = new THREE.AxesHelper(Math.max(0.22, bounds.span * 0.45));
        axes.position.set(bounds.min.x - Math.max(0.06, bounds.span * 0.08), grid.position.y, bounds.min.z - Math.max(0.06, bounds.span * 0.08));
        scene.add(axes);

        const tracerEntries = [];
        const markerRadius = Math.max(0.009, bounds.span * 0.035);

        tracks.forEach((track, index) => {
          const pathPoints = track.points.map((point) => new THREE.Vector3(point.x, point.y, point.z));
          const color = new THREE.Color(track.color);

          if (pathPoints.length >= 2) {
            const lineGeometry = new THREE.BufferGeometry().setFromPoints(pathPoints);
            const lineMaterial = new THREE.LineBasicMaterial({
              color,
              transparent: true,
              opacity: 0.94,
            });
            const line = new THREE.Line(lineGeometry, lineMaterial);
            root.add(line);
          }

          const startMaterial = new THREE.MeshStandardMaterial({
            color,
            emissive: color,
            emissiveIntensity: 0.08,
            roughness: 0.48,
            metalness: 0.06,
            transparent: true,
            opacity: 0.6,
          });
          const endMaterial = new THREE.MeshStandardMaterial({
            color,
            emissive: color,
            emissiveIntensity: 0.2,
            roughness: 0.3,
            metalness: 0.06,
          });
          const tracerMaterial = new THREE.MeshStandardMaterial({
            color: '#f5f7fb',
            emissive: color,
            emissiveIntensity: 0.3,
            roughness: 0.16,
            metalness: 0.08,
          });
          const markerGeometry = new THREE.SphereGeometry(markerRadius, 20, 20);
          const startMarker = new THREE.Mesh(markerGeometry, startMaterial);
          startMarker.position.copy(pathPoints[0]);
          startMarker.scale.setScalar(0.78);
          root.add(startMarker);

          const endMarker = new THREE.Mesh(markerGeometry, endMaterial);
          endMarker.position.copy(pathPoints[pathPoints.length - 1]);
          endMarker.scale.setScalar(1.22);
          root.add(endMarker);

          const tracer = new THREE.Mesh(new THREE.SphereGeometry(markerRadius * 0.82, 16, 16), tracerMaterial);
          tracer.position.copy(pathPoints[0]);
          root.add(tracer);
          tracerEntries.push({
            tracer,
            points: pathPoints,
            durationMs: 2600 + (index * 360),
          });
        });

        const setRendererSize = () => {
          if (!containerRef.current || !renderer) return;
          const width = containerRef.current.clientWidth || 720;
          const height = Math.max(360, Math.round(width * 0.6));
          renderer.setSize(width, height, false);
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
        };

        const renderScene = () => {
          renderer.render(scene, camera);
        };

        const animate = (timeMs) => {
          if (disposed || !renderer) {
            return;
          }
          tracerEntries.forEach((entry) => {
            const progress = entry.points.length === 1
              ? 0
              : ((timeMs % entry.durationMs) / entry.durationMs);
            const nextPoint = interpolatePoint(entry.points, progress, THREE);
            entry.tracer.position.copy(nextPoint);
          });
          controls.update();
          renderScene();
          animationFrame = window.requestAnimationFrame(animate);
        };

        setRendererSize();
        animationFrame = window.requestAnimationFrame(animate);

        const handleResize = () => {
          setRendererSize();
          renderScene();
        };
        window.addEventListener('resize', handleResize);
        cleanupCallbacks.push(() => window.removeEventListener('resize', handleResize));
      } catch (error) {
        if (!disposed) {
          setRenderError(error?.message || '3D trajectory preview failed to initialize.');
        }
      } finally {
        if (!disposed) {
          setLoading(false);
        }
      }
    }

    mount();

    return () => {
      disposed = true;
      cleanupCallbacks.forEach((callback) => callback());
      if (animationFrame) {
        window.cancelAnimationFrame(animationFrame);
      }
      controls?.dispose?.();
      disposeObject(mountedRoot);
      renderer?.dispose?.();
    };
  }, [bounds, tracks]);

  if (!trajectory?.available || !tracks.length) {
    return <div className="empty">No simulation trajectory snapshots were recorded for this scene.</div>;
  }

  const snapshotLabel = trajectory?.snapshot_source === 'forward_simulation_frames'
    ? 'forward simulation frame'
    : 'optimization snapshot';

  return (
    <div className="trajectory-shell">
      <div className="trajectory-summary-bar">
        <div className="trajectory-summary-chip">
          <span className="meta-label">Trajectory source</span>
          <strong>{snapshotLabel}</strong>
        </div>
        <div className="trajectory-summary-chip">
          <span className="meta-label">Snapshots</span>
          <strong>{safeText(trajectory?.snapshot_count, '0')}</strong>
        </div>
        <div className="trajectory-summary-chip">
          <span className="meta-label">Tracked objects</span>
          <strong>{tracks.length}</strong>
        </div>
      </div>
      <div className="trajectory-review-grid">
        <div className="trajectory-viewport-card">
          <div className="trajectory-viewport" role="img" aria-label="Simulation trajectory 3D view">
            <div className="trajectory-canvas" ref={containerRef} />
            <div className="trajectory-hud">
              <span className="trajectory-hud-pill">3D motion paths</span>
              <span className="trajectory-hud-pill">Soft markers: start</span>
              <span className="trajectory-hud-pill">Bright markers: final</span>
              <span className="trajectory-hud-pill">Animated tracer: motion direction</span>
            </div>
            {loading ? <div className="trajectory-overlay">Loading 3D trajectory preview…</div> : null}
            {renderError ? (
              <div className="trajectory-overlay trajectory-overlay-warning">
                {`3D preview unavailable: ${renderError}`}
              </div>
            ) : null}
          </div>
          <div className="note-box">
            Orbit to inspect X, Y, and Z displacement. The grid anchors world space while the animated tracer replays each object path.
          </div>
        </div>
        <aside className="trajectory-legend-card">
          <div className="trajectory-legend-head">
            <div>
              <span className="section-kicker">Trajectory legend</span>
              <h4 className="card-title">Object motion in 3D</h4>
            </div>
            <div className="meta-label">{tracks.length} objects</div>
          </div>
          <div className="trajectory-legend-list" role="list" aria-label="Trajectory legend">
            {tracks.map((track) => (
              <article
                key={track.id}
                className="trajectory-track-card"
                role="listitem"
                style={{ '--track-accent': track.color }}
              >
                <div className="trajectory-track-head">
                  <div className="trajectory-track-label">
                    <span className="trajectory-track-swatch" aria-hidden="true" />
                    <strong>{track.label}</strong>
                  </div>
                  <span className="meta-label">{track.pointCount} frames</span>
                </div>
                <div className="trajectory-track-deltas">
                  <span>{`Δx ${formatDisplacement(track.delta.x)}`}</span>
                  <span>{`Δy ${formatDisplacement(track.delta.y)}`}</span>
                  <span>{`Δz ${formatDisplacement(track.delta.z)}`}</span>
                  <span>{`distance ${formatAxisValue(track.distance)}`}</span>
                </div>
                <div className="trajectory-track-range">
                  <span>{`start (${formatAxisValue(track.start.x)}, ${formatAxisValue(track.start.y)}, ${formatAxisValue(track.start.z)})`}</span>
                  <span>{`end (${formatAxisValue(track.end.x)}, ${formatAxisValue(track.end.y)}, ${formatAxisValue(track.end.z)})`}</span>
                </div>
              </article>
            ))}
          </div>
        </aside>
      </div>
      <div className="note-box">
        Loaded {trajectory.snapshot_count} {snapshotLabel}{trajectory.snapshot_count === 1 ? '' : 's'} from {safeText(trajectory.param_dir)}.
      </div>
    </div>
  );
}
