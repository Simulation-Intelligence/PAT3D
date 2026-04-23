import { useEffect, useMemo, useRef, useState } from 'react';
import { artifactUrl } from './runtimeViewModel';

const COLOR_PALETTE = [
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

function nextColor(index) {
  return COLOR_PALETTE[index % COLOR_PALETTE.length];
}

function slugifyLabel(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'object';
}

function createLayer(label, index, overrides = {}) {
  const normalizedLabel = String(label || '').trim() || `object ${index + 1}`;
  return {
    instanceId: overrides.instanceId || `${slugifyLabel(normalizedLabel)}-${index + 1}`,
    label: normalizedLabel,
    color: overrides.color || nextColor(index),
    maskPath: overrides.maskPath || null,
  };
}

function createCanvas(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load image: ${url}`));
    image.src = url;
  });
}

function canvasHasPaint(canvas) {
  const context = canvas.getContext('2d', { willReadFrequently: true });
  const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
  for (let index = 3; index < data.length; index += 4) {
    if (data[index] > 0) {
      return true;
    }
  }
  return false;
}

export default function SegmentationEditor({
  referencePath,
  versionToken,
  initialInstances = [],
  suggestedLabels = [],
  busy = false,
  onSubmit,
}) {
  const displayCanvasRef = useRef(null);
  const imageRef = useRef(null);
  const pointerStateRef = useRef(null);
  const layerCanvasesRef = useRef(new Map());
  const tintCanvasRef = useRef(null);

  const [draftLabel, setDraftLabel] = useState('');
  const [layers, setLayers] = useState([]);
  const [activeLayerId, setActiveLayerId] = useState(null);
  const [brushSize, setBrushSize] = useState(20);
  const [tool, setTool] = useState('paint');
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  const [loading, setLoading] = useState(false);

  const suggestions = useMemo(
    () => [...new Set((suggestedLabels || []).map((label) => String(label).trim()).filter(Boolean))],
    [suggestedLabels],
  );

  async function redrawComposite(nextLayers = layers, nextActiveLayerId = activeLayerId) {
    const displayCanvas = displayCanvasRef.current;
    const baseImage = imageRef.current;
    if (!displayCanvas || !baseImage || !canvasSize.width || !canvasSize.height) {
      return;
    }

    if (displayCanvas.width !== canvasSize.width || displayCanvas.height !== canvasSize.height) {
      displayCanvas.width = canvasSize.width;
      displayCanvas.height = canvasSize.height;
    }
    if (!tintCanvasRef.current
      || tintCanvasRef.current.width !== canvasSize.width
      || tintCanvasRef.current.height !== canvasSize.height) {
      tintCanvasRef.current = createCanvas(canvasSize.width, canvasSize.height);
    }

    const displayContext = displayCanvas.getContext('2d');
    displayContext.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
    displayContext.drawImage(baseImage, 0, 0, displayCanvas.width, displayCanvas.height);

    const tintCanvas = tintCanvasRef.current;
    const tintContext = tintCanvas.getContext('2d');

    nextLayers.forEach((layer, index) => {
      const maskCanvas = layerCanvasesRef.current.get(layer.instanceId);
      if (!maskCanvas) return;

      tintContext.clearRect(0, 0, tintCanvas.width, tintCanvas.height);
      tintContext.drawImage(maskCanvas, 0, 0);
      tintContext.globalCompositeOperation = 'source-in';
      tintContext.fillStyle = layer.color || nextColor(index);
      tintContext.fillRect(0, 0, tintCanvas.width, tintCanvas.height);
      tintContext.globalCompositeOperation = 'source-over';

      displayContext.save();
      displayContext.globalAlpha = layer.instanceId === nextActiveLayerId ? 0.58 : 0.42;
      displayContext.drawImage(tintCanvas, 0, 0);
      displayContext.restore();
    });
  }

  useEffect(() => {
    if (!referencePath) {
      imageRef.current = null;
      layerCanvasesRef.current = new Map();
      setLayers([]);
      setActiveLayerId(null);
      setCanvasSize({ width: 0, height: 0 });
      return undefined;
    }

    let cancelled = false;
    setLoading(true);

    async function initialize() {
      const baseImage = await loadImage(artifactUrl(referencePath));
      if (cancelled) return;
      imageRef.current = baseImage;

      const width = baseImage.naturalWidth || baseImage.width;
      const height = baseImage.naturalHeight || baseImage.height;
      const seededLayers = initialInstances.length
        ? initialInstances.map((instance, index) => createLayer(instance.label, index, {
            instanceId: instance.instance_id,
            color: instance.color,
            maskPath: instance.mask_path,
          }))
        : [];

      const nextLayerCanvases = new Map();
      for (let index = 0; index < seededLayers.length; index += 1) {
        const layer = seededLayers[index];
        const maskCanvas = createCanvas(width, height);
        if (layer.maskPath) {
          try {
            const maskImage = await loadImage(artifactUrl(layer.maskPath));
            if (cancelled) return;
            maskCanvas.getContext('2d').drawImage(maskImage, 0, 0, width, height);
          } catch {
            // Ignore missing persisted masks and keep the layer empty.
          }
        }
        nextLayerCanvases.set(layer.instanceId, maskCanvas);
      }

      if (cancelled) return;
      layerCanvasesRef.current = nextLayerCanvases;
      tintCanvasRef.current = createCanvas(width, height);
      setCanvasSize({ width, height });
      setLayers(seededLayers);
      setActiveLayerId((current) => (
        current && seededLayers.some((layer) => layer.instanceId === current)
          ? current
          : seededLayers[0]?.instanceId || null
      ));
      setLoading(false);
    }

    initialize().catch(() => {
      if (!cancelled) {
        setLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [referencePath, versionToken]);

  useEffect(() => {
    redrawComposite();
  }, [layers, activeLayerId, canvasSize]);

  function ensureLayerCanvas(layerId) {
    const existing = layerCanvasesRef.current.get(layerId);
    if (existing) return existing;
    if (!canvasSize.width || !canvasSize.height) return null;
    const canvas = createCanvas(canvasSize.width, canvasSize.height);
    layerCanvasesRef.current.set(layerId, canvas);
    return canvas;
  }

  function addLayer(labelText) {
    const label = String(labelText || draftLabel).trim();
    if (!label) return;
    const index = layers.length;
    const nextLayer = createLayer(label, index);
    while (layers.some((layer) => layer.instanceId === nextLayer.instanceId)) {
      nextLayer.instanceId = `${nextLayer.instanceId}-${index + 1}`;
    }
    ensureLayerCanvas(nextLayer.instanceId);
    const nextLayers = [...layers, nextLayer];
    setLayers(nextLayers);
    setActiveLayerId(nextLayer.instanceId);
    setDraftLabel('');
    redrawComposite(nextLayers, nextLayer.instanceId);
  }

  function updateLayer(nextLayerId, updater) {
    const nextLayers = layers.map((layer) => (
      layer.instanceId === nextLayerId ? { ...layer, ...updater(layer) } : layer
    ));
    setLayers(nextLayers);
    redrawComposite(nextLayers, activeLayerId);
  }

  function removeLayer(layerId) {
    layerCanvasesRef.current.delete(layerId);
    const nextLayers = layers.filter((layer) => layer.instanceId !== layerId);
    const fallbackLayerId = nextLayers[0]?.instanceId || null;
    setLayers(nextLayers);
    setActiveLayerId((current) => (current === layerId ? fallbackLayerId : current));
    redrawComposite(nextLayers, activeLayerId === layerId ? fallbackLayerId : activeLayerId);
  }

  function clearLayer(layerId) {
    const canvas = ensureLayerCanvas(layerId);
    if (!canvas) return;
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    redrawComposite();
  }

  function imagePointFromEvent(event) {
    const canvas = displayCanvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const x = ((event.clientX - rect.left) * canvas.width) / rect.width;
    const y = ((event.clientY - rect.top) * canvas.height) / rect.height;
    return { x, y };
  }

  function drawStroke(layerId, fromPoint, toPoint) {
    const canvas = ensureLayerCanvas(layerId);
    if (!canvas) return;
    const context = canvas.getContext('2d');
    context.save();
    context.globalCompositeOperation = tool === 'erase' ? 'destination-out' : 'source-over';
    context.strokeStyle = 'rgba(255,255,255,1)';
    context.fillStyle = 'rgba(255,255,255,1)';
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.lineWidth = brushSize;
    context.beginPath();
    context.moveTo(fromPoint.x, fromPoint.y);
    context.lineTo(toPoint.x, toPoint.y);
    context.stroke();
    context.beginPath();
    context.arc(toPoint.x, toPoint.y, brushSize / 2, 0, Math.PI * 2);
    context.fill();
    context.restore();
  }

  function handlePointerDown(event) {
    if (busy || !activeLayerId) return;
    const point = imagePointFromEvent(event);
    if (!point) return;
    pointerStateRef.current = { pointerId: event.pointerId, lastPoint: point };
    drawStroke(activeLayerId, point, point);
    redrawComposite();
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handlePointerMove(event) {
    const pointerState = pointerStateRef.current;
    if (!pointerState || pointerState.pointerId !== event.pointerId || !activeLayerId) return;
    const point = imagePointFromEvent(event);
    if (!point) return;
    drawStroke(activeLayerId, pointerState.lastPoint, point);
    pointerStateRef.current = { ...pointerState, lastPoint: point };
    redrawComposite();
  }

  function handlePointerEnd(event) {
    if (pointerStateRef.current?.pointerId === event.pointerId) {
      pointerStateRef.current = null;
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function serializeLayers() {
    return layers
      .map((layer) => {
        const canvas = layerCanvasesRef.current.get(layer.instanceId);
        if (!canvas || !canvasHasPaint(canvas)) {
          return null;
        }
        return {
          instanceId: layer.instanceId,
          label: layer.label,
          color: layer.color,
          width: canvas.width,
          height: canvas.height,
          maskDataUrl: canvas.toDataURL('image/png'),
        };
      })
      .filter(Boolean);
  }

  async function submit(action) {
    if (!onSubmit) return;
    await onSubmit({
      action,
      instances: serializeLayers(),
    });
  }

  const activeLayer = layers.find((layer) => layer.instanceId === activeLayerId) || null;

  return (
    <section className="manual-segmentation-card">
      <div className="manual-segmentation-head">
        <div>
          <h3 className="card-title">Hand-tuned masking</h3>
          <p className="card-text">
            Add object layers, paint directly on the reference image, and continue the pipeline once the masks look right.
          </p>
        </div>
        <div className="manual-segmentation-actions">
          <button type="button" onClick={() => submit('save')} disabled={busy || loading || !referencePath}>
            {busy ? 'Working…' : 'Save masks'}
          </button>
          <button type="button" onClick={() => submit('continue')} disabled={busy || loading || !referencePath}>
            {busy ? 'Working…' : 'Continue pipeline'}
          </button>
        </div>
      </div>

      <div className="manual-segmentation-toolbar">
        <div className="manual-input-group">
          <label className="form-label" htmlFor="mask-label-input">New object</label>
          <div className="manual-inline-form">
            <input
              id="mask-label-input"
              className="text-input"
              value={draftLabel}
              onChange={(event) => setDraftLabel(event.target.value)}
              placeholder="chair"
            />
            <button type="button" onClick={() => addLayer()} disabled={busy || loading || !referencePath}>
              Add object
            </button>
          </div>
        </div>

        {suggestions.length ? (
          <div className="manual-suggestion-row">
            <span className="meta-label">Suggested labels</span>
            <div className="manual-suggestion-list">
              {suggestions.map((label) => (
                <button
                  key={label}
                  type="button"
                  className="chip chip-button"
                  onClick={() => addLayer(label)}
                  disabled={busy || loading || !referencePath}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        <div className="manual-tool-row">
          <button
            type="button"
            className={`backend-option ${tool === 'paint' ? 'backend-option-active' : ''}`}
            onClick={() => setTool('paint')}
            disabled={busy}
          >
            Paint
          </button>
          <button
            type="button"
            className={`backend-option ${tool === 'erase' ? 'backend-option-active' : ''}`}
            onClick={() => setTool('erase')}
            disabled={busy}
          >
            Erase
          </button>
          <label className="brush-control" htmlFor="brush-size-input">
            <span className="meta-label">Brush</span>
            <input
              id="brush-size-input"
              type="range"
              min="4"
              max="72"
              step="2"
              value={brushSize}
              onChange={(event) => setBrushSize(Number(event.target.value))}
              disabled={busy}
            />
            <span className="mono">{brushSize}px</span>
          </label>
          {activeLayer ? (
            <button type="button" onClick={() => clearLayer(activeLayer.instanceId)} disabled={busy}>
              Clear active
            </button>
          ) : null}
        </div>
      </div>

      <div className="manual-segmentation-grid">
        <div className="manual-canvas-panel">
          <div className="manual-canvas-frame">
            {referencePath ? (
              <>
                <canvas
                  ref={displayCanvasRef}
                  className="manual-canvas"
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={handlePointerEnd}
                  onPointerCancel={handlePointerEnd}
                  onPointerLeave={handlePointerEnd}
                />
                {!activeLayerId ? (
                  <div className="manual-canvas-overlay">
                    Add an object layer to start painting a mask.
                  </div>
                ) : null}
              </>
            ) : (
              <div className="empty">The reference image is not available yet.</div>
            )}
            {loading ? <div className="manual-loading">Loading reference image…</div> : null}
          </div>
        </div>

        <div className="manual-layer-panel">
          <div className="manual-layer-header">
            <span className="meta-label">Mask layers</span>
            <span className="mono">{layers.length}</span>
          </div>

          {layers.length ? (
            <div className="manual-layer-list">
              {layers.map((layer, index) => (
                <div
                  key={layer.instanceId}
                  className={`manual-layer-row ${layer.instanceId === activeLayerId ? 'manual-layer-row-active' : ''}`}
                >
                  <button
                    type="button"
                    className="manual-layer-select"
                    onClick={() => {
                      setActiveLayerId(layer.instanceId);
                      redrawComposite(layers, layer.instanceId);
                    }}
                  >
                    <span className="manual-layer-swatch" style={{ backgroundColor: layer.color || nextColor(index) }} />
                    <span>{layer.label}</span>
                  </button>
                  <input
                    className="manual-layer-input"
                    value={layer.label}
                    onChange={(event) => updateLayer(layer.instanceId, () => ({ label: event.target.value }))}
                    disabled={busy}
                  />
                  <div className="manual-layer-row-actions">
                    <button type="button" onClick={() => clearLayer(layer.instanceId)} disabled={busy}>
                      Clear
                    </button>
                    <button type="button" onClick={() => removeLayer(layer.instanceId)} disabled={busy}>
                      Remove
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty">
              No manual objects yet. Add one and paint directly on the reference image.
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
