import { useState, useEffect } from 'react'

function App() {
  const [config, setConfig] = useState({
    living_rooms: 1,
    bedrooms: 2,
    master_bedrooms: 1,
    bathrooms: 2,
    kitchens: 1,
    storage: 0
  });

  const [status, setStatus] = useState('IDLE'); // IDLE, GEN_TOPO, TOPO_READY, GEN_PLANS, PLANS_READY, GEN_DOORS, DONE
  const [topology, setTopology] = useState(null);
  const [floorPlans, setFloorPlans] = useState([]);
  const [selectedPlans, setSelectedPlans] = useState([]);
  const [finalPlans, setFinalPlans] = useState([]);
  const [loadingText, setLoadingText] = useState('');
  const [lightboxImage, setLightboxImage] = useState(null);

  const updateConfig = (key, delta) => {
    setConfig(prev => ({
      ...prev,
      [key]: Math.max(0, prev[key] + delta)
    }));
  };

  const handleGenerateTopology = async () => {
    setStatus('GEN_TOPO');
    try {
      const res = await fetch('/api/topology', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      const data = await res.json();
      setTopology(data);
      setStatus('TOPO_READY');
    } catch (err) {
      console.error(err);
      setStatus('IDLE');
    }
  };

  const pollTask = async (taskId, onComplete, onError) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/status/${taskId}`);
        const data = await res.json();
        if (data.status === 'completed') {
          clearInterval(interval);
          onComplete(data.results);
        } else if (data.status === 'failed') {
          clearInterval(interval);
          onError(data.error);
        }
      } catch (err) {
        clearInterval(interval);
        onError(err);
      }
    }, 5000);
  };

  const handleGeneratePlans = async () => {
    setStatus('GEN_PLANS');
    setLoadingText('Generating 15 architectural variations... (This takes about 2-3 minutes)');
    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ nodes: topology.nodes, edges: topology.edges })
      });
      const { task_id } = await res.json();
      
      pollTask(task_id, (results) => {
        setFloorPlans(results);
        setStatus('PLANS_READY');
      }, (err) => {
        console.error(err);
        setStatus('TOPO_READY');
      });
    } catch (err) {
      console.error(err);
      setStatus('TOPO_READY');
    }
  };

  const toggleSelect = (id) => {
    setSelectedPlans(prev => 
      prev.includes(id) ? prev.filter(p => p !== id) : [...prev, id]
    );
  };

  const handleGenerateDoors = async () => {
    setStatus('GEN_DOORS');
    setLoadingText('Applying optimal door placements to selected plans...');
    try {
      const res = await fetch('/api/doors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plans: selectedPlans })
      });
      const { task_id } = await res.json();
      
      pollTask(task_id, (results) => {
        setFinalPlans(results);
        setStatus('DONE');
      }, (err) => {
        console.error(err);
        setStatus('PLANS_READY');
      });
    } catch (err) {
      console.error(err);
      setStatus('PLANS_READY');
    }
  };

  return (
    <div>
      <h1>AI Floor Plan Studio</h1>

      {/* STEP 1: CONFIGURATION */}
      <div className="app-section glass-container">
        <h2>1. Configure Rooms</h2>
        <div className="form-grid">
          {Object.entries(config).map(([key, val]) => (
            <div className="form-group" key={key}>
              <label>{key.replace('_', ' ').toUpperCase()}</label>
              <div className="number-input">
                <button onClick={() => updateConfig(key, -1)} disabled={status !== 'IDLE' && status !== 'TOPO_READY'}>-</button>
                <input type="text" readOnly value={val} />
                <button onClick={() => updateConfig(key, 1)} disabled={status !== 'IDLE' && status !== 'TOPO_READY'}>+</button>
              </div>
            </div>
          ))}
        </div>
        
        {status === 'IDLE' && (
          <button className="btn" onClick={handleGenerateTopology}>
            Generate Topology (Bubble Diagram)
          </button>
        )}
        
        {status === 'GEN_TOPO' && (
          <div className="loader">
            <div className="spinner"></div> Creating logical room connections...
          </div>
        )}
      </div>

      {/* STEP 2: TOPOLOGY */}
      {(status === 'TOPO_READY' || ['GEN_PLANS', 'PLANS_READY', 'GEN_DOORS', 'DONE'].includes(status)) && topology && (
        <div className="app-section glass-container">
          <h2>2. Procedural Topology</h2>
          <div className="image-preview" style={{ marginBottom: '1.5rem' }}>
            <img src={`${topology.image_url}?t=${Date.now()}`} alt="Bubble Diagram" />
          </div>
          
          {status === 'TOPO_READY' && (
             <button className="btn" onClick={handleGeneratePlans}>
               Design Floor Plans (Diffusion)
             </button>
          )}
          
          {status === 'GEN_PLANS' && (
            <div className="loader">
              <div className="spinner"></div> {loadingText}
            </div>
          )}
        </div>
      )}

      {/* STEP 3: FLOOR PLANS */}
      {(status === 'PLANS_READY' || ['GEN_DOORS', 'DONE'].includes(status)) && floorPlans.length > 0 && (
        <div className="app-section glass-container">
          <h2>3. Generated Floor Plans</h2>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
            Select the variations you want to proceed with for physical door placement.
          </p>
          
          <div className="gallery-grid">
            {floorPlans.map(plan => (
              <div 
                key={plan.id}
                className={`gallery-item ${selectedPlans.includes(plan.id) ? 'selected' : ''}`}
                onClick={() => (status === 'PLANS_READY' ? toggleSelect(plan.id) : null)}
              >
                <img src={`${plan.image_url}?t=${Date.now()}`} alt={`Plan ${plan.id}`} />
              </div>
            ))}
          </div>

          {status === 'PLANS_READY' && selectedPlans.length > 0 && (
            <button className="btn" style={{ marginTop: '2rem' }} onClick={handleGenerateDoors}>
              Place Doors on {selectedPlans.length} Selected Plan(s)
            </button>
          )}

          {status === 'GEN_DOORS' && (
            <div className="loader" style={{ marginTop: '2rem' }}>
              <div className="spinner"></div> {loadingText}
            </div>
          )}
        </div>
      )}

      {/* STEP 4: FINAL RESULTS */}
      {status === 'DONE' && finalPlans.length > 0 && (
        <div className="app-section glass-container">
          <h2>4. Final Door Placements</h2>
          <div className="gallery-grid">
            {finalPlans.map(plan => (
              <div key={plan.id} className="gallery-item" onClick={() => setLightboxImage(`${plan.image_url}?t=${Date.now()}`)}>
                <img src={`${plan.image_url}?t=${Date.now()}`} alt={`Final Plan ${plan.id}`} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* LIGHTBOX MODAL */}
      {lightboxImage && (
        <div className="lightbox" onClick={() => setLightboxImage(null)}>
          <div className="lightbox-content" onClick={(e) => e.stopPropagation()}>
            <button className="lightbox-close" onClick={() => setLightboxImage(null)}>×</button>
            <img src={lightboxImage} alt="Enlarged Plan" />
          </div>
        </div>
      )}

    </div>
  )
}

export default App
