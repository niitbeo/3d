import React, { Suspense, useState, useRef, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stage } from '@react-three/drei'
import { useLoader } from '@react-three/fiber'
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader'
import * as THREE from 'three'

function DynamicModel({ url, viewMode }) {
  const isGLB = url.toLowerCase().includes('.glb')
  const Loader = isGLB ? GLTFLoader : OBJLoader
  const loadedData = useLoader(Loader, url)
  
  const modelRoot = isGLB ? loadedData.scene : loadedData

  useEffect(() => {
    if (modelRoot) {
      modelRoot.traverse((child) => {
        if (child.isMesh) {
          if (viewMode === 'normal') {
            child.material = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
          } else if (viewMode === 'clay') {
            // Clay/matcap render - trắng bóng, ấn tượng
            child.material = new THREE.MeshPhysicalMaterial({
              color: 0xf0f0f0,
              roughness: 0.3,
              metalness: 0.1,
              clearcoat: 0.8,
              clearcoatRoughness: 0.2,
              side: THREE.DoubleSide,
            });
          } else {
            // Keep original material for GLB if not 'normal' mode.
            if (!isGLB || !child.material) {
              child.material = new THREE.MeshStandardMaterial({ 
                vertexColors: true, 
                roughness: 1, 
                metalness: 0,
                side: THREE.DoubleSide
              });
            } else {
              child.material.side = THREE.DoubleSide;
            }
          }
        }
      })
    }
  }, [modelRoot, viewMode, isGLB])

  return <primitive object={modelRoot} />
}

function SceneControls({ resetTrigger }) {
  const controlsRef = useRef()
  
  useEffect(() => {
    if (resetTrigger > 0 && controlsRef.current) {
      controlsRef.current.reset()
    }
  }, [resetTrigger])

  return <OrbitControls ref={controlsRef} makeDefault />
}

export default function App() {
  // Steps: 0 = Upload, 1 = Tách nền, 2 = Chuẩn hóa, 3 = Không gian, 4 = Conditioning, 5 = 3D, 6 = Texture
  const [step, setStep] = useState(0)
  
  const [modelUrl, setModelUrl] = useState(null)
  
  const [isRemovingBg, setIsRemovingBg] = useState(false)
  const [bgProgress, setBgProgress] = useState(0)
  const [bgData, setBgData] = useState(null)
  
  const [isGeometry, setIsGeometry] = useState(false)
  const [geometryProgress, setGeometryProgress] = useState(0)
  const [geometryData, setGeometryData] = useState(null)
  
  const [isConditioning, setIsConditioning] = useState(false)
  const [conditioningProgress, setConditioningProgress] = useState(0)
  const [conditioningData, setConditioningData] = useState(null)
  
  const [isGenerating3D, setIsGenerating3D] = useState(false)
  const [generate3dProgress, setGenerate3dProgress] = useState(0)
  const [generate3dData, setGenerate3dData] = useState(null)
  const [generate3dLogs, setGenerate3dLogs] = useState([])
  const [isTexture, setIsTexture] = useState(false)
  const [textureProgress, setTextureProgress] = useState(0)
  const [textureData, setTextureData] = useState(null)
  
  const [resetTrigger, setResetTrigger] = useState(0)
  const [viewMode, setViewMode] = useState('color')
  const fileInputRef = useRef(null)

  // Polling % tiến độ từ Backend (BG Removal)
  useEffect(() => {
    let interval;
    if (isRemovingBg) {
      interval = setInterval(async () => {
        try {
          const res = await fetch('/api/progress_bg')
          const data = await res.json()
          if (data.progress >= 0 && data.progress <= 100) {
            setBgProgress(data.progress)
          }
        } catch (e) {}
      }, 500)
    } else {
      setBgProgress(0)
    }
    return () => clearInterval(interval)
  }, [isRemovingBg])
  
  // Polling % tiến độ từ Backend (Geometry)
  useEffect(() => {
    let interval;
    if (isGeometry) {
      interval = setInterval(async () => {
        try {
          const res = await fetch('/api/progress_geometry')
          const data = await res.json()
          if (data.progress >= 0 && data.progress <= 100) {
            setGeometryProgress(data.progress)
          }
        } catch (e) {}
      }, 500)
    } else {
      setGeometryProgress(0)
    }
    return () => clearInterval(interval)
  }, [isGeometry])

  useEffect(() => {
    let interval;
    if (isConditioning) {
      interval = setInterval(async () => {
        try {
          const res = await fetch('/api/progress_conditioning')
          const data = await res.json()
          if (data.progress >= 0 && data.progress <= 100) setConditioningProgress(data.progress)
        } catch (e) {}
      }, 500)
    } else { setConditioningProgress(0) }
    return () => clearInterval(interval)
  }, [isConditioning])
  
  useEffect(() => {
    let interval;
    if (isGenerating3D) {
      interval = setInterval(async () => {
        try {
          const res = await fetch('/api/progress_generate_3d')
          const data = await res.json()
          if (data.progress >= 0 && data.progress <= 100) setGenerate3dProgress(data.progress)
          
          const logRes = await fetch('/api/logs_generate_3d')
          const logData = await logRes.json()
          if (logData.logs) setGenerate3dLogs(logData.logs)
        } catch (e) {}
      }, 2000)
    } else { 
      setGenerate3dProgress(0) 
      if (!isGenerating3D) setGenerate3dLogs([])
    }
    return () => clearInterval(interval)
  }, [isGenerating3D])
  
  useEffect(() => {
    let interval;
    if (isTexture) {
      interval = setInterval(async () => {
        try {
          const res = await fetch('/api/progress_texture')
          const data = await res.json()
          if (data.progress >= 0 && data.progress <= 100) setTextureProgress(data.progress)
        } catch (e) {}
      }, 1000)
    } else { setTextureProgress(0) }
    return () => clearInterval(interval)
  }, [isTexture])

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    setIsRemovingBg(true)
    setBgProgress(0)
    setStep(0)
    setGeometryData(null)
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('/api/remove_bg', {
        method: 'POST',
        body: formData,
      })
      
      const data = await response.json()
      if (response.ok && data.status === 'success') {
        setBgData(data)
        setStep(1) // Move to Verify BG step
      } else {
        alert("Có lỗi xảy ra khi tách nền: " + (data.detail || "Không rõ lỗi"))
      }
    } catch (error) {
      console.error(error)
      alert("Lỗi kết nối tới server! Vui lòng kiểm tra terminal xem server Python đã chạy chưa.")
    } finally {
      setIsRemovingBg(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleGeometry = async () => {
    if (!bgData || !bgData.base_name) return;
    
    setIsGeometry(true)
    setGeometryProgress(0)
    
    const formData = new FormData()
    formData.append('base_name', bgData.base_name)

    try {
      const response = await fetch('/api/geometry', {
        method: 'POST',
        body: formData,
      })
      
      const data = await response.json()
      if (response.ok && data.status === 'success') {
        setGeometryData(data)
        setStep(3) // Geometry step
      } else {
        alert("Có lỗi xảy ra: " + (data.detail || "Không rõ lỗi"))
      }
    } catch (error) {
      console.error(error)
      alert("Lỗi kết nối tới server Geometry!")
    } finally {
      setIsGeometry(false)
    }
  }

  const handleConditioning = async () => {
    if (!bgData || !bgData.base_name) return;
    setIsConditioning(true)
    setConditioningProgress(0)
    const formData = new FormData()
    formData.append('bg_image', bgData.base_name)
    try {
      const response = await fetch('/api/conditioning', { method: 'POST', body: formData })
      const data = await response.json()
      if (response.ok && data.status === 'success') {
        setConditioningData(data)
        setStep(4)
      } else alert("Có lỗi: " + (data.detail || "Không rõ lỗi"))
    } catch (error) { alert("Lỗi kết nối!") } finally { setIsConditioning(false) }
  }
  
  const handleGenerate3D = async () => {
    if (!conditioningData || !conditioningData.npz_file) return;
    setIsGenerating3D(true)
    setGenerate3dProgress(0)
    setModelUrl(null)
    const formData = new FormData()
    formData.append('npz_file', conditioningData.npz_file)
    try {
      const response = await fetch('/api/generate_3d', { method: 'POST', body: formData })
      const data = await response.json()
      if (response.ok && data.status === 'success') {
        setGenerate3dData(data)
        setModelUrl(data.model_url)
        setStep(5)
      } else { alert("Có lỗi: " + (data.detail || "Không rõ lỗi")) }
    } catch (error) { alert("Lỗi kết nối!") } finally { setIsGenerating3D(false) }
  }
  
  const handleTexture = async () => {
    if (!bgData || !bgData.base_name) return;
    setIsTexture(true)
    setTextureProgress(0)
    const formData = new FormData()
    formData.append('bg_image', bgData.base_name)
    try {
      const response = await fetch('/api/texture', { method: 'POST', body: formData })
      const data = await response.json()
      if (response.ok && data.status === 'success') {
        setTextureData(data)
        setModelUrl(data.model_url)
        setStep(6)
      } else { alert("Có lỗi: " + (data.detail || "Không rõ lỗi")) }
    } catch (error) { alert("Lỗi kết nối!") } finally { setIsTexture(false) }
  }

  // Progress Bar Component
  const ProgressBar = ({ label, progress }) => (
    <div className="loading-indicator" style={{ alignItems: 'flex-start' }}>
      <div className="spinner" style={{ marginTop: '2px' }}></div>
      <div style={{ flex: 1 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontWeight: 500 }}>
          <span>{label}</span>
          <span>{progress}%</span>
        </div>
        <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
          <div style={{ width: `${progress}%`, height: '100%', background: 'linear-gradient(90deg, #00c6ff, #0072ff)', transition: 'width 0.4s ease-out' }}></div>
        </div>
      </div>
    </div>
  )

  return (
    <>
      <div className="ui-layer" style={{ width: step > 0 && step < 7 ? '600px' : '400px' }}>
        <h1>AI Photobooth Pipeline</h1>
        <p>Quy trình: Tách nền &rarr; Chuẩn hóa &rarr; Geometry &rarr; 3D</p>
        
        <input 
          type="file" 
          accept="image/*" 
          style={{ display: 'none' }} 
          ref={fileInputRef}
          onChange={handleFileChange}
        />
        
        {step === 0 && (
          <button 
            className="upload-btn" 
            onClick={handleUploadClick}
            disabled={isRemovingBg}
          >
            Tải ảnh lên để bắt đầu
          </button>
        )}
        
        {step === 999 && (
          <button 
            className="upload-btn" 
            onClick={() => setStep(0)}
          >
            Tải ảnh khác
          </button>
        )}

        {isRemovingBg && <ProgressBar label="BƯỚC 1: Tách nền (BiRefNet + SAM2)" progress={bgProgress} />}
        
        {step === 1 && bgData && (
          <div style={{ marginTop: '20px', background: 'rgba(0,0,0,0.4)', padding: '20px', borderRadius: '12px' }}>
            <h3 style={{ marginTop: 0, marginBottom: '15px' }}>BƯỚC 1: KẾT QUẢ TÁCH NỀN</h3>
            <div style={{ display: 'flex', gap: '15px' }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '12px', marginBottom: '5px', opacity: 0.8 }}>MASK TỪ AI</div>
                <img src={bgData.mask_url} alt="Mask" style={{ width: '100%', borderRadius: '8px', background: '#333' }} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '12px', marginBottom: '5px', opacity: 0.8 }}>ẢNH TRONG SUỐT</div>
                <img src={bgData.alpha_url} alt="Alpha" style={{ width: '100%', borderRadius: '8px', background: '#333' }} />
              </div>
            </div>
            
            <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
              <button 
                className="upload-btn" 
                style={{ flex: 1, background: 'linear-gradient(90deg, #ff8a00, #e52e71)', color: 'white', borderColor: 'transparent' }}
                onClick={() => setStep(2)}
              >
                Tiếp tục: Chuẩn hóa Form &rarr;
              </button>
            </div>
          </div>
        )}
        
        {step === 2 && bgData && (
          <div style={{ marginTop: '20px', background: 'rgba(0,0,0,0.4)', padding: '20px', borderRadius: '12px' }}>
            <h3 style={{ marginTop: 0, marginBottom: '15px' }}>BƯỚC 2: CHUẨN HÓA KÍCH THƯỚC (CANONICAL)</h3>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '12px', marginBottom: '5px', opacity: 0.8 }}>CANVAS 2048x2048 (SCALE 85%)</div>
              <img src={bgData.canonical_preview} alt="Canonical" style={{ width: '80%', borderRadius: '8px', border: '2px solid #555' }} />
            </div>
            
            {isGeometry ? (
              <div style={{ marginTop: '20px' }}>
                <ProgressBar label="Đang chạy Depth Anything V2 Large..." progress={geometryProgress} />
              </div>
            ) : (
              <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
                <button 
                  className="upload-btn" 
                  style={{ flex: 1, background: 'rgba(255, 255, 255, 0.1)', borderColor: 'transparent' }}
                  onClick={() => setStep(1)}
                >
                  &larr; Quay lại
                </button>
                <button 
                  className="upload-btn" 
                  style={{ flex: 2, background: 'linear-gradient(90deg, #00c6ff, #0072ff)', color: 'white', borderColor: 'transparent' }}
                  onClick={handleGeometry}
                >
                  Tính toán Bản đồ Không gian &rarr;
                </button>
              </div>
            )}
          </div>
        )}
        
        {step === 3 && geometryData && (
          <div style={{ marginTop: '20px', background: 'rgba(0,0,0,0.4)', padding: '20px', borderRadius: '12px' }}>
            <h3 style={{ marginTop: 0, marginBottom: '15px' }}>BƯỚC 3: CHIỀU SÂU & ÁNH SÁNG</h3>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '12px', marginBottom: '5px', opacity: 0.8 }}>RGB | DEPTH MAP | SURFACE NORMAL</div>
              <img src={geometryData.preview_url} alt="Geometry" style={{ width: '100%', borderRadius: '8px', border: '1px solid #444' }} />
            </div>
            
            {geometryData.metadata && (
              <div style={{ marginTop: '20px', background: 'rgba(0,0,0,0.6)', padding: '15px', borderRadius: '8px', border: '1px solid #333' }}>
                <h4 style={{ marginTop: 0, marginBottom: '10px', color: '#7eb3ff', fontSize: '14px' }}>⚙️ TECHNICAL METADATA</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', fontSize: '13px', fontFamily: 'monospace' }}>
                  {Object.entries(geometryData.metadata).map(([key, value]) => (
                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px dashed #444', paddingBottom: '4px' }}>
                      <span style={{ color: '#aaa' }}>{key}</span>
                      <span style={{ color: value === 'PASS' ? '#00e676' : '#fff' }}>{value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            
            <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
              <button 
                className="upload-btn" 
                style={{ flex: 1, background: 'rgba(255, 255, 255, 0.1)', borderColor: 'transparent' }}
                onClick={() => setStep(2)}
              >
                &larr; Quay lại
              </button>
              <button 
                className="upload-btn" 
                style={{ flex: 2, background: 'linear-gradient(90deg, #ff8a00, #e52e71)', color: 'white', borderColor: 'transparent' }}
                onClick={handleConditioning}
              >
                Điều kiện Hình học &rarr;
              </button>
            </div>
          </div>
        )}

        {isConditioning && (
          <div style={{ marginTop: '20px' }}>
            <ProgressBar label="BƯỚC 4: Geometry Conditioning..." progress={conditioningProgress} />
          </div>
        )}

        {step === 4 && conditioningData && (
          <div style={{ marginTop: '20px', background: 'rgba(0,0,0,0.4)', padding: '20px', borderRadius: '12px' }}>
            <h3 style={{ marginTop: 0, marginBottom: '15px' }}>BƯỚC 4: GEOMETRY CONDITIONING</h3>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '12px', marginBottom: '5px', opacity: 0.8 }}>MULTI-VIEW HINT & VISIBILITY</div>
              <img src={conditioningData.preview_url} alt="Conditioning" style={{ width: '100%', borderRadius: '8px', border: '1px solid #444' }} />
            </div>
            {isGenerating3D ? (
              <div style={{ marginTop: '20px' }}>
                <ProgressBar label="BƯỚC 5: Đang dựng Mesh 3D (InstantMesh)..." progress={generate3dProgress} />
              </div>
            ) : (
              <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
                <button className="upload-btn" style={{ flex: 1, background: 'rgba(255, 255, 255, 0.1)', borderColor: 'transparent' }} onClick={() => setStep(3)}>&larr; Quay lại</button>
                <button className="upload-btn" style={{ flex: 2, background: 'linear-gradient(90deg, #ff8a00, #e52e71)', color: 'white', borderColor: 'transparent' }} onClick={handleGenerate3D}>Dựng Lưới 3D (TRELLIS 4B) &rarr;</button>
              </div>
            )}
          </div>
        )}

        {step === 5 && !isGenerating3D && generate3dData && (
          <div style={{ marginTop: '20px', background: 'rgba(0,0,0,0.4)', padding: '20px', borderRadius: '12px' }}>
            <h3 style={{ marginTop: 0, marginBottom: '15px' }}>BƯỚC 5: 3D RECONSTRUCTION (InstantMesh)</h3>
            <p style={{ fontSize: '14px', color: '#aaa', marginBottom: '15px' }}>Lưới Mesh đã được tạo và tinh chỉnh với dữ liệu conditioning.</p>
            
            <div style={{ display: 'flex', gap: '10px', flexDirection: 'column' }}>
                <button className="upload-btn" style={{ background: 'rgba(255, 255, 255, 0.05)' }} onClick={() => setResetTrigger(prev => prev + 1)}>Trở về góc nhìn mặc định</button>
                <button className="upload-btn" style={{ background: 'linear-gradient(90deg, #ff7eb3, #7eb3ff)', color: 'white' }} onClick={() => setViewMode(prev => prev === 'color' ? 'clay' : prev === 'clay' ? 'normal' : 'color')}>
                  {viewMode === 'color' ? "🏛️ Đổi sang Clay (Trắng bóng)" : viewMode === 'clay' ? "🎨 Đổi sang Mesh Normal" : "🖼️ Đổi sang Texture"}
                </button>
            </div>

            {isTexture ? (
              <div style={{ marginTop: '20px' }}><ProgressBar label="BƯỚC 6: Đang phủ Vật liệu PBR (Texture)..." progress={textureProgress} /></div>
            ) : (
              <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
                <button className="upload-btn" style={{ flex: 1, background: 'rgba(255, 255, 255, 0.1)', borderColor: 'transparent' }} onClick={() => setStep(4)}>&larr; Quay lại</button>
                <button className="upload-btn" style={{ flex: 2, background: 'linear-gradient(90deg, #00c6ff, #0072ff)', color: 'white', borderColor: 'transparent' }} onClick={handleTexture}>Trải UV & Đổ Texture (PBR) &rarr;</button>
              </div>
            )}
          </div>
        )}

        {step === 6 && !isTexture && textureData && (
          <div style={{ marginTop: '20px', background: 'rgba(0,0,0,0.4)', padding: '20px', borderRadius: '12px' }}>
            <h3 style={{ marginTop: 0, marginBottom: '15px' }}>BƯỚC 6: PBR TEXTURE ENGINE</h3>
            <p style={{ fontSize: '14px', color: '#00e676', fontWeight: 'bold' }}>Đã hoàn tất quy trình AI Image-to-3D Pipeline!</p>
            
            <div style={{ display: 'flex', gap: '10px', flexDirection: 'column', marginTop: '15px' }}>
                <button className="upload-btn" style={{ background: 'rgba(255, 255, 255, 0.05)' }} onClick={() => setResetTrigger(prev => prev + 1)}>Trở về góc nhìn mặc định</button>
            </div>

            <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
              <button className="upload-btn" style={{ flex: 1, background: 'rgba(255, 255, 255, 0.1)' }} onClick={() => setStep(0)}>Tải ảnh khác để bắt đầu lại</button>
            </div>
          </div>
        )}
      </div>

      {(step === 0 || step === 5 || step === 6) && (
        <Canvas shadows dpr={[1, 2]} camera={{ position: [0, 0, 5], fov: 45 }}>
          <color attach="background" args={['#1a1a1a']} />
          <ambientLight intensity={0.4} />
          <directionalLight position={[5, 5, 5]} intensity={1.2} castShadow />
          <directionalLight position={[-3, 3, -3]} intensity={0.6} />
          <directionalLight position={[0, -2, 4]} intensity={0.3} />
          <Suspense fallback={null}>
            <Stage environment="studio" intensity={0.8}>
              {modelUrl && <DynamicModel key={modelUrl + viewMode} url={modelUrl} viewMode={viewMode} />}
            </Stage>
          </Suspense>
          <SceneControls resetTrigger={resetTrigger} />
        </Canvas>
      )}

      {isGenerating3D && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: 'calc(650px + (100vw - 650px) / 2)',
          transform: 'translate(-50%, -50%)',
          width: '700px',
          maxWidth: '80%',
          height: '500px',
          maxHeight: '80%',
          background: 'rgba(10, 10, 10, 0.85)', 
          backdropFilter: 'blur(10px)',
          border: '1px solid #333', 
          borderRadius: '12px', 
          padding: '20px', 
          overflowY: 'auto', 
          fontFamily: 'monospace', 
          fontSize: '14px', 
          color: '#00ff00',
          textAlign: 'left',
          display: 'flex',
          flexDirection: 'column-reverse',
          zIndex: 5,
          boxShadow: '0 0 40px rgba(0, 255, 0, 0.15)'
        }}>
          {generate3dLogs.length === 0 ? (
            <div style={{ opacity: 0.7, paddingBottom: '20px' }}>
              <div style={{ fontSize: '18px', color: '#fff', marginBottom: '10px' }}>[TRELLIS 4B] INITIALIZING...</div>
              <div>&gt; Checking quality gate...</div>
              <div>&gt; Loading Checkpoint microsoft/TRELLIS.2-4B (FP16)...</div>
              <div>&gt; Please wait...</div>
            </div>
          ) : (
            <>
              <div style={{ opacity: 0.7, borderTop: '1px solid #333', marginTop: '10px', paddingTop: '10px' }}>
                <span className="blinking-cursor">_</span>
              </div>
              {generate3dLogs.slice().reverse().map((log, i) => (
                <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <span style={{ color: '#888', marginRight: '10px' }}>{new Date().toISOString().split('T')[1].slice(0,8)}</span> 
                  {log}
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </>
  )
}

