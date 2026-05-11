import React, { useRef, useState, useCallback } from 'react'
import styles from './ImageUploader.module.css'

type Props = {
  onFileChange: (file: File | null) => void
  onPredict: (file: File | null) => void
  preview?: string | null
  loading?: boolean
}

const ImageUploader: React.FC<Props> = ({ onFileChange, onPredict, preview, loading }) => {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)

  const handleSelect = (f?: File) => {
    const selected = f ?? null
    setFile(selected)
    onFileChange(selected)
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) handleSelect(f)
  }, [])

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) handleSelect(f)
  }

  return (
    <div className={styles.container}>
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={styles.row}
      >
        <div style={{ flex: 1 }}>
          <div className={`${styles.dropArea} ${dragOver ? styles.dragOver : ''}`}>
            <div className={styles.dropInner}>
              {!preview && (
                <>
                  <svg className="mb-4" width="64" height="64" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 4v8" stroke="#9CA3AF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    <path d="M20 12v6a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-6" stroke="#9CA3AF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <p style={{ margin: '0 0 8px 0', fontSize: 18, color: '#cbd5e1' }}>Drag & drop an X‑ray</p>
                  <p style={{ margin: 0, fontSize: 13, color: '#94a3b8' }}>Supported: JPG, PNG — high contrast X‑rays work best</p>
                  <div style={{ marginTop: 12 }}>
                    <button type="button" onClick={() => inputRef.current?.click()} className={styles.selectButton}>Select Image</button>
                  </div>
                  <input ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={onFileInput} />
                </>
              )}

              {preview && (
                <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <img src={preview} alt="preview" className={styles.previewImg} />
                </div>
              )}
            </div>
          </div>
        </div>

        <div style={{ width: '18rem' }}>
          <div className={styles.actionsCard}>
            <h3 style={{ margin: 0, marginBottom: 8, color: '#cbd5e1' }}>Actions</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <label style={{ color: '#94a3b8', fontSize: 13 }}>Selected file</label>
              <div className={styles.fileName}>{file?.name ?? 'No file selected'}</div>
              <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                <button
                  type="button"
                  onClick={() => onPredict(file)}
                  disabled={!file || loading}
                  className={styles.primary}
                >
                  {loading ? 'Predicting...' : 'Predict'}
                </button>
                <button type="button" onClick={() => { setFile(null); onFileChange(null); }} className={styles.clear}>Clear</button>
              </div>
              <div className={styles.tip}>Tip: Crop to region of interest for better results.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ImageUploader
