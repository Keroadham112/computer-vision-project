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
        className={`${styles.dropArea} ${dragOver ? styles.dragOver : ''}`}
      >
        <div className={styles.dropInner}>
          {!preview && (
            <>
              <div className={styles.iconWrap}>
                <svg width="54" height="54" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 4v8" stroke="#9CA3AF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M20 12v6a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-6" stroke="#9CA3AF" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
              <p className={styles.primaryText}>Drag and drop an X-ray</p>
              <p className={styles.secondaryText}>PNG or JPG. High-contrast images work best.</p>
              <button type="button" onClick={() => inputRef.current?.click()} className={styles.selectButton}>Select Image</button>
              <input ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={onFileInput} />
            </>
          )}

          {preview && (
            <div className={styles.previewFrame}>
              <img src={preview} alt="preview" className={styles.previewImg} />
            </div>
          )}
        </div>
      </div>

      <div className={styles.actionsCard}>
        <div className={styles.actionsHeader}>
          <div>
            <div className={styles.actionsLabel}>Selected file</div>
            <div className={styles.fileName}>{file?.name ?? 'No file selected'}</div>
          </div>
          <div className={styles.actionsHint}>{preview ? 'Preview loaded' : 'Waiting'}</div>
        </div>

        <div className={styles.actionsRow}>
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

        <div className={styles.tip}>Tip: crop to the region of interest for cleaner Grad-CAM overlays.</div>
      </div>
    </div>
  )
}

export default ImageUploader
