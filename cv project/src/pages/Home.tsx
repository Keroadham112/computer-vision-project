import React, { useState } from 'react'
import ImageUploader from '../components/ImageUploader'
import StatusCard from '../components/StatusCard'
import { predictImage } from '../services/api'
import type { PredictionResponse } from '../types'
import styles from './Home.module.css'

const Home: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)

  const handleFile = (file: File | null) => {
    if (!file) {
      setPreview(null)
      setResult(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    setResult(null)
  }

  const handlePredict = async (file: File | null) => {
    if (!file) return
    setLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await predictImage(form)
      setResult(res)
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(err)
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <div className={styles.container}>
        <header className={styles.header}>
          <div>
            <h1 className={styles.title}>Radiology AI — Fracture Detection</h1>
            <p className={styles.subtitle}>Upload an X‑ray and receive a quick fracture prediction with confidence.</p>
          </div>
          <div style={{ color: '#9ca3af', fontSize: 13 }}>Local dev • FastAPI backend</div>
        </header>

        <main className={styles.grid}>
          <section className={styles.leftCard}>
            <ImageUploader onFileChange={(file) => handleFile(file)} onPredict={(file) => handlePredict(file)} preview={preview} loading={loading} />
            <div className={styles.info}><strong>How it works:</strong> The app sends the uploaded image to the FastAPI backend. The model was trained to detect bone fractures — results are for research/demo only.</div>
          </section>

          <aside>
            <div className={styles.rightCard}>
              <h2 style={{ marginTop: 0, marginBottom: 12 }}>Prediction</h2>
              {loading && <div style={{ color: '#9ca3af' }}>Running model...</div>}
              {!loading && result && (<StatusCard prediction={result.prediction} confidence={result.confidence} />)}
              {!loading && !result && <div style={{ color: '#9ca3af' }}>No prediction yet</div>}
            </div>
            <div className={styles.notes}>
              <div style={{ fontWeight: 700, marginBottom: 8 }}>Notes</div>
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                <li>Use high-quality X‑rays for better accuracy.</li>
                <li>Model outputs a probability — consult clinicians.</li>
              </ul>
            </div>
          </aside>
        </main>
      </div>
    </div>
  )
}

export default Home
