import React, { useState } from 'react'
import ImageUploader from '../components/ImageUploader'
import PredictionVisuals from '../components/PredictionVisuals'
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
        <header className={styles.hero}>
          <div className={styles.heroCopy}>
            <p className={styles.eyebrow}>AI radiology dashboard</p>
            <h1 className={styles.title}>Fracture Detection Workspace</h1>
            <p className={styles.subtitle}>
              Upload an X-ray, review the model result, and inspect the Grad-CAM overlay in one balanced clinical view.
            </p>
          </div>

          <div className={styles.heroMeta}>
            <div className={styles.metaCard}>
              <span className={styles.metaLabel}>Backend</span>
              <strong>FastAPI</strong>
            </div>
            <div className={styles.metaCard}>
              <span className={styles.metaLabel}>Model</span>
              <strong>MobileNetV2</strong>
            </div>
            <div className={styles.metaCard}>
              <span className={styles.metaLabel}>Explainability</span>
              <strong>Grad-CAM</strong>
            </div>
          </div>
        </header>

        <main className={styles.dashboard}>
          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <div>
                <p className={styles.panelKicker}>Step 1</p>
                <h2 className={styles.panelTitle}>Upload X-ray</h2>
              </div>
              <p className={styles.panelHint}>Use a clear AP or lateral view for best results.</p>
            </div>

            <ImageUploader onFileChange={(file) => handleFile(file)} onPredict={(file) => handlePredict(file)} preview={preview} loading={loading} />

            <div className={styles.infoBox}>
              <strong>How it works:</strong> The app sends the uploaded image to the FastAPI backend. The model was trained to
              detect bone fractures, and the response includes both a prediction and a Grad-CAM explanation.
            </div>
          </section>

          <aside className={styles.sidebar}>
            <div className={styles.panel}>
              <div className={styles.panelHeader}>
                <div>
                  <p className={styles.panelKicker}>Step 2</p>
                  <h2 className={styles.panelTitle}>Prediction result</h2>
                </div>
                <p className={styles.panelHint}>{loading ? 'Analyzing image…' : result ? 'Ready' : 'Waiting for upload'}</p>
              </div>

              {loading && <div className={styles.placeholderText}>Running model on the uploaded X-ray…</div>}
              {!loading && result && <StatusCard prediction={result.prediction} confidence={result.confidence} />}
              {!loading && !result && <div className={styles.placeholderText}>No prediction yet. Upload an image to begin.</div>}
            </div>

            <div className={styles.panel}>
              <div className={styles.panelHeader}>
                <div>
                  <p className={styles.panelKicker}>Step 3</p>
                  <h2 className={styles.panelTitle}>Grad-CAM view</h2>
                </div>
                <p className={styles.panelHint}>Heatmap overlay from the final convolutional block.</p>
              </div>

              <PredictionVisuals
                originalImage={preview}
                gradCamImage={result?.gradCamImage ?? null}
                loading={loading}
              />
            </div>

            <div className={styles.panelSoft}>
              <div className={styles.panelTitleSmall}>Clinical notes</div>
              <ul className={styles.noteList}>
                <li>High-quality X-rays improve accuracy.</li>
                <li>Grad-CAM is an explanation aid, not a diagnosis.</li>
              </ul>
            </div>
          </aside>
        </main>
      </div>
    </div>
  )
}

export default Home
