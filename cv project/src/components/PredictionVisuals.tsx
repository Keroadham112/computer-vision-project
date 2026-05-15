import React from 'react'
import styles from './PredictionVisuals.module.css'

type Props = {
  originalImage?: string | null
  gradCamImage?: string | null
  loading?: boolean
}

const PredictionVisuals: React.FC<Props> = ({ originalImage, gradCamImage, loading }) => {
  return (
    <div className={styles.card}>
      <div className={styles.headerRow}>
        <div>
          <div className={styles.kicker}>Visual explanation</div>
          <h3 className={styles.title}>What the model focused on</h3>
          <p className={styles.description}>The original X-ray sits beside the Grad-CAM overlay for direct comparison.</p>
        </div>
        <div className={styles.badge}>{loading ? 'Generating...' : gradCamImage ? 'Ready' : 'Waiting'}</div>
      </div>

      <div className={styles.grid}>
        <figure className={styles.figure}>
          <div className={styles.figureLabel}>Original X-ray</div>
          {originalImage ? (
            <img className={styles.image} src={originalImage} alt="Uploaded X-ray preview" />
          ) : (
            <div className={styles.placeholder}>Upload an image to preview it here.</div>
          )}
        </figure>

        <figure className={styles.figure}>
          <div className={styles.figureLabel}>Grad-CAM heatmap</div>
          {gradCamImage ? (
            <img className={styles.image} src={gradCamImage} alt="Grad-CAM overlay showing model focus" />
          ) : (
            <div className={styles.placeholder}>{loading ? 'Creating heatmap...' : 'Run a prediction to generate the explanation.'}</div>
          )}
        </figure>
      </div>

      <div className={styles.footer}>
        The overlay highlights regions that contributed most to the fracture prediction. Use it as a visual aid, not a diagnosis.
      </div>
    </div>
  )
}

export default PredictionVisuals
