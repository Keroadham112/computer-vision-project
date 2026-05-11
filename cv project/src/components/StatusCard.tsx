import React from 'react'
import styles from './StatusCard.module.css'

type Props = {
  prediction: 'Fractured' | 'Not Fractured'
  confidence: number
}

const StatusCard: React.FC<Props> = ({ prediction, confidence }) => {
  const isFractured = prediction === 'Fractured'
  const barWidth = Math.max(0, Math.min(100, confidence))
  const gradient = isFractured ? 'linear-gradient(90deg,#ef4444,#b91c1c)' : 'linear-gradient(90deg,#34d399,#06b6d4)'

  return (
    <div className={styles.card} style={{ background: isFractured ? 'linear-gradient(90deg, rgba(139, 0, 0, 0.18), rgba(139,0,0,0.06))' : 'linear-gradient(90deg, rgba(6,78,59,0.18), rgba(6,78,59,0.06))' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
        <div>
          <div className={styles.title}>Result</div>
          <div className={styles.label}>{prediction}</div>
        </div>
        <div className={styles.confidence}>
          <div style={{ color: '#cbd5e1', fontSize: 13 }}>Confidence</div>
          <div style={{ fontWeight: 800, fontSize: 20 }}>{confidence.toFixed(1)}%</div>
        </div>
      </div>

      <div className={styles.barWrap}>
        <div className={styles.barInner} style={{ width: `${barWidth}%`, background: gradient }} />
      </div>

      <div className={styles.note}>Model confidence is an estimate — use alongside clinical judgment.</div>
    </div>
  )
}

export default StatusCard
