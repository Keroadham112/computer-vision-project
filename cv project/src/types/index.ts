export type PredictionLabel = 'Fractured' | 'Not Fractured'

export interface PredictionResponse {
  prediction: PredictionLabel
  confidence: number
  gradCamImage?: string | null
  gradCamLayer?: string | null
}
