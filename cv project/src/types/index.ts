export type PredictionLabel = 'Fractured' | 'Not Fractured'

export interface PredictionResponse {
  prediction: PredictionLabel
  confidence: number
}
