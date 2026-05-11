import axios from 'axios'
import type { PredictionResponse } from '../types'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || 'http://localhost:8000',
  timeout: 30000,
})

export async function predictImage(form: FormData): Promise<PredictionResponse> {
  const { data } = await api.post('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  // Ensure response shape
  return {
    prediction: data.prediction,
    confidence: Number(data.confidence),
  }
}

export default api
