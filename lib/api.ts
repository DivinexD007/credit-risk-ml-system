import axios from "axios";

import type {
  BatchPredictionResponse,
  HealthResponse,
  LoanApplication,
  ModelInfoResponse,
  PredictionResponse
} from "@/lib/types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    "Content-Type": "application/json"
  }
});

export async function getHealth() {
  const { data } = await apiClient.get<HealthResponse>("/health");
  return data;
}

export async function getModelInfo() {
  const { data } = await apiClient.get<ModelInfoResponse>("/model/info");
  return data;
}

export async function predictApplication(application: LoanApplication) {
  const { data } = await apiClient.post<PredictionResponse>(
    "/predict",
    application
  );
  return data;
}

export async function predictBatch(applications: LoanApplication[]) {
  const { data } = await apiClient.post<BatchPredictionResponse>(
    "/predict/batch",
    applications
  );
  return data;
}
