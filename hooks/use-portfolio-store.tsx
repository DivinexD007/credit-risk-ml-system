"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode
} from "react";

import type {
  DashboardMetrics,
  LoanApplication,
  PredictionResponse,
  ScoredLoanApplication
} from "@/lib/types";

interface PortfolioStoreValue {
  singlePredictions: ScoredLoanApplication[];
  batchPredictions: ScoredLoanApplication[];
  addSinglePrediction: (
    application: LoanApplication,
    prediction: PredictionResponse
  ) => void;
  setBatchPredictions: (
    applications: LoanApplication[],
    predictions: PredictionResponse[]
  ) => void;
  metrics: DashboardMetrics;
}

const STORAGE_KEY = "credit-risk-dashboard-store";

const PortfolioStoreContext = createContext<PortfolioStoreValue | null>(null);

function mergeScored(
  application: LoanApplication,
  prediction: PredictionResponse
): ScoredLoanApplication {
  return {
    ...application,
    ...prediction
  };
}

export function PortfolioStoreProvider({ children }: { children: ReactNode }) {
  const [singlePredictions, setSinglePredictions] = useState<
    ScoredLoanApplication[]
  >([]);
  const [batchPredictions, setBatchPredictionsState] = useState<
    ScoredLoanApplication[]
  >([]);

  useEffect(() => {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      return;
    }

    try {
      const parsed = JSON.parse(stored) as {
        singlePredictions?: ScoredLoanApplication[];
        batchPredictions?: ScoredLoanApplication[];
      };

      setSinglePredictions(parsed.singlePredictions ?? []);
      setBatchPredictionsState(parsed.batchPredictions ?? []);
    } catch {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  useEffect(() => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ singlePredictions, batchPredictions })
    );
  }, [singlePredictions, batchPredictions]);

  const metrics = useMemo<DashboardMetrics>(() => {
    const records = [...singlePredictions, ...batchPredictions];
    const totalApplications = records.length;
    const approvedLoans = records.filter(
      (record) => record.decision === "Approve"
    ).length;
    const declinedLoans = totalApplications - approvedLoans;
    const averageDefaultProbability =
      totalApplications === 0
        ? 0
        : records.reduce(
            (sum, record) => sum + record.probability_default,
            0
          ) / totalApplications;

    return {
      totalApplications,
      approvedLoans,
      declinedLoans,
      averageDefaultProbability
    };
  }, [batchPredictions, singlePredictions]);

  const value = useMemo<PortfolioStoreValue>(
    () => ({
      singlePredictions,
      batchPredictions,
      addSinglePrediction: (application, prediction) => {
        setSinglePredictions((current) => [
          mergeScored(application, prediction),
          ...current
        ]);
      },
      setBatchPredictions: (applications, predictions) => {
        setBatchPredictionsState(
          applications.map((application, index) =>
            mergeScored(application, predictions[index])
          )
        );
      },
      metrics
    }),
    [batchPredictions, metrics, singlePredictions]
  );

  return (
    <PortfolioStoreContext.Provider value={value}>
      {children}
    </PortfolioStoreContext.Provider>
  );
}

export function usePortfolioStore() {
  const context = useContext(PortfolioStoreContext);

  if (!context) {
    throw new Error("usePortfolioStore must be used within PortfolioStoreProvider");
  }

  return context;
}
