"use client";

import { useState } from "react";

import type { LoanApplication, PredictionResponse } from "@/lib/types";
import { LoanForm } from "@/components/loan-form";
import { PredictionCard } from "@/components/prediction-card";

export default function LoanApplicationPage() {
  const [application, setApplication] = useState<LoanApplication | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  const handlePrediction = (
    nextApplication: LoanApplication,
    nextPrediction: PredictionResponse
  ) => {
    setApplication(nextApplication);
    setPrediction(nextPrediction);
  };

  return (
    <div className="space-y-8">
      <section>
        <p className="text-sm uppercase tracking-[0.24em] text-primary/75">
          Origination workflow
        </p>
        <h2 className="font-heading text-4xl font-semibold">Loan Application</h2>
        <p className="mt-3 max-w-3xl text-muted-foreground">
          Capture applicant details, submit them to the ML service, and review a
          decision-ready risk assessment immediately.
        </p>
      </section>

      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <LoanForm onPrediction={handlePrediction} />
        <PredictionCard application={application} prediction={prediction} />
      </div>
    </div>
  );
}
