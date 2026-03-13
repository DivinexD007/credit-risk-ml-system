"use client";

import {
  PolarAngleAxis,
  RadialBar,
  RadialBarChart,
  ResponsiveContainer
} from "recharts";

import type { LoanApplication, PredictionResponse } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface PredictionCardProps {
  application?: LoanApplication | null;
  prediction?: PredictionResponse | null;
}

function getRiskVariant(riskTier: PredictionResponse["risk_tier"]) {
  if (riskTier === "Low") {
    return "success";
  }

  if (riskTier === "Medium") {
    return "warning";
  }

  return "danger";
}

export function PredictionCard({
  application,
  prediction
}: PredictionCardProps) {
  if (!prediction || !application) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle>Decision Output</CardTitle>
        </CardHeader>
        <CardContent className="flex h-[420px] items-center justify-center text-center text-muted-foreground">
          Submit a loan application to view model decision, risk tier, and probability profile.
        </CardContent>
      </Card>
    );
  }

  const probabilityDefaultPct = prediction.probability_default * 100;

  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-start justify-between">
        <div>
          <CardTitle className="text-2xl">
            {prediction.decision === "Approve" ? "Approve" : "Decline"}
          </CardTitle>
          <p className="mt-2 text-sm text-muted-foreground">
            Applicant from {application.City} with credit score {application.CreditScore}.
          </p>
        </div>
        <Badge variant={getRiskVariant(prediction.risk_tier)}>
          {prediction.risk_tier} Risk
        </Badge>
      </CardHeader>
      <CardContent className="space-y-8">
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <RadialBarChart
              innerRadius="60%"
              outerRadius="100%"
              data={[{ name: "Default", value: probabilityDefaultPct }]}
              startAngle={180}
              endAngle={0}
            >
              <PolarAngleAxis
                type="number"
                domain={[0, 100]}
                tick={false}
              />
              <RadialBar
                dataKey="value"
                cornerRadius={18}
                fill="hsl(var(--chart-4))"
                background={{ fill: "rgba(148,163,184,0.15)" }}
              />
              <text
                x="50%"
                y="65%"
                textAnchor="middle"
                className="fill-white text-3xl font-semibold"
              >
                {probabilityDefaultPct.toFixed(1)}%
              </text>
              <text
                x="50%"
                y="80%"
                textAnchor="middle"
                className="fill-slate-400 text-sm"
              >
                Probability of Default
              </text>
            </RadialBarChart>
          </ResponsiveContainer>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded-2xl border border-border/60 bg-background/60 p-4">
            <p className="text-sm text-muted-foreground">Approval Probability</p>
            <p className="mt-2 text-2xl font-semibold">
              {(prediction.probability_approve * 100).toFixed(2)}%
            </p>
            <Progress className="mt-4" value={prediction.probability_approve * 100} />
          </div>
          <div className="rounded-2xl border border-border/60 bg-background/60 p-4">
            <p className="text-sm text-muted-foreground">Decision Threshold</p>
            <p className="mt-2 text-2xl font-semibold">
              {(prediction.threshold_used * 100).toFixed(1)}%
            </p>
            <p className="mt-4 text-sm text-muted-foreground">
              Model version {prediction.model_version}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
