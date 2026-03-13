"use client";

import { useEffect, useMemo, useState } from "react";

import { getHealth } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";
import { usePortfolioStore } from "@/hooks/use-portfolio-store";
import { DashboardMetrics } from "@/components/dashboard-metrics";
import { RiskCharts } from "@/components/risk-charts";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle
} from "@/components/ui/card";

export default function DashboardPage() {
  const { metrics, batchPredictions, singlePredictions } = usePortfolioStore();
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const records = useMemo(
    () => [...singlePredictions, ...batchPredictions],
    [batchPredictions, singlePredictions]
  );

  return (
    <div className="space-y-8">
      <section className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-sm uppercase tracking-[0.24em] text-primary/75">
            Underwriting control center
          </p>
          <h2 className="font-heading text-4xl font-semibold">
            Credit Risk Decision Dashboard
          </h2>
          <p className="mt-3 max-w-3xl text-muted-foreground">
            Review live scoring operations, monitor portfolio quality, and track
            approval performance across submitted applications.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant={health?.model_loaded ? "success" : "warning"}>
            API {health?.status ?? "unavailable"}
          </Badge>
          <Badge variant="outline">
            Threshold {health ? `${(health.threshold * 100).toFixed(1)}%` : "--"}
          </Badge>
        </div>
      </section>

      <DashboardMetrics metrics={metrics} />

      <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <Card>
          <CardHeader>
            <CardTitle>Portfolio trend snapshot</CardTitle>
            <CardDescription>
              Current distribution of risk and decision volume from recent scoring activity.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <RiskCharts data={records} showHistogram={false} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Operational signal</CardTitle>
            <CardDescription>Latest underwriting posture based on scored pipeline.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <SignalBlock
              label="Latest batch size"
              value={batchPredictions.length.toString()}
            />
            <SignalBlock
              label="Latest single submissions"
              value={singlePredictions.length.toString()}
            />
            <SignalBlock
              label="Average default probability"
              value={`${(metrics.averageDefaultProbability * 100).toFixed(1)}%`}
            />
            <SignalBlock
              label="Decision mix"
              value={
                metrics.totalApplications === 0
                  ? "No applications"
                  : `${metrics.approvedLoans} approved / ${metrics.declinedLoans} declined`
              }
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function SignalBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-border/60 bg-background/60 p-4">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="mt-2 text-2xl font-semibold">{value}</p>
    </div>
  );
}
