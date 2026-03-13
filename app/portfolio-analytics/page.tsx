"use client";

import { usePortfolioStore } from "@/hooks/use-portfolio-store";
import { RiskCharts } from "@/components/risk-charts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function PortfolioAnalyticsPage() {
  const { batchPredictions } = usePortfolioStore();

  return (
    <div className="space-y-8">
      <section>
        <p className="text-sm uppercase tracking-[0.24em] text-primary/75">
          Portfolio monitoring
        </p>
        <h2 className="font-heading text-4xl font-semibold">Portfolio Analytics</h2>
        <p className="mt-3 max-w-3xl text-muted-foreground">
          Analyze batch scoring output to understand risk concentration, default
          probability dispersion, and approval posture.
        </p>
      </section>

      {batchPredictions.length === 0 ? (
        <Card>
          <CardHeader>
            <CardTitle>No batch portfolio loaded</CardTitle>
          </CardHeader>
          <CardContent className="text-muted-foreground">
            Run batch scoring first to populate portfolio analytics.
          </CardContent>
        </Card>
      ) : (
        <RiskCharts data={batchPredictions} />
      )}
    </div>
  );
}
