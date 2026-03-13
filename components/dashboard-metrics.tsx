"use client";

import { ArrowDownRight, ArrowUpRight, Layers3, ShieldCheck } from "lucide-react";

import type { DashboardMetrics as Metrics } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface DashboardMetricsProps {
  metrics: Metrics;
}

const cards = [
  {
    key: "totalApplications",
    label: "Total Applications",
    icon: Layers3
  },
  {
    key: "approvedLoans",
    label: "Approved Loans",
    icon: ArrowUpRight
  },
  {
    key: "declinedLoans",
    label: "Declined Loans",
    icon: ArrowDownRight
  },
  {
    key: "averageDefaultProbability",
    label: "Average Default Probability",
    icon: ShieldCheck
  }
] as const;

export function DashboardMetrics({ metrics }: DashboardMetricsProps) {
  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {cards.map((card) => {
        const Icon = card.icon;
        const value = metrics[card.key];

        return (
          <Card key={card.key}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
              <CardTitle className="text-sm text-muted-foreground">
                {card.label}
              </CardTitle>
              <div className="rounded-full bg-primary/10 p-2 text-primary">
                <Icon className="h-4 w-4" />
              </div>
            </CardHeader>
            <CardContent>
              <div className="font-heading text-3xl font-semibold">
                {card.key === "averageDefaultProbability"
                  ? `${(value * 100).toFixed(1)}%`
                  : value}
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
