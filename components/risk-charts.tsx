"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import type { ScoredLoanApplication } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface RiskChartsProps {
  data: ScoredLoanApplication[];
  showHistogram?: boolean;
}

const riskColors = {
  Low: "hsl(var(--chart-2))",
  Medium: "hsl(var(--chart-3))",
  High: "hsl(var(--chart-4))"
};

function buildRiskDistribution(data: ScoredLoanApplication[]) {
  return ["Low", "Medium", "High"].map((tier) => ({
    name: tier,
    value: data.filter((item) => item.risk_tier === tier).length
  }));
}

function buildDecisionData(data: ScoredLoanApplication[]) {
  const approved = data.filter((item) => item.decision === "Approve").length;
  return [
    {
      name: "Approved",
      value: approved
    },
    {
      name: "Declined",
      value: data.length - approved
    }
  ];
}

function buildHistogram(data: ScoredLoanApplication[]) {
  const bins = [
    { name: "0-20%", min: 0, max: 0.2 },
    { name: "20-40%", min: 0.2, max: 0.4 },
    { name: "40-60%", min: 0.4, max: 0.6 },
    { name: "60-80%", min: 0.6, max: 0.8 },
    { name: "80-100%", min: 0.8, max: 1.001 }
  ];

  return bins.map((bin) => ({
    name: bin.name,
    count: data.filter(
      (item) =>
        item.probability_default >= bin.min &&
        item.probability_default < bin.max
    ).length
  }));
}

export function RiskCharts({ data, showHistogram = true }: RiskChartsProps) {
  const riskDistribution = buildRiskDistribution(data);
  const decisionData = buildDecisionData(data);
  const histogram = buildHistogram(data);

  return (
    <div className="grid gap-6 xl:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Risk Tier Distribution</CardTitle>
        </CardHeader>
        <CardContent className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={riskDistribution}
                dataKey="value"
                nameKey="name"
                innerRadius={70}
                outerRadius={105}
                paddingAngle={5}
              >
                {riskDistribution.map((entry) => (
                  <Cell
                    key={entry.name}
                    fill={riskColors[entry.name as keyof typeof riskColors]}
                  />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Approval vs Decline</CardTitle>
        </CardHeader>
        <CardContent className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={decisionData}>
              <CartesianGrid stroke="rgba(148,163,184,0.16)" vertical={false} />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="value" radius={[8, 8, 0, 0]} fill="hsl(var(--chart-1))" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {showHistogram ? (
        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle>Default Probability Histogram</CardTitle>
          </CardHeader>
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={histogram}>
                <CartesianGrid stroke="rgba(148,163,184,0.16)" vertical={false} />
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" radius={[8, 8, 0, 0]} fill="hsl(var(--chart-3))" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
