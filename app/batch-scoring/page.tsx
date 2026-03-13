"use client";

import { useState } from "react";

import type { ScoredLoanApplication } from "@/lib/types";
import { BatchUpload } from "@/components/batch-upload";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";

export default function BatchScoringPage() {
  const [results, setResults] = useState<ScoredLoanApplication[]>([]);

  return (
    <div className="space-y-8">
      <section>
        <p className="text-sm uppercase tracking-[0.24em] text-primary/75">
          Portfolio intake
        </p>
        <h2 className="font-heading text-4xl font-semibold">Batch Scoring</h2>
        <p className="mt-3 max-w-3xl text-muted-foreground">
          Upload a CSV of applicants, score them in one request, and review a
          portfolio-ready decision table.
        </p>
      </section>

      <BatchUpload onResults={setResults} />

      <Card>
        <CardHeader>
          <CardTitle>Scored Applicants</CardTitle>
        </CardHeader>
        <CardContent>
          {results.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-border/60 px-6 py-12 text-center text-muted-foreground">
              No batch results yet. Upload a CSV and run batch scoring to populate this table.
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Age</TableHead>
                  <TableHead>Income</TableHead>
                  <TableHead>LoanAmount</TableHead>
                  <TableHead>Prediction</TableHead>
                  <TableHead>Probability Default</TableHead>
                  <TableHead>Risk Tier</TableHead>
                  <TableHead>Decision</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {results.map((row, index) => (
                  <TableRow key={`${row.City}-${row.Age}-${index}`}>
                    <TableCell>{row.Age}</TableCell>
                    <TableCell>${row.Income.toLocaleString()}</TableCell>
                    <TableCell>${row.LoanAmount.toLocaleString()}</TableCell>
                    <TableCell>{row.prediction}</TableCell>
                    <TableCell>
                      {(row.probability_default * 100).toFixed(2)}%
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          row.risk_tier === "Low"
                            ? "success"
                            : row.risk_tier === "Medium"
                              ? "warning"
                              : "danger"
                        }
                      >
                        {row.risk_tier}
                      </Badge>
                    </TableCell>
                    <TableCell>{row.decision}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
