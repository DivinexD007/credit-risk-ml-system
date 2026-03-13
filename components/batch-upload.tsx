"use client";

import type { ChangeEvent } from "react";
import { useMemo, useState } from "react";
import { Loader2, UploadCloud } from "lucide-react";
import axios from "axios";

import { predictBatch } from "@/lib/api";
import type { LoanApplication, ScoredLoanApplication } from "@/lib/types";
import { usePortfolioStore } from "@/hooks/use-portfolio-store";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";

const requiredHeaders: Array<keyof LoanApplication> = [
  "Age",
  "Income",
  "LoanAmount",
  "CreditScore",
  "MonthsEmployed",
  "YearsExperience",
  "NumCreditLines",
  "InterestRate",
  "LoanTerm",
  "EmploymentType",
  "Gender",
  "Education",
  "City"
];

interface BatchUploadProps {
  onResults: (results: ScoredLoanApplication[]) => void;
}

export function BatchUpload({ onResults }: BatchUploadProps) {
  const [rows, setRows] = useState<LoanApplication[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const { setBatchPredictions } = usePortfolioStore();

  const summary = useMemo(
    () => ({
      count: rows.length,
      totalExposure: rows.reduce((sum, row) => sum + row.LoanAmount, 0)
    }),
    [rows]
  );

  const handleFileChange = async (
    event: ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    try {
      const text = await file.text();
      const parsedRows = parseCsv(text);
      setRows(parsedRows);
      setError(null);
    } catch (parseError) {
      setRows([]);
      setError(
        parseError instanceof Error
          ? parseError.message
          : "Unable to parse CSV file."
      );
    }
  };

  const handleSubmit = async () => {
    if (rows.length === 0) {
      setError("Upload a CSV file before running batch scoring.");
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const response = await predictBatch(rows);
      const scoredRows = rows.map((row, index) => ({
        ...row,
        ...response.predictions[index]
      }));
      setBatchPredictions(rows, response.predictions);
      onResults(scoredRows);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.detail ?? "Unable to score batch.");
      } else {
        setError("Unexpected error while scoring batch.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Batch Scoring Upload</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <label className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-border/60 bg-background/50 px-6 py-12 text-center hover:border-primary/60 hover:bg-primary/5">
            <UploadCloud className="h-10 w-10 text-primary" />
            <span className="mt-4 text-lg font-medium">
              Upload CSV of loan applicants
            </span>
            <span className="mt-2 text-sm text-muted-foreground">
              Include all application fields exactly as named in the upload template.
            </span>
            <input
              className="hidden"
              accept=".csv"
              type="file"
              onChange={handleFileChange}
            />
          </label>

          <div className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-background/60 p-4 md:flex-row md:items-center md:justify-between">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Rows prepared</p>
              <p className="text-2xl font-semibold">{summary.count}</p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Total exposure</p>
              <p className="text-2xl font-semibold">
                ${summary.totalExposure.toLocaleString()}
              </p>
            </div>
            <Button disabled={submitting || rows.length === 0} onClick={handleSubmit}>
              {submitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Scoring batch
                </>
              ) : (
                "Run batch scoring"
              )}
            </Button>
          </div>

          {error ? (
            <div className="rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive-foreground">
              {error}
            </div>
          ) : null}

          {rows.length > 0 ? (
            <div className="rounded-2xl border border-border/60">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Age</TableHead>
                    <TableHead>Income</TableHead>
                    <TableHead>Loan Amount</TableHead>
                    <TableHead>Employment</TableHead>
                    <TableHead>Education</TableHead>
                    <TableHead>City</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.slice(0, 5).map((row, index) => (
                    <TableRow key={`${row.City}-${index}`}>
                      <TableCell>{row.Age}</TableCell>
                      <TableCell>${row.Income.toLocaleString()}</TableCell>
                      <TableCell>${row.LoanAmount.toLocaleString()}</TableCell>
                      <TableCell>{row.EmploymentType}</TableCell>
                      <TableCell>{row.Education}</TableCell>
                      <TableCell>{row.City}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              {rows.length > 5 ? (
                <div className="border-t border-border/60 px-4 py-3 text-sm text-muted-foreground">
                  Previewing first 5 rows of {rows.length} uploaded applicants.
                </div>
              ) : null}
            </div>
          ) : (
            <div className="flex flex-wrap gap-2">
              {requiredHeaders.map((header) => (
                <Badge key={header} variant="outline">
                  {header}
                </Badge>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function parseCsv(text: string): LoanApplication[] {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length < 2) {
    throw new Error("CSV must include a header row and at least one data row.");
  }

  const headers = parseCsvLine(lines[0]);
  const missingHeaders = requiredHeaders.filter((header) => !headers.includes(header));

  if (missingHeaders.length > 0) {
    throw new Error(`Missing required headers: ${missingHeaders.join(", ")}`);
  }

  return lines.slice(1).map((line, index) => {
    const values = parseCsvLine(line);
    const row = Object.fromEntries(
      headers.map((header, valueIndex) => [header, values[valueIndex] ?? ""])
    ) as Record<keyof LoanApplication, string>;

    try {
      return {
        Age: Number(row.Age),
        Income: Number(row.Income),
        LoanAmount: Number(row.LoanAmount),
        CreditScore: Number(row.CreditScore),
        MonthsEmployed: Number(row.MonthsEmployed),
        YearsExperience: Number(row.YearsExperience),
        NumCreditLines: Number(row.NumCreditLines),
        InterestRate: Number(row.InterestRate),
        LoanTerm: Number(row.LoanTerm),
        EmploymentType: row.EmploymentType as LoanApplication["EmploymentType"],
        Gender: row.Gender as LoanApplication["Gender"],
        Education: row.Education as LoanApplication["Education"],
        City: row.City
      };
    } catch {
      throw new Error(`Unable to parse CSV row ${index + 2}.`);
    }
  });
}

function parseCsvLine(line: string) {
  const values: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];

    if (character === '"') {
      if (inQuotes && line[index + 1] === '"') {
        current += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (character === "," && !inQuotes) {
      values.push(current.trim());
      current = "";
      continue;
    }

    current += character;
  }

  values.push(current.trim());
  return values;
}
