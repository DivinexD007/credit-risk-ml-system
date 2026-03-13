"use client";

import type { FormEvent, ReactNode } from "react";
import { useState } from "react";
import { Loader2 } from "lucide-react";
import axios from "axios";

import { predictApplication } from "@/lib/api";
import type {
  Education,
  EmploymentType,
  Gender,
  LoanApplication,
  PredictionResponse
} from "@/lib/types";
import { usePortfolioStore } from "@/hooks/use-portfolio-store";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";

const employmentTypes: EmploymentType[] = [
  "Full-time",
  "Part-time",
  "Self-Employed",
  "Unemployed"
];

const genders: Gender[] = ["Male", "Female", "Other"];
const educations: Education[] = ["High School", "Bachelor", "Master", "PhD"];

const initialState: LoanApplication = {
  Age: 35,
  Income: 85000,
  LoanAmount: 24000,
  CreditScore: 710,
  MonthsEmployed: 48,
  YearsExperience: 9,
  NumCreditLines: 4,
  InterestRate: 8.5,
  LoanTerm: 36,
  EmploymentType: "Full-time",
  Gender: "Female",
  Education: "Bachelor",
  City: "Berlin"
};

interface LoanFormProps {
  onPrediction: (
    application: LoanApplication,
    prediction: PredictionResponse
  ) => void;
}

export function LoanForm({ onPrediction }: LoanFormProps) {
  const [form, setForm] = useState<LoanApplication>(initialState);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { addSinglePrediction } = usePortfolioStore();

  const handleChange = (
    field: keyof LoanApplication,
    value: string | number
  ) => {
    setForm((current) => ({
      ...current,
      [field]: value
    }));
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);

    try {
      const prediction = await predictApplication(form);
      addSinglePrediction(form, prediction);
      onPrediction(form, prediction);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.detail ?? "Unable to score application.");
      } else {
        setError("Unexpected error while scoring application.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Loan Application Intake</CardTitle>
      </CardHeader>
      <CardContent>
        <form className="grid gap-5 md:grid-cols-2" onSubmit={handleSubmit}>
          <Field label="Age">
            <Input
              min={18}
              max={100}
              type="number"
              value={form.Age}
              onChange={(event) => handleChange("Age", Number(event.target.value))}
            />
          </Field>
          <Field label="Income">
            <Input
              min={0}
              type="number"
              value={form.Income}
              onChange={(event) =>
                handleChange("Income", Number(event.target.value))
              }
            />
          </Field>
          <Field label="Loan Amount">
            <Input
              min={0}
              type="number"
              value={form.LoanAmount}
              onChange={(event) =>
                handleChange("LoanAmount", Number(event.target.value))
              }
            />
          </Field>
          <Field label="Credit Score">
            <Input
              min={300}
              max={850}
              type="number"
              value={form.CreditScore}
              onChange={(event) =>
                handleChange("CreditScore", Number(event.target.value))
              }
            />
          </Field>
          <Field label="Months Employed">
            <Input
              min={0}
              type="number"
              value={form.MonthsEmployed}
              onChange={(event) =>
                handleChange("MonthsEmployed", Number(event.target.value))
              }
            />
          </Field>
          <Field label="Years Experience">
            <Input
              min={0}
              type="number"
              value={form.YearsExperience}
              onChange={(event) =>
                handleChange("YearsExperience", Number(event.target.value))
              }
            />
          </Field>
          <Field label="Number of Credit Lines">
            <Input
              min={0}
              type="number"
              value={form.NumCreditLines}
              onChange={(event) =>
                handleChange("NumCreditLines", Number(event.target.value))
              }
            />
          </Field>
          <Field label="Interest Rate">
            <Input
              min={0}
              max={100}
              step="0.1"
              type="number"
              value={form.InterestRate}
              onChange={(event) =>
                handleChange("InterestRate", Number(event.target.value))
              }
            />
          </Field>
          <Field label="Loan Term">
            <Input
              min={1}
              type="number"
              value={form.LoanTerm}
              onChange={(event) =>
                handleChange("LoanTerm", Number(event.target.value))
              }
            />
          </Field>
          <Field label="Employment Type">
            <Select
              value={form.EmploymentType}
              onChange={(event) =>
                handleChange("EmploymentType", event.target.value)
              }
            >
              {employmentTypes.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Gender">
            <Select
              value={form.Gender}
              onChange={(event) => handleChange("Gender", event.target.value)}
            >
              {genders.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Education">
            <Select
              value={form.Education}
              onChange={(event) => handleChange("Education", event.target.value)}
            >
              {educations.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </Select>
          </Field>
          <Field className="md:col-span-2" label="City">
            <Input
              type="text"
              value={form.City}
              onChange={(event) => handleChange("City", event.target.value)}
            />
          </Field>

          {error ? (
            <div className="md:col-span-2 rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive-foreground">
              {error}
            </div>
          ) : null}

          <div className="md:col-span-2 flex justify-end">
            <Button disabled={submitting} size="lg" type="submit">
              {submitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Scoring application
                </>
              ) : (
                "Submit for decision"
              )}
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

function Field({
  children,
  className,
  label
}: {
  children: ReactNode;
  className?: string;
  label: string;
}) {
  return (
    <div className={className}>
      <Label className="mb-2 block">{label}</Label>
      {children}
    </div>
  );
}
