"use client";

import { useEffect, useState } from "react";
import axios from "axios";

import { getModelInfo } from "@/lib/api";
import type { ModelInfoResponse } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function ModelInfoPage() {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getModelInfo()
      .then((data) => {
        setModelInfo(data);
        setError(null);
      })
      .catch((err) => {
        if (axios.isAxiosError(err)) {
          setError(err.response?.data?.detail ?? "Unable to load model information.");
        } else {
          setError("Unable to load model information.");
        }
      });
  }, []);

  return (
    <div className="space-y-8">
      <section>
        <p className="text-sm uppercase tracking-[0.24em] text-primary/75">
          Model governance
        </p>
        <h2 className="font-heading text-4xl font-semibold">Model Info</h2>
        <p className="mt-3 max-w-3xl text-muted-foreground">
          Review model metadata, threshold policy, regulatory context, and feature coverage.
        </p>
      </section>

      {error ? (
        <Card>
          <CardContent className="pt-6 text-destructive-foreground">{error}</CardContent>
        </Card>
      ) : null}

      {modelInfo ? (
        <div className="grid gap-6 xl:grid-cols-[0.9fr_1.1fr]">
          <Card>
            <CardHeader>
              <CardTitle>Model summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-5">
              <InfoRow label="Model type" value={modelInfo.model_type} />
              <InfoRow label="Challenger" value={modelInfo.challenger} />
              <InfoRow label="Framework" value={modelInfo.framework} />
              <InfoRow label="Version" value={modelInfo.version} />
              <InfoRow
                label="Decision threshold"
                value={`${(modelInfo.decision_threshold * 100).toFixed(1)}%`}
              />
              <InfoRow
                label="Threshold derivation"
                value={modelInfo.threshold_derivation}
              />
            </CardContent>
          </Card>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Regulatory context</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-wrap gap-2">
                {modelInfo.regulatory_context.map((item) => (
                  <Badge key={item} variant="outline">
                    {item}
                  </Badge>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Feature list</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="mb-3 text-sm uppercase tracking-[0.18em] text-muted-foreground">
                    Numerical
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {modelInfo.features.numerical.map((feature) => (
                      <Badge key={feature}>{feature}</Badge>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="mb-3 text-sm uppercase tracking-[0.18em] text-muted-foreground">
                    Categorical
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {modelInfo.features.categorical.map((feature) => (
                      <Badge key={feature} variant="secondary">
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      ) : (
        <Card>
          <CardContent className="pt-6 text-muted-foreground">
            Loading model metadata from the ML service.
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-border/60 bg-background/50 p-4">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="mt-2 text-lg font-medium">{value}</p>
    </div>
  );
}
