"use client";

import type { ReactNode } from "react";

import { PortfolioStoreProvider } from "@/hooks/use-portfolio-store";

export function Providers({ children }: { children: ReactNode }) {
  return <PortfolioStoreProvider>{children}</PortfolioStoreProvider>;
}
