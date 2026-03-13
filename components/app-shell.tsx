"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import type { Route } from "next";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  Building2,
  Files,
  LayoutDashboard
} from "lucide-react";

import { cn } from "@/lib/utils";

const navigation = [
  {
    href: "/" as Route,
    label: "Dashboard",
    icon: LayoutDashboard
  },
  {
    href: "/loan-application" as Route,
    label: "Loan Application",
    icon: Building2
  },
  {
    href: "/batch-scoring" as Route,
    label: "Batch Scoring",
    icon: Files
  },
  {
    href: "/portfolio-analytics" as Route,
    label: "Portfolio Analytics",
    icon: BarChart3
  },
  {
    href: "/model-info" as Route,
    label: "Model Info",
    icon: BrainCircuit
  }
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-[1680px] flex-col lg:flex-row">
        <aside className="border-b border-border/60 bg-[#07101f]/90 lg:min-h-screen lg:w-80 lg:border-b-0 lg:border-r">
          <div className="flex h-full flex-col px-6 py-8">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-5">
              <div className="flex items-center gap-3">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-primary/15 text-primary">
                  <Activity className="h-6 w-6" />
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.28em] text-primary/80">
                    Credit Risk
                  </p>
                  <h1 className="font-heading text-xl font-semibold">
                    Decision Dashboard
                  </h1>
                </div>
              </div>
              <p className="mt-4 text-sm text-slate-300">
                Banking workflow for underwriters, batch scoring, and portfolio monitoring.
              </p>
            </div>

            <nav className="mt-8 space-y-2">
              {navigation.map((item) => {
                const Icon = item.icon;
                const active = pathname === item.href;

                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={cn(
                      "flex items-center gap-3 rounded-xl px-4 py-3 text-sm transition-colors",
                      active
                        ? "bg-primary text-primary-foreground"
                        : "text-slate-300 hover:bg-white/5 hover:text-white"
                    )}
                  >
                    <Icon className="h-4 w-4" />
                    {item.label}
                  </Link>
                );
              })}
            </nav>

            <div className="mt-auto rounded-2xl border border-emerald-400/15 bg-emerald-500/10 p-5 text-sm text-emerald-100">
              <p className="font-medium">Connected ML workflow</p>
              <p className="mt-2 text-emerald-100/80">
                FastAPI scoring service expected at <code>localhost:8000</code>.
              </p>
            </div>
          </div>
        </aside>

        <main className="flex-1 bg-[radial-gradient(circle_at_top,rgba(16,185,129,0.12),transparent_28%),linear-gradient(180deg,#020817_0%,#061223_100%)]">
          <div className="min-h-screen bg-dashboard-grid bg-[size:36px_36px]">
            <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-10 lg:py-10">
              {children}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
