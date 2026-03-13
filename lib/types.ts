export type EmploymentType =
  | "Full-time"
  | "Part-time"
  | "Self-Employed"
  | "Unemployed";

export type Gender = "Male" | "Female" | "Other";

export type Education =
  | "High School"
  | "Bachelor"
  | "Master"
  | "PhD";

export type RiskTier = "Low" | "Medium" | "High";
export type Decision = "Approve" | "Decline";

export interface LoanApplication {
  Age: number;
  Income: number;
  LoanAmount: number;
  CreditScore: number;
  MonthsEmployed: number;
  YearsExperience: number;
  NumCreditLines: number;
  InterestRate: number;
  LoanTerm: number;
  EmploymentType: EmploymentType;
  Gender: Gender;
  Education: Education;
  City: string;
}

export interface PredictionResponse {
  prediction: 0 | 1;
  probability_approve: number;
  probability_default: number;
  risk_tier: RiskTier;
  decision: Decision;
  threshold_used: number;
  model_version: string;
  timestamp: string;
}

export interface ScoredLoanApplication extends LoanApplication, PredictionResponse {}

export interface BatchPredictionResponse {
  predictions: PredictionResponse[];
  total: number;
  approved: number;
  declined: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_path: string;
  threshold: number;
  timestamp: string;
}

export interface ModelInfoResponse {
  model_type: string;
  challenger: string;
  framework: string;
  version: string;
  decision_threshold: number;
  threshold_derivation: string;
  regulatory_context: string[];
  features: {
    numerical: string[];
    categorical: string[];
  };
}

export interface DashboardMetrics {
  totalApplications: number;
  approvedLoans: number;
  declinedLoans: number;
  averageDefaultProbability: number;
}
