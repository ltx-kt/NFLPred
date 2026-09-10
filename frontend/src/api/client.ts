import { useQuery } from "@tanstack/react-query";
import type {
  Calibration,
  GameDetail,
  Health,
  Index,
  Members,
  Performance,
  Week,
} from "./types";

// Same-origin in production (FastAPI serves dist/); the Vite dev server proxies
// /api to :8000. Either way the base is just "/api".
const BASE = "/api";

class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

const qs = (params: Record<string, string | number | undefined>): string => {
  const parts = Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== "")
    .map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`);
  return parts.length ? `?${parts.join("&")}` : "";
};

export { ApiError };

export const useIndex = () =>
  useQuery({ queryKey: ["index"], queryFn: () => get<Index>("/index") });

export const useHealth = () =>
  useQuery({ queryKey: ["health"], queryFn: () => get<Health>("/health") });

export const useWeek = (season: number | undefined, week: number | undefined) =>
  useQuery({
    queryKey: ["week", season, week],
    queryFn: () => get<Week>(`/weeks/${season}/${week}`),
    enabled: season !== undefined && week !== undefined,
  });

export const useGame = (gameId: string | undefined, modelVersion?: string) =>
  useQuery({
    queryKey: ["game", gameId, modelVersion],
    queryFn: () => get<GameDetail>(`/games/${gameId}${qs({ model_version: modelVersion })}`),
    enabled: !!gameId,
  });

export const usePerformance = (config?: string, season?: number) =>
  useQuery({
    queryKey: ["performance", config, season],
    queryFn: () => get<Performance>(`/performance${qs({ config, season })}`),
  });

export const useCalibration = (config?: string, season?: number, bins = 10) =>
  useQuery({
    queryKey: ["calibration", config, season, bins],
    queryFn: () => get<Calibration>(`/calibration${qs({ config, season, bins })}`),
  });

export const useMembers = (window = 4, config?: string) =>
  useQuery({
    queryKey: ["members", window, config],
    queryFn: () => get<Members>(`/members${qs({ window, config })}`),
  });
