import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { lazy, Suspense } from "react";
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Spinner } from "./components/Feedback";
import { About } from "./pages/About";
import { GameDetail } from "./pages/GameDetail";
import { History } from "./pages/History";
import { ThisWeek } from "./pages/ThisWeek";

// The Model page is the only Recharts consumer - split it so the other routes
// don't ship the charting bundle.
const Model = lazy(() => import("./pages/Model").then((m) => ({ default: m.Model })));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

const router = createBrowserRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <ThisWeek /> },
      { path: "history", element: <History /> },
      {
        path: "model",
        element: (
          <Suspense fallback={<Spinner label="Loading diagnostics" />}>
            <Model />
          </Suspense>
        ),
      },
      { path: "about", element: <About /> },
      { path: "game/:gameId", element: <GameDetail /> },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
]);

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
