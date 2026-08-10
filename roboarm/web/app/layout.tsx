import type { Metadata } from "next";
import type { ReactNode } from "react";

import "./globals.css";

export const metadata: Metadata = {
  title: "Godel-Kolmogorov machine · RoboArm",
  description:
    "Replay viewer for a standalone Godel-Kolmogorov machine RoboArm campaign.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
