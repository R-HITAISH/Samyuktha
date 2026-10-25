import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SAMYUKTA — Intelligent Hybrid Weather Forecast Fusion',
  description: 'Regime-Skill-Conditioned Mixture of Experts for adaptive multi-model weather forecast blending. SIH26081 — Ministry of Earth Sciences / NCMRWF.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="bg-[#0a0e1a] text-gray-100 antialiased">
        {children}
      </body>
    </html>
  );
}
