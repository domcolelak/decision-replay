import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Decision Replay",
  description:
    "Institutional memory for business decisions: what was decided, why, and how it turned out.",
};

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/decisions", label: "Decisions" },
  { href: "/outcomes", label: "Outcomes due" },
  { href: "/templates", label: "Templates" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <aside className="sidebar">
            <div className="brand">
              <span className="brand-mark" aria-hidden />
              <span>Decision Replay</span>
            </div>
            <nav>
              {NAV.map((item) => (
                <Link key={item.href} href={item.href}>
                  {item.label}
                </Link>
              ))}
            </nav>
            <p className="sidebar-note">
              Precedents are ranked by similarity, not by whether they worked. Every
              component of the ranking is shown, so you can disagree with it.
            </p>
          </aside>
          <main className="content">{children}</main>
        </div>
      </body>
    </html>
  );
}
